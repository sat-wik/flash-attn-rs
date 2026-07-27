//! AVX2 + FMA version of the tiled kernel with two upgrades over the baseline
//! SIMD path:
//!   1. Causal block-skipping (same as the portable tiled kernel).
//!   2. A vectorized `exp` (see `vexp`) so the softmax exponentials are computed
//!      8 lanes at a time instead of one scalar libm call per score. In the
//!      baseline kernel that scalar `exp` was the main thing capping the speedup.
//!
//! Runtime dispatch: AVX2+FMA path if the CPU has it, else the portable tiled
//! kernel. You never assume the target has the ISA.

use crate::{tiled, Mat};

pub fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { attention_avx2(q, k, v, causal) };
        }
    }
    tiled::attention(q, k, v, causal)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attention_avx2(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    use crate::vexp;
    use std::arch::x86_64::*;

    let n = q.rows;
    let d = q.cols;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = Mat::zeros(n, d);
    let mut m = vec![f32::NEG_INFINITY; n];
    let mut l = vec![0.0f32; n];

    const BLOCK: usize = tiled::BLOCK;

    let mut q0 = 0;
    while q0 < n {
        let q1 = (q0 + BLOCK).min(n);

        let mut kv0 = 0;
        while kv0 < n {
            let kv1 = (kv0 + BLOCK).min(n);
            if causal && kv0 > q1 - 1 {
                break;
            }

            for i in q0..q1 {
                let qi = q.row(i);
                let oi_ptr = out.row_mut(i).as_mut_ptr();

                let jhi = if causal { kv1.min(i + 1) } else { kv1 };
                if jhi <= kv0 {
                    continue;
                }

                // Score against the (possibly truncated) KV block.
                let mut block_scores = [0.0f32; BLOCK];
                let mut block_max = f32::NEG_INFINITY;
                for (bj, j) in (kv0..jhi).enumerate() {
                    let acc = dot_avx2(qi, k.row(j)) * scale;
                    block_scores[bj] = acc;
                    if acc > block_max {
                        block_max = acc;
                    }
                }

                let m_old = m[i];
                let m_new = m_old.max(block_max);
                let correction = if m_old == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m_old - m_new).exp()
                };

                let oi = std::slice::from_raw_parts_mut(oi_ptr, d);
                if correction != 1.0 {
                    scale_avx2(oi, correction);
                }
                l[i] *= correction;

                // Vectorized exp over the block scores, 8 at a time.
                let valid = jhi - kv0;
                let mut probs = [0.0f32; BLOCK];
                let m_vec = _mm256_set1_ps(m_new);
                let mut bj = 0;
                while bj + 8 <= valid {
                    let s = _mm256_loadu_ps(block_scores.as_ptr().add(bj));
                    let e = vexp::exp8(_mm256_sub_ps(s, m_vec));
                    _mm256_storeu_ps(probs.as_mut_ptr().add(bj), e);
                    bj += 8;
                }
                while bj < valid {
                    probs[bj] = (block_scores[bj] - m_new).exp();
                    bj += 1;
                }

                // Accumulate l and out = out + sum_j probs_j * V_j.
                let mut lsum = 0.0f32;
                for (bj, j) in (kv0..jhi).enumerate() {
                    let p = probs[bj];
                    lsum += p;
                    axpy_avx2(oi, v.row(j), p);
                }
                l[i] += lsum;
                m[i] = m_new;
            }
            kv0 = kv1;
        }
        q0 = q1;
    }

    for (i, &li) in l.iter().enumerate() {
        let denom = if li == 0.0 { 1.0 } else { li };
        scale_avx2(out.row_mut(i), 1.0 / denom);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut t = 0;
    while t + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(t));
        let vb = _mm256_loadu_ps(b.as_ptr().add(t));
        acc = _mm256_fmadd_ps(va, vb, acc);
        t += 8;
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    let mut total = _mm_cvtss_f32(s);
    while t < n {
        total += a[t] * b[t];
        t += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(out: &mut [f32], v: &[f32], p: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let vp = _mm256_set1_ps(p);
    let mut t = 0;
    while t + 8 <= n {
        let vo = _mm256_loadu_ps(out.as_ptr().add(t));
        let vv = _mm256_loadu_ps(v.as_ptr().add(t));
        _mm256_storeu_ps(out.as_mut_ptr().add(t), _mm256_fmadd_ps(vp, vv, vo));
        t += 8;
    }
    while t < n {
        out[t] += p * v[t];
        t += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_avx2(out: &mut [f32], s: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let vs = _mm256_set1_ps(s);
    let mut t = 0;
    while t + 8 <= n {
        let vo = _mm256_loadu_ps(out.as_ptr().add(t));
        _mm256_storeu_ps(out.as_mut_ptr().add(t), _mm256_mul_ps(vo, vs));
        t += 8;
    }
    while t < n {
        out[t] *= s;
        t += 1;
    }
}

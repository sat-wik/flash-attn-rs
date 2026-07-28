//! AVX-512 path — 16 f32 lanes per instruction, double the AVX2 width.
//!
//! AVX-512 intrinsics (`_mm512_*`) were stabilized in Rust 1.89, above this
//! crate's declared MSRV, so the module is gated on an `avx512` cfg you opt into
//! explicitly. No nightly needed on a current toolchain:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native --cfg avx512" cargo build --release
//! ```
//!
//! Gating it this way keeps the default build on the lower floor while the code
//! stays present, reviewable and covered — CI lints and tests it under the cfg
//! on stable. `simd::attention` adds an `is_x86_feature_detected!("avx512f")`
//! arm ahead of the AVX2 one when the cfg is active.
//!
//! Design note: the interesting measurement here isn't just "2× the lanes." The
//! per-block reduction and the scalar softmax tail don't get wider, so the
//! expected gain over AVX2 is sub-2×, and on parts that down-clock under heavy
//! AVX-512 load it can even regress. That crossover — where wider vectors stop
//! paying because of frequency throttling and reduction overhead — is the thing
//! worth plotting on the target silicon.

#![cfg(all(target_arch = "x86_64", avx512))]
// The `_mm512_*` intrinsics are stable since 1.89, above the crate's declared
// MSRV of 1.80. That gap is the entire reason this module sits behind an opt-in
// cfg: the default build keeps the lower floor, and anyone passing
// `--cfg avx512` is opting into a newer toolchain at the same time.
#![allow(clippy::incompatible_msrv)]

use crate::Mat;

/// 16-wide dot product with FMA and a single-instruction horizontal reduction.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F (e.g. via
/// `is_x86_feature_detected!`). Executing this without it is undefined behaviour.
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut t = 0;
    while t + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(t));
        let vb = _mm512_loadu_ps(b.as_ptr().add(t));
        acc = _mm512_fmadd_ps(va, vb, acc);
        t += 16;
    }
    let mut total = _mm512_reduce_add_ps(acc); // hardware horizontal add
    while t < n {
        total += a[t] * b[t];
        t += 1;
    }
    total
}

/// out += p * v, 16-wide.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn axpy_avx512(out: &mut [f32], v: &[f32], p: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let vp = _mm512_set1_ps(p);
    let mut t = 0;
    while t + 16 <= n {
        let vo = _mm512_loadu_ps(out.as_ptr().add(t));
        let vv = _mm512_loadu_ps(v.as_ptr().add(t));
        _mm512_storeu_ps(out.as_mut_ptr().add(t), _mm512_fmadd_ps(vp, vv, vo));
        t += 16;
    }
    while t < n {
        out[t] += p * v[t];
        t += 1;
    }
}

/// `out += sum_j probs[j] * V[j]` over one KV block, with the output row held in
/// registers across the block. The 16-wide counterpart of `simd::accumulate_avx2`
/// — same loop interchange, same reason: one `axpy` per key reloads and restores
/// the whole output row every time, and that traffic is loop structure rather
/// than arithmetic. 64-float chunks give four zmm accumulators.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn accumulate_avx512(oi: &mut [f32], v: &Mat, kv0: usize, jhi: usize, probs: &[f32]) {
    use std::arch::x86_64::*;
    let d = oi.len();
    let mut base = 0;

    while base + 64 <= d {
        let mut a0 = _mm512_loadu_ps(oi.as_ptr().add(base));
        let mut a1 = _mm512_loadu_ps(oi.as_ptr().add(base + 16));
        let mut a2 = _mm512_loadu_ps(oi.as_ptr().add(base + 32));
        let mut a3 = _mm512_loadu_ps(oi.as_ptr().add(base + 48));

        for (bj, j) in (kv0..jhi).enumerate() {
            let vp = _mm512_set1_ps(probs[bj]);
            let vr = v.row(j).as_ptr().add(base);
            a0 = _mm512_fmadd_ps(vp, _mm512_loadu_ps(vr), a0);
            a1 = _mm512_fmadd_ps(vp, _mm512_loadu_ps(vr.add(16)), a1);
            a2 = _mm512_fmadd_ps(vp, _mm512_loadu_ps(vr.add(32)), a2);
            a3 = _mm512_fmadd_ps(vp, _mm512_loadu_ps(vr.add(48)), a3);
        }

        _mm512_storeu_ps(oi.as_mut_ptr().add(base), a0);
        _mm512_storeu_ps(oi.as_mut_ptr().add(base + 16), a1);
        _mm512_storeu_ps(oi.as_mut_ptr().add(base + 32), a2);
        _mm512_storeu_ps(oi.as_mut_ptr().add(base + 48), a3);
        base += 64;
    }

    if base < d {
        for (bj, j) in (kv0..jhi).enumerate() {
            axpy_avx512(&mut oi[base..], &v.row(j)[base..], probs[bj]);
        }
    }
}

/// out *= s, 16-wide.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn scale_avx512(out: &mut [f32], s: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let vs = _mm512_set1_ps(s);
    let mut t = 0;
    while t + 16 <= n {
        let vo = _mm512_loadu_ps(out.as_ptr().add(t));
        _mm512_storeu_ps(out.as_mut_ptr().add(t), _mm512_mul_ps(vo, vs));
        t += 16;
    }
    while t < n {
        out[t] *= s;
        t += 1;
    }
}

/// The full kernel: structurally identical to `simd::attention_avx2` — same
/// online-softmax bookkeeping, same causal block-skipping, same vectorized-exp
/// pattern — with every 8-wide primitive swapped for its 16-wide counterpart.
///
/// Keeping the two bodies parallel rather than abstracting over lane width is
/// deliberate: the AVX2 and AVX-512 intrinsic families share no trait, and any
/// abstraction that hid the difference would also hide the thing the comparison
/// is meant to expose.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    use crate::vexp;
    use std::arch::x86_64::*;

    let n = q.rows;
    let d = q.cols;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = Mat::zeros(n, d);
    let mut m = vec![f32::NEG_INFINITY; n];
    let mut l = vec![0.0f32; n];

    const BLOCK: usize = crate::tiled::BLOCK;

    let mut q0 = 0;
    while q0 < n {
        let q1 = (q0 + BLOCK).min(n);

        let mut kv0 = 0;
        while kv0 < n {
            let kv1 = (kv0 + BLOCK).min(n);
            // Causal block skip: this KV block sits entirely after the query
            // block, and so does every later one.
            if causal && kv0 > q1 - 1 {
                break;
            }

            for i in q0..q1 {
                let qi = q.row(i);
                let jhi = if causal { kv1.min(i + 1) } else { kv1 };
                if jhi <= kv0 {
                    continue;
                }

                let mut block_scores = [0.0f32; BLOCK];
                let mut block_max = f32::NEG_INFINITY;
                for (bj, j) in (kv0..jhi).enumerate() {
                    let acc = dot_avx512(qi, k.row(j)) * scale;
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

                if correction != 1.0 {
                    scale_avx512(out.row_mut(i), correction);
                }
                l[i] *= correction;

                // Vectorized exp over the block scores, 16 at a time.
                let valid = jhi - kv0;
                let mut probs = [0.0f32; BLOCK];
                let m_vec = _mm512_set1_ps(m_new);
                let mut bj = 0;
                while bj + 16 <= valid {
                    let s = _mm512_loadu_ps(block_scores.as_ptr().add(bj));
                    let e = vexp::exp16(_mm512_sub_ps(s, m_vec));
                    _mm512_storeu_ps(probs.as_mut_ptr().add(bj), e);
                    bj += 16;
                }
                while bj < valid {
                    probs[bj] = (block_scores[bj] - m_new).exp();
                    bj += 1;
                }

                // out += sum_j probs_j * V_j, and accumulate the normalizer.
                let lsum: f32 = probs[..valid].iter().sum();
                accumulate_avx512(out.row_mut(i), v, kv0, jhi, &probs);
                l[i] += lsum;
                m[i] = m_new;
            }
            kv0 = kv1;
        }
        q0 = q1;
    }

    for (i, &li) in l.iter().enumerate() {
        let denom = if li == 0.0 { 1.0 } else { li };
        scale_avx512(out.row_mut(i), 1.0 / denom);
    }
    out
}

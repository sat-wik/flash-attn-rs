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
        // Widest first. This arm only exists when built with `--cfg avx512`;
        // the feature test still runs, so a binary built with the cfg stays
        // correct on a CPU that lacks AVX-512 and simply falls through.
        #[cfg(avx512)]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { crate::avx512::attention(q, k, v, causal) };
            }
        }
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
                // Four keys per horizontal reduction. The reduction at the tail
                // of a dot product does not vectorize and was the largest single
                // item in the gap to peak; scoring four keys against the same
                // query lets one reduction serve all four.
                let mut block_scores = [0.0f32; BLOCK];
                let mut block_max = f32::NEG_INFINITY;
                let mut bj = 0usize;
                let mut j = kv0;
                while j + 4 <= jhi {
                    let four = dot4_avx2(qi, k.row(j), k.row(j + 1), k.row(j + 2), k.row(j + 3));
                    for (o, &raw) in four.iter().enumerate() {
                        let s = raw * scale;
                        block_scores[bj + o] = s;
                        if s > block_max {
                            block_max = s;
                        }
                    }
                    bj += 4;
                    j += 4;
                }
                while j < jhi {
                    let s = dot_avx2(qi, k.row(j)) * scale;
                    block_scores[bj] = s;
                    if s > block_max {
                        block_max = s;
                    }
                    bj += 1;
                    j += 1;
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
                let lsum: f32 = probs[..valid].iter().sum();
                accumulate_avx2(oi, v, kv0, jhi, &probs);
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

/// Four dot products against a shared `a`, sharing one horizontal reduction.
///
/// A single 8-wide dot ends with `extractf128 + add + hadd + hadd`, which is
/// serial, does not use the FMA units, and costs the same whether the vector
/// body was 8 elements or 64. At `d = 64` that tail runs once per query-key
/// pair and is a large part of why the kernel sits well under peak.
///
/// Four accumulators collapse in three `hadd`s plus one 128-bit fold, because
/// `hadd` interleaves two sources: `hadd(hadd(a,b), hadd(c,d))` leaves the four
/// partial sums in known lanes, and folding the halves finishes all four at
/// once. Roughly a quarter of the reduction work per key.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut t = 0;
    while t + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(t));
        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0.as_ptr().add(t)), acc0);
        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1.as_ptr().add(t)), acc1);
        acc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2.as_ptr().add(t)), acc2);
        acc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3.as_ptr().add(t)), acc3);
        t += 8;
    }

    // hadd(a,b) yields [a01 a23 b01 b23 | a45 a67 b45 b67], so nesting it once
    // more puts the four full sums in lanes 0..3 of each 128-bit half.
    let ab = _mm256_hadd_ps(acc0, acc1);
    let cd = _mm256_hadd_ps(acc2, acc3);
    let abcd = _mm256_hadd_ps(ab, cd);
    let folded = _mm_add_ps(_mm256_castps256_ps128(abcd), _mm256_extractf128_ps(abcd, 1));

    let mut out = [0.0f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), folded);

    while t < n {
        let x = a[t];
        out[0] += x * b0[t];
        out[1] += x * b1[t];
        out[2] += x * b2[t];
        out[3] += x * b3[t];
        t += 1;
    }
    out
}

/// `out += Σ_j probs[j] · V[j]` over one KV block, holding the output row in
/// registers for the whole block.
///
/// The obvious loop is one `axpy` per key, and that reloads and restores the
/// entire output row every time: at `d = 64` it is 8 loads and 8 stores of `out`
/// per key, on top of the 8 loads of `V` that are actually unavoidable. The
/// output traffic is pure loop structure — the same 64 floats going out to L1
/// and back for every key in the block.
///
/// Interchanging the loops fixes it. Walking `out` in 32-float chunks and
/// putting the key loop *inside* means four accumulators stay in registers
/// across every key in the block, and `out` is touched once at each end. Only
/// `V` is streamed, which is the part that genuinely has to move.
///
/// Chunk width is 32 rather than 64 so `d = 32` benefits too and register
/// pressure stays low: four accumulators plus a broadcast and a load is six of
/// sixteen ymm registers. `V` is re-read once per chunk, but a block of it is
/// 16 KB at the default tile size and sits in L1, so that costs far less than
/// the output traffic it removes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_avx2(oi: &mut [f32], v: &Mat, kv0: usize, jhi: usize, probs: &[f32]) {
    use std::arch::x86_64::*;
    let d = oi.len();
    let mut base = 0;

    while base + 32 <= d {
        let mut a0 = _mm256_loadu_ps(oi.as_ptr().add(base));
        let mut a1 = _mm256_loadu_ps(oi.as_ptr().add(base + 8));
        let mut a2 = _mm256_loadu_ps(oi.as_ptr().add(base + 16));
        let mut a3 = _mm256_loadu_ps(oi.as_ptr().add(base + 24));

        for (bj, j) in (kv0..jhi).enumerate() {
            let vp = _mm256_set1_ps(probs[bj]);
            let vr = v.row(j).as_ptr().add(base);
            a0 = _mm256_fmadd_ps(vp, _mm256_loadu_ps(vr), a0);
            a1 = _mm256_fmadd_ps(vp, _mm256_loadu_ps(vr.add(8)), a1);
            a2 = _mm256_fmadd_ps(vp, _mm256_loadu_ps(vr.add(16)), a2);
            a3 = _mm256_fmadd_ps(vp, _mm256_loadu_ps(vr.add(24)), a3);
        }

        _mm256_storeu_ps(oi.as_mut_ptr().add(base), a0);
        _mm256_storeu_ps(oi.as_mut_ptr().add(base + 8), a1);
        _mm256_storeu_ps(oi.as_mut_ptr().add(base + 16), a2);
        _mm256_storeu_ps(oi.as_mut_ptr().add(base + 24), a3);
        base += 32;
    }

    // Whatever is left over keeps the per-key form; at d = 64 there is none.
    if base < d {
        for (bj, j) in (kv0..jhi).enumerate() {
            axpy_avx2(&mut oi[base..], &v.row(j)[base..], probs[bj]);
        }
    }
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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::filled;

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    /// `dot4_avx2` folds four accumulators through a nest of `hadd`s that relies
    /// on which lanes each one leaves its partial sums in. That is easy to get
    /// subtly wrong — a transposed pair would still produce plausible-looking
    /// attention output — so check it against the one-at-a-time version directly
    /// rather than trusting the end-to-end test to notice.
    #[test]
    fn dot4_matches_dot() {
        if !has_avx2() {
            eprintln!("SKIPPED: no AVX2+FMA on this CPU");
            return;
        }
        // Lengths either side of the 8-wide body: exact multiples, a short
        // vector, and ragged tails.
        for &len in &[1usize, 7, 8, 9, 15, 16, 63, 64, 65, 128] {
            let a = filled(1, len, 1);
            let b = filled(4, len, 2);
            let got = unsafe { dot4_avx2(a.row(0), b.row(0), b.row(1), b.row(2), b.row(3)) };
            for (lane, &g) in got.iter().enumerate() {
                let want = unsafe { dot_avx2(a.row(0), b.row(lane)) };
                let diff = (g - want).abs();
                let tol = 1e-4 * want.abs().max(1.0);
                assert!(
                    diff < tol,
                    "len={len} lane={lane}: dot4 {g} vs dot {want} (diff {diff:e})"
                );
            }
        }
    }

    /// The four lanes must stay in key order. A test that used the same data for
    /// every row would pass even if the fold shuffled them.
    #[test]
    fn dot4_preserves_lane_order() {
        if !has_avx2() {
            return;
        }
        let a = crate::Mat {
            rows: 1,
            cols: 8,
            data: vec![1.0; 8],
        };
        // Each row sums to a distinct value, so a permuted fold is visible.
        let mut b = crate::Mat::zeros(4, 8);
        for r in 0..4 {
            for t in 0..8 {
                b.row_mut(r)[t] = if t == 0 { (r + 1) as f32 } else { 0.0 };
            }
        }
        let got = unsafe { dot4_avx2(a.row(0), b.row(0), b.row(1), b.row(2), b.row(3)) };
        assert_eq!(got, [1.0, 2.0, 3.0, 4.0], "lanes came back out of order");
    }

    /// `accumulate_avx2` interchanges the loops and keeps the output row in
    /// registers, which changes both the traversal order and the summation
    /// order. Check it against the obvious per-key form.
    #[test]
    fn accumulate_matches_per_key_axpy() {
        if !has_avx2() {
            return;
        }
        // d values chosen to hit every path: below one 32-float chunk, exactly
        // one, one plus a ragged tail, and two whole chunks.
        for &d in &[8usize, 17, 32, 49, 64, 96] {
            for &(kv0, jhi) in &[(0usize, 1usize), (0, 7), (0, 64), (3, 20), (11, 12)] {
                let v = filled(jhi.max(1), d, 5);
                let probs: Vec<f32> = (0..64).map(|i| 0.25 + (i as f32) * 0.01).collect();
                let start = filled(1, d, 9);

                let mut got = start.row(0).to_vec();
                unsafe { accumulate_avx2(&mut got, &v, kv0, jhi, &probs) };

                let mut want = start.row(0).to_vec();
                for (bj, j) in (kv0..jhi).enumerate() {
                    unsafe { axpy_avx2(&mut want, v.row(j), probs[bj]) };
                }

                for (t, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    let diff = (g - w).abs();
                    let tol = 1e-4 * w.abs().max(1.0);
                    assert!(
                        diff < tol,
                        "d={d} kv0={kv0} jhi={jhi} t={t}: {g} vs {w} (diff {diff:e})"
                    );
                }
            }
        }
    }
}

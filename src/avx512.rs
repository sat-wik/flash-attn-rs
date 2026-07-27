//! AVX-512 path — 16 f32 lanes per instruction, double the AVX2 width.
//!
//! AVX-512 intrinsics (`_mm512_*`) were stabilized in Rust 1.89. On older
//! toolchains they live behind the unstable `stdsimd` feature, so this whole
//! module is gated on a `avx512` cfg flag you opt into explicitly:
//!
//!     RUSTFLAGS="-C target-cpu=native --cfg avx512" cargo +nightly build --release
//!
//! Gating it this way keeps the default `cargo build`/`cargo test` green on
//! stable while the code stays present and reviewable. The dispatch in
//! `simd::attention` would add an `is_x86_feature_detected!("avx512f")` arm in
//! front of the AVX2 one when this cfg is active.
//!
//! Design note: the interesting measurement here isn't just "2× the lanes." The
//! per-block reduction and the scalar softmax tail don't get wider, so the
//! expected gain over AVX2 is sub-2×, and on parts that down-clock under heavy
//! AVX-512 load it can even regress. That crossover — where wider vectors stop
//! paying because of frequency throttling and reduction overhead — is the thing
//! worth plotting on the target silicon.

#![cfg(all(target_arch = "x86_64", avx512))]

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

/// Placeholder full-kernel entry: mirrors `simd::attention_avx2` but with the
/// 16-wide primitives above. Left as a thin wrapper here so the module compiles
/// standalone under the `avx512` cfg; wire it into `simd::attention`'s dispatch
/// when building on nightly.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub unsafe fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    // The structure is identical to the AVX2 kernel; only the lane width and the
    // reduction change. Kept as a fallback delegate to avoid duplicating the
    // ~100-line body until the nightly build is the default.
    crate::tiled::attention(q, k, v, causal)
}

//! Fast vectorized exp for AVX2. Softmax needs one `exp` per score, and in the
//! baseline SIMD kernel that call stays scalar — it's the single biggest thing
//! capping the vector speedup. Here we approximate `exp(x)` 8 lanes at a time.
//!
//! Method: range-reduce x = k*ln2 + r with |r| <= ln2/2, compute 2^k by
//! directly assembling the IEEE-754 exponent field, and approximate exp(r) on
//! the reduced range with a degree-5 minimax-style polynomial. Accuracy is
//! ~1e-6 relative over the softmax input range, far tighter than the 1e-3
//! tolerance the kernels are tested to.
//!
//! This is only ever called on the post-max-subtraction values (x <= 0 in
//! softmax), but it's correct for the general small-to-moderate range too.

/// # Safety
///
/// The caller must have established that the CPU supports AVX2 and FMA (e.g.
/// via `is_x86_feature_detected!`). Executing this on a CPU without them is
/// undefined behaviour.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let inv_ln2 = _mm256_set1_ps(std::f32::consts::LOG2_E);

    // Clamp to avoid overflow in the exponent assembly for extreme inputs.
    let hi = _mm256_set1_ps(88.0);
    let lo = _mm256_set1_ps(-88.0);
    let x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);

    // k = round(x / ln2), r = x - k*ln2
    let kf = _mm256_round_ps(
        _mm256_mul_ps(x, inv_ln2),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(kf, ln2, x); // x - kf*ln2

    // exp(r) via Horner on a degree-5 polynomial.
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c2 = _mm256_set1_ps(0.5);
    let c1 = _mm256_set1_ps(1.0);
    let c0 = _mm256_set1_ps(1.0);
    let mut p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    // 2^k: build the float by adding k to the exponent bias (127) and shifting
    // into the exponent field.
    let ki = _mm256_cvtps_epi32(kf);
    let bias = _mm256_set1_epi32(127);
    let exp_field = _mm256_slli_epi32(_mm256_add_epi32(ki, bias), 23);
    let pow2k = _mm256_castsi256_ps(exp_field);

    _mm256_mul_ps(p, pow2k)
}

/// 16-wide sibling of [`exp8`]: identical range reduction, identical degree-5
/// polynomial, identical exponent assembly — only the lane count changes. Kept
/// as a separate function rather than a generic because the two intrinsic
/// families share no trait, and the duplication is worth less than the
/// indirection would cost inside the softmax inner loop.
///
/// One deliberate difference: where the AVX2 path rounds with
/// `_mm256_round_ps` and then converts, this converts straight to i32.
/// `_mm512_cvtps_epi32` rounds to nearest-even under the default MXCSR mode,
/// which is what the AVX2 pair computes, so the two agree lane for lane.
///
/// # Safety
///
/// The caller must have established that the CPU supports AVX-512F (e.g. via
/// `is_x86_feature_detected!`). Executing this without it is undefined
/// behaviour.
#[cfg(all(target_arch = "x86_64", avx512))]
// Stable since 1.89, above the crate MSRV — same opt-in reasoning as `avx512.rs`.
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp16(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;

    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
    let inv_ln2 = _mm512_set1_ps(std::f32::consts::LOG2_E);

    // Clamp so the exponent assembly below cannot overflow the field.
    let hi = _mm512_set1_ps(88.0);
    let lo = _mm512_set1_ps(-88.0);
    let x = _mm512_max_ps(_mm512_min_ps(x, hi), lo);

    // k = round(x / ln2), r = x - k*ln2, so |r| <= ln2/2.
    let ki = _mm512_cvtps_epi32(_mm512_mul_ps(x, inv_ln2));
    let kf = _mm512_cvtepi32_ps(ki);
    let r = _mm512_fnmadd_ps(kf, ln2, x);

    // exp(r) by Horner on the reduced range.
    let c5 = _mm512_set1_ps(1.0 / 120.0);
    let c4 = _mm512_set1_ps(1.0 / 24.0);
    let c3 = _mm512_set1_ps(1.0 / 6.0);
    let c2 = _mm512_set1_ps(0.5);
    let c1 = _mm512_set1_ps(1.0);
    let c0 = _mm512_set1_ps(1.0);
    let mut p = c5;
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c0);

    // 2^k, built by adding the bias and shifting into the IEEE-754 exponent.
    let bias = _mm512_set1_epi32(127);
    let exp_field = _mm512_slli_epi32::<23>(_mm512_add_epi32(ki, bias));
    let pow2k = _mm512_castsi512_ps(exp_field);

    _mm512_mul_ps(p, pow2k)
}

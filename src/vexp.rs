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

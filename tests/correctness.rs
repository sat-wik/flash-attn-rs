use flash_attn_rs::{filled, max_abs_diff, naive, simd, tiled};

#[test]
fn tiled_matches_naive() {
    for &causal in &[false, true] {
        for &(n, d) in &[(1, 8), (7, 16), (64, 32), (130, 64), (200, 48)] {
            let q = filled(n, d, 1);
            let k = filled(n, d, 2);
            let v = filled(n, d, 3);
            let a = naive::attention(&q, &k, &v, causal);
            let b = tiled::attention(&q, &k, &v, causal);
            let diff = max_abs_diff(&a, &b);
            assert!(diff < 1e-4, "tiled causal={causal} n={n} d={d} diff={diff}");
        }
    }
}

#[test]
fn simd_matches_naive() {
    for &causal in &[false, true] {
        for &(n, d) in &[(1, 8), (7, 15), (64, 32), (130, 64), (200, 49)] {
            let q = filled(n, d, 4);
            let k = filled(n, d, 5);
            let v = filled(n, d, 6);
            let a = naive::attention(&q, &k, &v, causal);
            let b = simd::attention(&q, &k, &v, causal);
            let diff = max_abs_diff(&a, &b);
            assert!(diff < 2e-3, "simd causal={causal} n={n} d={d} diff={diff}");
        }
    }
}

/// Only compiled with `--cfg avx512`, and only asserts anything on a CPU that
/// actually has AVX-512F. On a runner without it the kernel is unreachable —
/// calling it anyway would be undefined behaviour — so the test reports the
/// skip rather than silently passing and implying coverage it did not get.
#[cfg(avx512)]
#[test]
fn avx512_matches_naive() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("SKIPPED: this CPU has no AVX-512F; the kernel was compiled but not run");
        return;
    }
    for &causal in &[false, true] {
        for &(n, d) in &[(1, 8), (7, 15), (64, 32), (130, 64), (200, 49)] {
            let q = filled(n, d, 7);
            let k = filled(n, d, 8);
            let v = filled(n, d, 9);
            let a = naive::attention(&q, &k, &v, causal);
            let b = unsafe { flash_attn_rs::avx512::attention(&q, &k, &v, causal) };
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 2e-3,
                "avx512 causal={causal} n={n} d={d} diff={diff}"
            );
        }
    }
}

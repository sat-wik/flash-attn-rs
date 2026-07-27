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

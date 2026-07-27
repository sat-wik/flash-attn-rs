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

#[test]
fn multihead_matches_per_head() {
    use flash_attn_rs::multihead;
    for &causal in &[false, true] {
        for &(heads, n, d) in &[(1usize, 33usize, 16usize), (4, 130, 64), (7, 64, 32)] {
            let q = multihead::filled_heads(heads, n, d, 11);
            let k = multihead::filled_heads(heads, n, d, 22);
            let v = multihead::filled_heads(heads, n, d, 33);

            let serial = multihead::attention(&q, &k, &v, causal);
            assert_eq!(serial.len(), heads);

            // Each head must equal the single-head kernel on that head's data.
            for h in 0..heads {
                let one = simd::attention(&q[h], &k[h], &v[h], causal);
                assert_eq!(max_abs_diff(&serial[h], &one), 0.0);
            }

            // And splitting across threads must not change a single value,
            // whatever the chunking works out to.
            for threads in [2usize, 3, 8] {
                let par = multihead::attention_parallel(&q, &k, &v, causal, threads);
                assert_eq!(par.len(), heads);
                for h in 0..heads {
                    let diff = max_abs_diff(&serial[h], &par[h]);
                    assert_eq!(diff, 0.0, "threads={threads} head={h} diff={diff}");
                }
            }
        }
    }
}

#[test]
fn block_sizes_agree() {
    use flash_attn_rs::tiled;
    for &causal in &[false, true] {
        for &(n, d) in &[(65usize, 32usize), (200, 64)] {
            let q = filled(n, d, 4);
            let k = filled(n, d, 5);
            let v = filled(n, d, 6);
            let reference = naive::attention(&q, &k, &v, causal);
            macro_rules! check {
                ($($b:literal),*) => {$({
                    let got = tiled::attention_b::<$b>(&q, &k, &v, causal);
                    let diff = max_abs_diff(&reference, &got);
                    assert!(diff < 1e-4, "BLOCK={} n={n} causal={causal} diff={diff}", $b);
                })*};
            }
            check!(16, 32, 64, 128, 256);
        }
    }
}

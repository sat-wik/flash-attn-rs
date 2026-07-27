//! Zero-dependency benchmark. `cargo run --release --bin bench`.
//! Reports median GFLOP/s per kernel across sequence lengths, for both the
//! full (bidirectional) and causal masks.

use flash_attn_rs::{attention_flops, filled, max_abs_diff, naive, simd, tiled};
use std::time::Instant;

fn time_it<F: Fn()>(f: F, iters: usize) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        s.push(t.elapsed().as_secs_f64());
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn run(causal: bool) {
    let d = 64;
    let seqs = [128usize, 256, 512, 1024];
    println!(
        "\n=== {} mask, head_dim d = {d} ===",
        if causal { "causal" } else { "full" }
    );
    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>13}",
        "n", "naive GF/s", "tiled GF/s", "simd GF/s", "simd vs naive"
    );
    for &n in &seqs {
        let q = filled(n, d, 1);
        let k = filled(n, d, 2);
        let v = filled(n, d, 3);
        let flops = attention_flops(n, d, causal);

        let r = naive::attention(&q, &k, &v, causal);
        assert!(max_abs_diff(&r, &tiled::attention(&q, &k, &v, causal)) < 1e-4);
        assert!(max_abs_diff(&r, &simd::attention(&q, &k, &v, causal)) < 2e-3);

        let iters = if n <= 256 { 50 } else { 15 };
        let tn = time_it(
            || {
                naive::attention(&q, &k, &v, causal);
            },
            iters,
        );
        let tt = time_it(
            || {
                tiled::attention(&q, &k, &v, causal);
            },
            iters,
        );
        let ts = time_it(
            || {
                simd::attention(&q, &k, &v, causal);
            },
            iters,
        );
        let g = |t: f64| flops / t / 1e9;
        println!(
            "{:>6}  {:>14.2}  {:>14.2}  {:>14.2}  {:>12.2}x",
            n,
            g(tn),
            g(tt),
            g(ts),
            tn / ts
        );
    }
}

fn main() {
    run(false);
    run(true);
}

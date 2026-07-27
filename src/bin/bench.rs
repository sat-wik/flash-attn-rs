//! Zero-dependency benchmark. `cargo run --release --bin bench`.
//! Reports median GFLOP/s per kernel across sequence lengths, for both the
//! full (bidirectional) and causal masks.
//!
//! `--json` emits the same measurements as machine-readable JSON, including the
//! modelled arithmetic intensity of each point. For the roofline figure itself
//! use the `roofline` binary, which also measures the machine's ceilings.

use flash_attn_rs::{
    attention_flops, filled, max_abs_diff, naive, roofline, simd, tiled, time_stats,
};

fn iters_for(n: usize) -> usize {
    if n <= 256 {
        50
    } else {
        15
    }
}

fn run(causal: bool) {
    let d = 64;
    println!(
        "\n=== {} mask, head_dim d = {d} ===",
        if causal { "causal" } else { "full" }
    );
    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>13}  {:>12}",
        "n", "naive GF/s", "tiled GF/s", "simd GF/s", "simd vs naive", "simd spread"
    );
    for &n in &roofline::SEQ_LENS {
        let q = filled(n, d, 1);
        let k = filled(n, d, 2);
        let v = filled(n, d, 3);
        let flops = attention_flops(n, d, causal);

        let r = naive::attention(&q, &k, &v, causal);
        assert!(max_abs_diff(&r, &tiled::attention(&q, &k, &v, causal)) < 1e-4);
        assert!(max_abs_diff(&r, &simd::attention(&q, &k, &v, causal)) < 2e-3);

        let iters = iters_for(n);
        let tn = time_stats(|| drop(naive::attention(&q, &k, &v, causal)), iters);
        let tt = time_stats(|| drop(tiled::attention(&q, &k, &v, causal)), iters);
        let ts = time_stats(|| drop(simd::attention(&q, &k, &v, causal)), iters);

        let g = |t: f64| flops / t / 1e9;
        println!(
            "{:>6}  {:>14.2}  {:>14.2}  {:>14.2}  {:>12.2}x  {:>11.1}%",
            n,
            g(tn.median),
            g(tt.median),
            g(ts.median),
            tn.median / ts.median,
            ts.spread_pct
        );
    }
}

/// Causal block-skipping, timed directly rather than inferred.
///
/// The GFLOP/s tables above cannot show this effect at all: they divide by a
/// causal-aware FLOP count, which normalizes the halved work back out. The only
/// way to see what the block-skip is worth is to put a stopwatch on the same
/// kernel under both masks and divide, which is what this does.
fn causal_speedup() {
    let d = 64;
    println!("\n=== causal block-skipping, measured wall-clock ===");
    println!(
        "{:>6}  {:>8}  {:>13}  {:>15}  {:>10}",
        "n", "kernel", "full (ms)", "causal (ms)", "speedup"
    );
    for &n in &roofline::SEQ_LENS {
        let q = filled(n, d, 1);
        let k = filled(n, d, 2);
        let v = filled(n, d, 3);
        let iters = iters_for(n);
        for &kernel in &["tiled", "simd"] {
            let time_it = |causal: bool| {
                if kernel == "tiled" {
                    time_stats(|| drop(tiled::attention(&q, &k, &v, causal)), iters)
                } else {
                    time_stats(|| drop(simd::attention(&q, &k, &v, causal)), iters)
                }
            };
            let full = time_it(false);
            let causal = time_it(true);
            println!(
                "{:>6}  {:>8}  {:>13.3}  {:>15.3}  {:>9.2}x",
                n,
                kernel,
                full.median * 1e3,
                causal.median * 1e3,
                full.median / causal.median
            );
        }
    }
    println!("\nIdeal is 2.00x — the causal mask leaves n(n+1)/2 of n^2 query-key pairs.");
}

fn main() {
    if std::env::args().any(|a| a == "--json") {
        let machine = roofline::measure_machine();
        let points = roofline::measure_points();
        print!("{}", roofline::to_json(&machine, &points));
        return;
    }
    run(false);
    run(true);
    causal_speedup();
}

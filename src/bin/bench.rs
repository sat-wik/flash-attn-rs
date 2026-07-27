//! Zero-dependency benchmark. `cargo run --release --bin bench`.
//! Reports median GFLOP/s per kernel across sequence lengths, for both the
//! full (bidirectional) and causal masks.
//!
//! `--json` emits the same measurements as machine-readable JSON, including the
//! modelled arithmetic intensity of each point. For the roofline figure itself
//! use the `roofline` binary, which also measures the machine's ceilings.

use flash_attn_rs::{
    attention_flops, filled, max_abs_diff, naive, roofline, simd, tiled, time_median,
};

fn run(causal: bool) {
    let d = 64;
    println!(
        "\n=== {} mask, head_dim d = {d} ===",
        if causal { "causal" } else { "full" }
    );
    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>13}",
        "n", "naive GF/s", "tiled GF/s", "simd GF/s", "simd vs naive"
    );
    for &n in &roofline::SEQ_LENS {
        let q = filled(n, d, 1);
        let k = filled(n, d, 2);
        let v = filled(n, d, 3);
        let flops = attention_flops(n, d, causal);

        let r = naive::attention(&q, &k, &v, causal);
        assert!(max_abs_diff(&r, &tiled::attention(&q, &k, &v, causal)) < 1e-4);
        assert!(max_abs_diff(&r, &simd::attention(&q, &k, &v, causal)) < 2e-3);

        let iters = if n <= 256 { 50 } else { 15 };
        let tn = time_median(|| drop(naive::attention(&q, &k, &v, causal)), iters);
        let tt = time_median(|| drop(tiled::attention(&q, &k, &v, causal)), iters);
        let ts = time_median(|| drop(simd::attention(&q, &k, &v, causal)), iters);

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
    if std::env::args().any(|a| a == "--json") {
        let machine = roofline::measure_machine();
        let points = roofline::measure_points();
        print!("{}", roofline::to_json(&machine, &points));
        return;
    }
    run(false);
    run(true);
}

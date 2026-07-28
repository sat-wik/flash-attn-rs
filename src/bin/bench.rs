//! Zero-dependency benchmark. `cargo run --release --bin bench`.
//! Reports median GFLOP/s per kernel across sequence lengths, for both the
//! full (bidirectional) and causal masks.
//!
//! `--json` emits the same measurements as machine-readable JSON, including the
//! modelled arithmetic intensity of each point. For the roofline figure itself
//! use the `roofline` binary, which also measures the machine's ceilings.

use flash_attn_rs::{
    attention_flops, filled, max_abs_diff, multihead, naive, roofline, simd, tiled,
    time_interleaved,
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

        // Interleaved so machine drift cannot land entirely in the ratio.
        let iters = iters_for(n);
        let mut f_naive = || drop(naive::attention(&q, &k, &v, causal));
        let mut f_tiled = || drop(tiled::attention(&q, &k, &v, causal));
        let mut f_simd = || drop(simd::attention(&q, &k, &v, causal));
        let t = time_interleaved(&mut [&mut f_naive, &mut f_tiled, &mut f_simd], iters);
        let (tn, tt, ts) = (t[0], t[1], t[2]);

        let g = |t: f64| flops / t / 1e9;
        println!(
            "{:>6}  {:>14.2}  {:>14.2}  {:>14.2}  {:>12.2}x  {:>11.1}%",
            n,
            g(tn.best),
            g(tt.best),
            g(ts.best),
            tn.best / ts.best,
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
            // The two masks are also interleaved: this ratio is the whole
            // measurement, so the pair must see the same machine conditions.
            let mut f_full = || {
                if kernel == "tiled" {
                    drop(tiled::attention(&q, &k, &v, false))
                } else {
                    drop(simd::attention(&q, &k, &v, false))
                }
            };
            let mut f_causal = || {
                if kernel == "tiled" {
                    drop(tiled::attention(&q, &k, &v, true))
                } else {
                    drop(simd::attention(&q, &k, &v, true))
                }
            };
            let t = time_interleaved(&mut [&mut f_full, &mut f_causal], iters);
            let (full, causal) = (t[0], t[1]);
            println!(
                "{:>6}  {:>8}  {:>13.3}  {:>15.3}  {:>9.2}x",
                n,
                kernel,
                full.best * 1e3,
                causal.best * 1e3,
                full.best / causal.best
            );
        }
    }
    println!("\nIdeal is 2.00x — the causal mask leaves n(n+1)/2 of n^2 query-key pairs.");
}

/// Multi-head scaling: the outer loop that makes this a layer.
///
/// Heads are independent, so this is the cheapest parallelism in the whole
/// problem — no shared state, no synchronization beyond the join. What it
/// cannot do is beat the core count, and with fewer heads than cores some
/// cores simply idle, which is where the numbers stop scaling.
fn multihead() {
    let (d, n) = (64usize, 512usize);
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1);
    println!("\n=== multi-head, n = {n}, d = {d}, {cores} logical cores ===");
    if cores < 2 {
        // available_parallelism() honours CPU affinity, so `taskset -c 0` makes
        // this report 1 and attention_parallel falls straight back to the serial
        // path. The two columns would then be the same code measured twice.
        println!(
            "\nOnly one core is available to this process, so there is nothing to\n\
             parallelize across and the two columns below run identical code.\n\
             If you pinned with `taskset -c 0`, re-run this benchmark unpinned\n\
             (or pin to a range, e.g. `taskset -c 0-7`) to measure head scaling.\n\
             The gap between the columns is then a useful read on the noise floor."
        );
    }
    println!(
        "{:>6}  {:>13}  {:>15}  {:>10}  {:>12}",
        "heads", "serial (ms)", "parallel (ms)", "speedup", "vs ideal"
    );

    for &h in &[1usize, 2, 4, 8] {
        let q = multihead::filled_heads(h, n, d, 11);
        let k = multihead::filled_heads(h, n, d, 22);
        let v = multihead::filled_heads(h, n, d, 33);

        // Same reason as everywhere else: these two are being divided.
        let mut f_serial = || drop(multihead::attention(&q, &k, &v, false));
        let mut f_par = || drop(multihead::attention_parallel(&q, &k, &v, false, cores));
        let t = time_interleaved(&mut [&mut f_serial, &mut f_par], 10);
        let (serial, par) = (t[0], t[1]);

        let speedup = serial.best / par.best;
        let ideal = (h.min(cores)) as f64;
        println!(
            "{:>6}  {:>13.2}  {:>15.2}  {:>9.2}x  {:>11.0}%",
            h,
            serial.best * 1e3,
            par.best * 1e3,
            speedup,
            speedup / ideal * 100.0
        );
    }
    println!("\n\"vs ideal\" is the fraction of min(heads, cores)x actually achieved.");
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
    multihead();
}

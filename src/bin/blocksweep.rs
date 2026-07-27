//! Sweeps the tile size across cache levels.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --bin blocksweep
//! ```
//!
//! `BLOCK` decides the working set of the inner loop: one query row against
//! `BLOCK` rows of K and `BLOCK` rows of V, so the resident bytes are
//! `2 * BLOCK * d * 4`. At `d = 64` that is 8 KB at BLOCK=16, 32 KB at 64,
//! 128 KB at 256 — spanning L1d into L2 on most parts. Too small and the
//! per-block bookkeeping (the online-softmax rescale, the loop setup) is paid
//! too often; too large and K/V stop fitting in the level you wanted.
//!
//! The sizes are const-generic instantiations, so each is compiled with its
//! bounds as literals — the same code the default kernel gets.

use flash_attn_rs::{attention_flops, filled, max_abs_diff, naive, tiled, time_interleaved, Mat};

const SIZES: [usize; 5] = [16, 32, 64, 128, 256];

fn run_for(block: usize, q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    match block {
        16 => tiled::attention_b::<16>(q, k, v, causal),
        32 => tiled::attention_b::<32>(q, k, v, causal),
        64 => tiled::attention_b::<64>(q, k, v, causal),
        128 => tiled::attention_b::<128>(q, k, v, causal),
        _ => tiled::attention_b::<256>(q, k, v, causal),
    }
}

fn main() {
    let d = 64;
    let causal = false;

    println!("tile sweep, head_dim d = {d}, full mask");
    println!("resident K+V per tile = 2 * BLOCK * d * 4 bytes\n");
    print!("{:>6}", "n");
    for b in SIZES {
        print!("{:>12}", format!("B={b}"));
    }
    println!("{:>10}", "best");
    print!("{:>6}", "");
    for b in SIZES {
        print!("{:>12}", format!("{} KB", 2 * b * d * 4 / 1024));
    }
    println!();

    for &n in &[256usize, 512, 1024] {
        let q = filled(n, d, 1);
        let k = filled(n, d, 2);
        let v = filled(n, d, 3);
        let flops = attention_flops(n, d, causal);

        // Correctness first: a fast tile size that computes the wrong thing is
        // not a data point.
        let reference = naive::attention(&q, &k, &v, causal);
        for b in SIZES {
            let got = run_for(b, &q, &k, &v, causal);
            let diff = max_abs_diff(&reference, &got);
            assert!(diff < 1e-4, "BLOCK={b} n={n} diff={diff}");
        }

        // Interleaved, for the same reason every other comparison here is: the
        // tile sizes are being compared against each other, so they have to see
        // the same machine.
        let iters = if n <= 256 { 40 } else { 12 };
        let mut f16 = || drop(run_for(16, &q, &k, &v, causal));
        let mut f32_ = || drop(run_for(32, &q, &k, &v, causal));
        let mut f64_ = || drop(run_for(64, &q, &k, &v, causal));
        let mut f128 = || drop(run_for(128, &q, &k, &v, causal));
        let mut f256 = || drop(run_for(256, &q, &k, &v, causal));
        let t = time_interleaved(
            &mut [&mut f16, &mut f32_, &mut f64_, &mut f128, &mut f256],
            iters,
        );

        let gf: Vec<f64> = t.iter().map(|x| flops / x.best / 1e9).collect();
        let best = gf
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        print!("{n:>6}");
        for g in &gf {
            print!("{g:>12.2}");
        }
        println!("{:>10}", format!("B={}", SIZES[best.0]));
    }
    println!("\nGFLOP/s, fastest of the interleaved rounds at each tile size.");
}

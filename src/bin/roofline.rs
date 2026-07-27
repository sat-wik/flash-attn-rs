//! Generates `docs/roofline.svg` and `docs/roofline.json` from measurements
//! taken on whatever machine runs it.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --bin roofline
//! ```
//!
//! The `target-cpu=native` matters: without it the AVX2 path is still selected
//! at runtime, but the surrounding scalar code is compiled for a baseline that
//! makes the comparison less representative of a tuned build.
//!
//! Everything on the figure is measured here, including both ceilings — see
//! `roofline::measure_peak_gflops` and `roofline::measure_bandwidth_gbs`. The
//! figure is stamped with the arch and ISA it was produced on, because a
//! roofline is a statement about one machine and nothing else.

use flash_attn_rs::roofline;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let out = std::path::Path::new("docs");
    std::fs::create_dir_all(out)?;

    eprint!("measuring compute and bandwidth ceilings... ");
    std::io::stderr().flush()?;
    let machine = roofline::measure_machine();
    eprintln!(
        "{:.1} GFLOP/s, {:.1} GB/s (ridge {:.2} FLOP/byte)",
        machine.peak_gflops,
        machine.bandwidth_gbs,
        machine.ridge()
    );

    eprint!("timing kernels... ");
    std::io::stderr().flush()?;
    let points = roofline::measure_points();
    eprintln!("{} operating points", points.len());

    let svg = out.join("roofline.svg");
    let json = out.join("roofline.json");
    std::fs::write(&svg, roofline::to_svg(&machine, &points))?;
    std::fs::write(&json, roofline::to_json(&machine, &points))?;

    eprintln!("wrote {} and {}", svg.display(), json.display());
    println!(
        "\n{:<8} {:>6} {:>8} {:>12} {:>14}",
        "kernel", "n", "mask", "GFLOP/s", "FLOP/byte"
    );
    for p in &points {
        println!(
            "{:<8} {:>6} {:>8} {:>12.2} {:>14.2}",
            p.kernel,
            p.n,
            if p.causal { "causal" } else { "full" },
            p.gflops,
            p.intensity
        );
    }
    Ok(())
}

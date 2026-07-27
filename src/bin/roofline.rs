//! Generates `docs/roofline.svg` and `docs/roofline.json` from measurements
//! taken on whatever machine runs it.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --bin roofline
//! cargo run --release --bin roofline -- --from-json docs/roofline.json
//! ```
//!
//! The `target-cpu=native` matters: without it the AVX2 path is still selected
//! at runtime, but the surrounding scalar code is compiled for a baseline that
//! makes the comparison less representative of a tuned build.
//!
//! `--from-json` re-renders the figure from an existing dataset without
//! measuring anything. That is what you want when the committed figure came
//! from hardware you do not have in front of you — running the tool to "refresh"
//! the SVG would otherwise silently replace someone else's numbers with your
//! machine's.
//!
//! Everything on the figure is measured here, including both ceilings — see
//! `roofline::measure_peak_gflops` and `roofline::measure_bandwidth_gbs`. The
//! figure is stamped with the arch and ISA it was produced on, because a
//! roofline is a statement about one machine and nothing else.

use flash_attn_rs::roofline;
use std::io::Write;

fn write_outputs(m: &roofline::Machine, pts: &[roofline::Point]) -> std::io::Result<()> {
    let out = std::path::Path::new("docs");
    std::fs::create_dir_all(out)?;
    let svg = out.join("roofline.svg");
    let json = out.join("roofline.json");
    std::fs::write(&svg, roofline::to_svg(m, pts))?;
    std::fs::write(&json, roofline::to_json(m, pts))?;
    eprintln!("wrote {} and {}", svg.display(), json.display());
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--from-json") {
        let path = args
            .get(i + 1)
            .map(String::as_str)
            .unwrap_or("docs/roofline.json");
        let text = std::fs::read_to_string(path)?;
        let (machine, points) = roofline::from_json(&text).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{path} is not a roofline dataset this build understands"),
            )
        })?;
        eprintln!(
            "re-rendering from {path}: {} on {}, {} points — nothing re-measured",
            machine.isa,
            machine.arch,
            points.len()
        );
        return write_outputs(&machine, &points);
    }

    eprint!("measuring compute and bandwidth ceilings... ");
    std::io::stderr().flush()?;
    let mut machine = roofline::measure_machine();
    eprintln!(
        "{:.1} GFLOP/s (+/-{:.1}%), {:.1} GB/s (+/-{:.1}%), ridge {:.2} FLOP/byte",
        machine.peak_gflops,
        machine.peak_spread_pct,
        machine.bandwidth_gbs,
        machine.bandwidth_spread_pct,
        machine.ridge()
    );

    // A ceiling measured on a contended host is not a ceiling. Say so loudly
    // rather than emitting a figure that looks authoritative and isn't.
    if machine.peak_spread_pct > roofline::COMPUTE_NOISE_WARN_PCT {
        eprintln!(
            "\nWARNING: the compute ceiling varied by {:.1}% across runs (limit {:.0}%).\n\
             That probe is a pure register loop, so it should repeat to within a\n\
             percent or two on a core it actually owns. This much spread means time\n\
             is being stolen — an oversubscribed vCPU, a noisy neighbour, or thermal\n\
             throttling. The ceiling is understated and the figure is not worth\n\
             committing. Re-run on a quiet or dedicated machine.",
            machine.peak_spread_pct,
            roofline::COMPUTE_NOISE_WARN_PCT
        );
    }
    if machine.bandwidth_spread_pct > roofline::BANDWIDTH_NOISE_WARN_PCT {
        eprintln!(
            "\nWARNING: the bandwidth ceiling varied by {:.1}% across runs (limit {:.0}%).\n\
             Some swing is normal here, but this much means something else on the box\n\
             is moving a lot of memory. The ridge point will be off. Re-run when idle.",
            machine.bandwidth_spread_pct,
            roofline::BANDWIDTH_NOISE_WARN_PCT
        );
    }

    eprint!("timing kernels... ");
    std::io::stderr().flush()?;
    let points = roofline::measure_points();
    eprintln!("{} operating points", points.len());

    // Re-measure the ceiling now that the kernels have run. The points are
    // plotted against the ceiling, so if the machine moved between the two the
    // whole figure is comparing numbers taken under different conditions.
    eprint!("re-checking the compute ceiling... ");
    std::io::stderr().flush()?;
    let (recheck, _) = roofline::measure_peak_gflops();
    machine.peak_recheck_gflops = recheck;
    eprintln!("{:.1} GFLOP/s (drift {:.1}%)", recheck, machine.drift_pct());
    if machine.drift_pct() > roofline::DRIFT_WARN_PCT {
        eprintln!(
            "\nWARNING: the compute ceiling moved {:.1}% between the start and end of\n\
             this run (limit {:.0}%). The operating points were measured against a\n\
             machine that was not holding still, so their position relative to the\n\
             roofline is not trustworthy. Re-run on a quieter host.",
            machine.drift_pct(),
            roofline::DRIFT_WARN_PCT
        );
    }

    write_outputs(&machine, &points)?;

    println!(
        "\n{:<8} {:>6} {:>8} {:>12} {:>14} {:>10}",
        "kernel", "n", "mask", "GFLOP/s", "FLOP/byte", "spread"
    );
    for p in &points {
        println!(
            "{:<8} {:>6} {:>8} {:>12.2} {:>14.2} {:>9.1}%",
            p.kernel,
            p.n,
            if p.causal { "causal" } else { "full" },
            p.gflops,
            p.intensity,
            p.spread_pct
        );
    }
    Ok(())
}

//! Roofline analysis: plot each kernel's measured throughput against the two
//! ceilings that can bound it, so "why is this kernel this fast" becomes a
//! position on a chart instead of an assertion.
//!
//! The roofline model says achievable GFLOP/s is capped by
//!
//! ```text
//! min(peak_compute, bandwidth * arithmetic_intensity)
//! ```
//!
//! where arithmetic intensity is FLOPs performed per byte moved. Below the
//! *ridge point* (intensity = peak_compute / bandwidth) you are memory-bound
//! and the only thing that helps is moving fewer bytes; above it you are
//! compute-bound and the only thing that helps is issuing better instructions.
//!
//! That distinction is the whole argument of this crate. Tiling attacks bytes
//! moved (it moves the operating point right); SIMD attacks instruction
//! throughput (it moves the point up). Each one only pays on its own side of
//! the ridge, which is exactly why tiling buys nothing here at d=64 while AVX2
//! buys ~5x.
//!
//! Both ceilings are *measured on the machine that runs this*, not read off a
//! spec sheet — see `measure_peak_gflops` and `measure_bandwidth_gbs`.

use crate::{attention_flops, filled, naive, simd, tiled, time_stats};
use std::time::Instant;

pub const HEAD_DIM: usize = 64;
pub const SEQ_LENS: [usize; 4] = [128, 256, 512, 1024];
pub const KERNELS: [&str; 3] = ["naive", "tiled", "simd"];

/// One kernel at one problem size: where it lands on the chart.
pub struct Point {
    pub kernel: &'static str,
    pub n: usize,
    pub causal: bool,
    pub gflops: f64,
    /// FLOPs per byte moved, from the traffic model in `bytes_moved`.
    pub intensity: f64,
    /// Median seconds per run. Emitted so the causal-vs-full wall-clock ratio is
    /// a division of two committed measurements rather than something inferred
    /// from GFLOP/s and a FLOP count.
    pub median_secs: f64,
    /// Interquartile spread of the timing, as a percentage of the median.
    pub spread_pct: f64,
}

/// The machine's two ceilings, plus enough identity to make the figure
/// reproducible and to stop anyone reading it as a claim about other hardware.
pub struct Machine {
    pub arch: &'static str,
    pub isa: String,
    pub peak_gflops: f64,
    pub bandwidth_gbs: f64,
    /// Sample spread for each ceiling, as a percentage of the best sample. On a
    /// quiet dedicated core this is a couple of percent. A large spread means
    /// the host was contended — a shared vCPU losing time to a neighbour, or
    /// thermal throttling — and the ceiling is not worth plotting against.
    pub peak_spread_pct: f64,
    pub bandwidth_spread_pct: f64,
}

/// Spread thresholds past which `roofline` warns that the host was too noisy.
///
/// The two ceilings deserve different limits. The compute probe is a pure
/// register loop that touches no memory, so on a machine with a core to itself
/// it repeats to within a percent or two — which makes it a sharp detector of
/// stolen time on a shared vCPU. The bandwidth probe contends with the page
/// cache, DRAM refresh and every other process on the box, so 10-20% swing is
/// ordinary even on a quiet machine and only a much larger spread is alarming.
pub const COMPUTE_NOISE_WARN_PCT: f64 = 10.0;
pub const BANDWIDTH_NOISE_WARN_PCT: f64 = 25.0;

/// Best and worst of a set of samples, reduced to (best, spread %).
fn summarize(samples: &[f64]) -> (f64, f64) {
    let best = samples.iter().copied().fold(0.0, f64::max);
    let worst = samples.iter().copied().fold(f64::MAX, f64::min);
    let spread = if best > 0.0 {
        (best - worst) / best * 100.0
    } else {
        0.0
    };
    (best, spread)
}

impl Machine {
    /// Arithmetic intensity at which the two ceilings cross. Left of it the
    /// workload is memory-bound, right of it compute-bound.
    pub fn ridge(&self) -> f64 {
        self.peak_gflops / self.bandwidth_gbs
    }

    /// The roofline itself: the best throughput this machine can offer a
    /// workload of the given intensity.
    pub fn ceiling_at(&self, intensity: f64) -> f64 {
        self.peak_gflops.min(self.bandwidth_gbs * intensity)
    }
}

/// Bytes of memory traffic for one attention pass.
///
/// This is a model, not a measurement. It counts the traffic each algorithm is
/// *obliged* to generate, at the one place where the kernels actually differ:
/// whether the `[n x n]` score matrix exists in memory at all. It deliberately
/// ignores cache hits, so for small `n` it overstates real DRAM traffic for
/// every kernel equally. That is the honest limitation — the figure is about
/// the ratio between the kernels, not an absolute DRAM byte count.
pub fn bytes_moved(kernel: &str, n: usize, d: usize, causal: bool) -> f64 {
    const F32: f64 = 4.0;
    let (nf, df) = (n as f64, d as f64);

    match kernel {
        "naive" => {
            // Q, K, V read once; O written once.
            let operands = 4.0 * nf * df * F32;
            // The score matrix is written by the QK^T pass, read and rewritten
            // in place by the softmax pass, and read once more by the PV pass:
            // four streams over the unmasked entries.
            let pairs = if causal {
                nf * (nf + 1.0) / 2.0
            } else {
                nf * nf
            };
            operands + 4.0 * pairs * F32
        }
        // tiled and simd share a traffic profile — simd changes which
        // instructions run, not which bytes move.
        _ => {
            // Q read once, O written once; the running softmax state (m, l) is
            // O(n) and rounds off against these.
            let operands = 2.0 * nf * df * F32;
            // K and V are re-streamed once per query block, since the query
            // block loop is outermost. Under a causal mask, query block b only
            // visits KV blocks 0..=b, so visits are triangular, not square.
            // Rounding up to whole blocks slightly overstates a ragged tail.
            let blocks = (nf / tiled::BLOCK as f64).ceil();
            let visits = if causal {
                blocks * (blocks + 1.0) / 2.0
            } else {
                blocks * blocks
            };
            operands + 2.0 * visits * tiled::BLOCK as f64 * df * F32
        }
    }
}

/// Empirical compute ceiling, in GFLOP/s.
///
/// A dependency-free chain of FMAs that touches no memory at all, so what it
/// reports is what the FMA units actually retire on this part — including any
/// clock the part drops under sustained vector load. That makes it a fairer
/// ceiling than `cores * clock * lanes * 2`, which no real code reaches.
pub fn measure_peak_gflops() -> (f64, f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { peak_avx2() };
        }
    }
    peak_portable()
}

/// Batch size in independent FMA lanes. Two constraints set this. It has to be
/// wide enough that the compiler vectorizes the portable loop — otherwise this
/// measures scalar throughput and the "ceiling" lands below the kernels it is
/// supposed to bound. And it has to keep enough FMAs in flight to cover the
/// latency of every issue port: a typical x86 core has two FMA ports at ~4
/// cycles latency, so fewer than 8 independent AVX2 accumulators measures
/// latency, not throughput, and reports roughly half the real peak. 64 lanes is
/// 8 ymm registers or 16 NEON registers — enough on both, spilling on neither.
const PEAK_WIDTH: usize = 64;

/// Grow the batch count until one timed run lasts `MIN_SECS`, so the
/// measurement is meaningful on a 7 GFLOP/s core and on a 500 GFLOP/s one
/// without a hand-tuned iteration count for each.
const MIN_SECS: f64 = 0.25;

/// A ceiling is a *best case*, so every probe takes the fastest of several runs
/// rather than one sample. This matters most on shared or virtualized hosts,
/// where an unlucky sample includes a neighbour's steal time and drags the roof
/// below the kernels it is supposed to bound.
const REPEATS: usize = 5;

fn peak_portable() -> (f64, f64) {
    let a = std::hint::black_box(1.000_000_1f32);
    let b = std::hint::black_box(0.999_999_9f32);
    let mut acc = [1.0f32; PEAK_WIDTH];

    let mut batches: u64 = 1 << 12;
    let mut samples = Vec::with_capacity(REPEATS);
    loop {
        let t = Instant::now();
        for _ in 0..batches {
            for x in acc.iter_mut() {
                *x = x.mul_add(a, b);
            }
        }
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(acc);

        if secs < MIN_SECS {
            batches *= 4;
            continue;
        }
        // One FMA is two FLOPs.
        samples.push((batches as f64) * (PEAK_WIDTH as f64) * 2.0 / secs / 1e9);
        if samples.len() == REPEATS {
            return summarize(&samples);
        }
    }
}

/// # Safety
///
/// The caller must have established that the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn peak_avx2() -> (f64, f64) {
    use std::arch::x86_64::*;
    // Same total lane count as the portable probe: 8 vectors x 8 f32.
    const VECS: usize = PEAK_WIDTH / 8;

    let a = _mm256_set1_ps(1.000_000_1);
    let b = _mm256_set1_ps(0.999_999_9);
    let mut acc = [_mm256_set1_ps(1.0); VECS];

    let mut batches: u64 = 1 << 12;
    let mut samples = Vec::with_capacity(REPEATS);
    loop {
        let t = Instant::now();
        for _ in 0..batches {
            for x in acc.iter_mut() {
                *x = _mm256_fmadd_ps(*x, a, b);
            }
        }
        let secs = t.elapsed().as_secs_f64();

        // Keep the accumulators live so the loop cannot be optimized away.
        let mut sink = [0.0f32; 8];
        for x in acc.iter() {
            _mm256_storeu_ps(sink.as_mut_ptr(), *x);
            std::hint::black_box(sink);
        }

        if secs < MIN_SECS {
            batches *= 4;
            continue;
        }
        // 8 lanes per vector, two FLOPs per FMA.
        samples.push((batches as f64) * (VECS as f64) * 16.0 / secs / 1e9);
        if samples.len() == REPEATS {
            return summarize(&samples);
        }
    }
}

/// Empirical bandwidth ceiling, in GB/s.
///
/// A STREAM-style triad (`a = b + s*c`) over buffers far larger than any
/// last-level cache, so the number reflects DRAM rather than cache. Counts the
/// three logical streams STREAM counts; on most parts the write also incurs a
/// read-for-ownership, so this is a conservative lower bound on real traffic.
pub fn measure_bandwidth_gbs() -> (f64, f64) {
    // 64 MB per buffer, 192 MB total — comfortably past any current LLC.
    const N: usize = 16 << 20;
    let mut a = vec![0.0f32; N];
    let b = vec![1.0f32; N];
    let c = vec![2.0f32; N];
    let s = std::hint::black_box(3.0f32);

    // Two warm-up passes, not one: the first faults 192 MB of pages in, and the
    // second settles the TLB. Timing either of them makes a quiet machine look
    // contended.
    triad(&mut a, &b, &c, s);
    triad(&mut a, &b, &c, s);

    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t = Instant::now();
        triad(&mut a, &b, &c, s);
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(&a);
        samples.push(3.0 * N as f64 * 4.0 / secs / 1e9);
    }
    summarize(&samples)
}

fn triad(a: &mut [f32], b: &[f32], c: &[f32], s: f32) {
    for ((x, &y), &z) in a.iter_mut().zip(b).zip(c) {
        *x = y + s * z;
    }
}

/// Which vector path `simd::attention` will actually take here. Recorded on the
/// figure so a plot made on a machine without AVX2 can't be mistaken for one
/// made with it.
pub fn detected_isa() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(avx512) && is_x86_feature_detected!("avx512f") {
            "AVX-512F".to_string()
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            "AVX2 + FMA".to_string()
        } else {
            "x86-64 baseline (scalar fallback)".to_string()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "portable fallback (no x86 SIMD path)".to_string()
    }
}

pub fn measure_machine() -> Machine {
    let (peak_gflops, peak_spread_pct) = measure_peak_gflops();
    let (bandwidth_gbs, bandwidth_spread_pct) = measure_bandwidth_gbs();
    Machine {
        arch: std::env::consts::ARCH,
        isa: detected_isa(),
        peak_gflops,
        bandwidth_gbs,
        peak_spread_pct,
        bandwidth_spread_pct,
    }
}

/// Time every kernel at every sequence length under both masks.
pub fn measure_points() -> Vec<Point> {
    let d = HEAD_DIM;
    let mut pts = Vec::new();

    for &causal in &[false, true] {
        for &n in &SEQ_LENS {
            let q = filled(n, d, 1);
            let k = filled(n, d, 2);
            let v = filled(n, d, 3);
            let flops = attention_flops(n, d, causal);
            let iters = if n <= 256 { 50 } else { 15 };

            for &kernel in &KERNELS {
                let t = match kernel {
                    "naive" => time_stats(|| drop(naive::attention(&q, &k, &v, causal)), iters),
                    "tiled" => time_stats(|| drop(tiled::attention(&q, &k, &v, causal)), iters),
                    _ => time_stats(|| drop(simd::attention(&q, &k, &v, causal)), iters),
                };
                pts.push(Point {
                    kernel,
                    n,
                    causal,
                    gflops: flops / t.median / 1e9,
                    intensity: flops / bytes_moved(kernel, n, d, causal),
                    median_secs: t.median,
                    spread_pct: t.spread_pct,
                });
            }
        }
    }
    pts
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

pub fn to_json(m: &Machine, pts: &[Point]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!(
        "  \"machine\": {{ \"arch\": \"{}\", \"isa\": \"{}\", \"peak_gflops\": {:.3}, \"bandwidth_gbs\": {:.3}, \"ridge_flop_per_byte\": {:.4}, \"peak_spread_pct\": {:.2}, \"bandwidth_spread_pct\": {:.2} }},\n",
        m.arch,
        m.isa,
        m.peak_gflops,
        m.bandwidth_gbs,
        m.ridge(),
        m.peak_spread_pct,
        m.bandwidth_spread_pct
    ));
    s.push_str(&format!("  \"head_dim\": {},\n", HEAD_DIM));
    s.push_str("  \"points\": [\n");
    for (i, p) in pts.iter().enumerate() {
        s.push_str(&format!(
            "    {{ \"kernel\": \"{}\", \"n\": {}, \"mask\": \"{}\", \"gflops\": {:.4}, \"intensity_flop_per_byte\": {:.4}, \"median_secs\": {:.9}, \"spread_pct\": {:.2} }}{}\n",
            p.kernel,
            p.n,
            if p.causal { "causal" } else { "full" },
            p.gflops,
            p.intensity,
            p.median_secs,
            p.spread_pct,
            if i + 1 == pts.len() { "" } else { "," }
        ));
    }
    s.push_str("  ]\n}\n");
    s
}

// Plot geometry.
const W: f64 = 900.0;
const H: f64 = 560.0;
const L: f64 = 78.0;
const R: f64 = 210.0;
const T: f64 = 88.0;
const B: f64 = 66.0;

const COLORS: [&str; 3] = ["#c2410c", "#0369a1", "#15803d"];

struct Axes {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

impl Axes {
    fn px(&self, x: f64) -> f64 {
        L + (x.log10() - self.x0.log10()) / (self.x1.log10() - self.x0.log10()) * (W - L - R)
    }
    fn py(&self, y: f64) -> f64 {
        H - B - (y.log10() - self.y0.log10()) / (self.y1.log10() - self.y0.log10()) * (H - T - B)
    }
}

/// Decade-anchored 1/2/5 ticks covering [lo, hi].
fn ticks(lo: f64, hi: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut decade = 10f64.powf(lo.log10().floor());
    while decade <= hi * 10.0 {
        for m in [1.0, 2.0, 5.0] {
            let v = decade * m;
            if v >= lo && v <= hi {
                out.push(v);
            }
        }
        decade *= 10.0;
    }
    out
}

fn fmt_tick(v: f64) -> String {
    if v >= 1.0 {
        format!("{v:.0}")
    } else if v >= 0.1 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Marker path for a kernel, centred on (x, y).
fn marker(kernel: &str, x: f64, y: f64, fill: &str, hollow: bool) -> String {
    let (stroke_w, fill_attr) = if hollow {
        (2.0, "#ffffff".to_string())
    } else {
        (1.2, fill.to_string())
    };
    match kernel {
        // circle
        "naive" => format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"5.2\" fill=\"{fill_attr}\" stroke=\"{fill}\" stroke-width=\"{stroke_w}\"/>"
        ),
        // square
        "tiled" => format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"9.6\" height=\"9.6\" fill=\"{fill_attr}\" stroke=\"{fill}\" stroke-width=\"{stroke_w}\"/>",
            x - 4.8,
            y - 4.8
        ),
        // triangle
        _ => format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{fill_attr}\" stroke=\"{fill}\" stroke-width=\"{stroke_w}\" stroke-linejoin=\"round\"/>",
            x,
            y - 6.0,
            x + 5.5,
            y + 4.0,
            x - 5.5,
            y + 4.0
        ),
    }
}

pub fn to_svg(m: &Machine, pts: &[Point]) -> String {
    // Range: cover every point and the ridge, with padding.
    let min_i = pts.iter().map(|p| p.intensity).fold(f64::MAX, f64::min);
    let max_i = pts.iter().map(|p| p.intensity).fold(0.0, f64::max);
    let min_g = pts.iter().map(|p| p.gflops).fold(f64::MAX, f64::min);

    let ax = Axes {
        x0: (min_i * 0.45).min(m.ridge() * 0.3),
        x1: (max_i * 2.2).max(m.ridge() * 3.0),
        y0: min_g * 0.45,
        y1: m.peak_gflops * 1.8,
    };

    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {W:.0} {H:.0}\" width=\"{W:.0}\" height=\"{H:.0}\" font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\">\n"
    ));
    s.push_str(&format!(
        "<rect width=\"{W:.0}\" height=\"{H:.0}\" fill=\"#ffffff\"/>\n"
    ));

    // Title + provenance.
    s.push_str(&format!(
        "<text x=\"{L:.0}\" y=\"30\" font-size=\"17\" font-weight=\"600\" fill=\"#0f172a\">Attention kernels vs the roofline (d = {HEAD_DIM})</text>\n"
    ));
    s.push_str(&format!(
        "<text x=\"{L:.0}\" y=\"50\" font-size=\"12\" fill=\"#475569\">measured on {} — {}</text>\n",
        m.arch, m.isa
    ));
    let worst_spread = pts.iter().map(|p| p.spread_pct).fold(0.0, f64::max);
    s.push_str(&format!(
        "<text x=\"{L:.0}\" y=\"68\" font-size=\"11.5\" fill=\"#64748b\">compute ceiling {:.0} GFLOP/s  ·  bandwidth {:.1} GB/s  ·  ridge {:.2} FLOP/byte  ·  worst point spread {:.0}%</text>\n",
        m.peak_gflops,
        m.bandwidth_gbs,
        m.ridge(),
        worst_spread
    ));

    // Grid + ticks.
    for t in ticks(ax.x0, ax.x1) {
        let x = ax.px(t);
        s.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n",
            T,
            H - B
        ));
        s.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#64748b\" text-anchor=\"middle\">{}</text>\n",
            H - B + 18.0,
            fmt_tick(t)
        ));
    }
    for t in ticks(ax.y0, ax.y1) {
        let y = ax.py(t);
        s.push_str(&format!(
            "<line x1=\"{L:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n",
            W - R
        ));
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#64748b\" text-anchor=\"end\">{}</text>\n",
            L - 8.0,
            y + 3.8,
            fmt_tick(t)
        ));
    }

    // Memory-bound region shading, left of the ridge.
    let ridge_x = ax.px(m.ridge()).clamp(L, W - R);
    s.push_str(&format!(
        "<rect x=\"{L:.1}\" y=\"{T:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#0369a1\" opacity=\"0.04\"/>\n",
        ridge_x - L,
        H - T - B
    ));

    // The roofline: sloped bandwidth bound, then flat compute bound.
    let mut roof = String::new();
    let steps = 220;
    for i in 0..=steps {
        let f = i as f64 / steps as f64;
        let x = 10f64.powf(ax.x0.log10() + f * (ax.x1.log10() - ax.x0.log10()));
        roof.push_str(&format!("{:.1},{:.1} ", ax.px(x), ax.py(m.ceiling_at(x))));
    }
    s.push_str(&format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"2.4\" stroke-linejoin=\"round\"/>\n",
        roof.trim_end()
    ));

    // Ridge marker.
    s.push_str(&format!(
        "<line x1=\"{ridge_x:.1}\" y1=\"{:.1}\" x2=\"{ridge_x:.1}\" y2=\"{:.1}\" stroke=\"#0f172a\" stroke-width=\"1.4\" stroke-dasharray=\"5 4\" opacity=\"0.65\"/>\n",
        T,
        H - B
    ));
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#0f172a\" text-anchor=\"end\" opacity=\"0.8\">memory-bound</text>\n",
        ridge_x - 8.0,
        T + 14.0
    ));
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#0f172a\" opacity=\"0.8\">compute-bound</text>\n",
        ridge_x + 8.0,
        T + 14.0
    ));
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10.5\" fill=\"#0f172a\" text-anchor=\"end\" opacity=\"0.75\">ridge {:.2}</text>\n",
        ridge_x - 7.0,
        H - B - 9.0,
        m.ridge()
    ));

    // Operating points.
    for p in pts {
        let ci = KERNELS.iter().position(|k| *k == p.kernel).unwrap_or(0);
        s.push_str(&marker(
            p.kernel,
            ax.px(p.intensity),
            ax.py(p.gflops),
            COLORS[ci],
            p.causal,
        ));
        s.push('\n');
    }

    // Axis labels.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12.5\" fill=\"#0f172a\" text-anchor=\"middle\">arithmetic intensity (FLOP / byte)</text>\n",
        L + (W - L - R) / 2.0,
        H - 18.0
    ));
    s.push_str(&format!(
        "<text transform=\"translate(22,{:.1}) rotate(-90)\" font-size=\"12.5\" fill=\"#0f172a\" text-anchor=\"middle\">GFLOP/s</text>\n",
        T + (H - T - B) / 2.0
    ));

    // Legend.
    let lx = W - R + 22.0;
    let mut ly = T + 6.0;
    s.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{ly:.1}\" font-size=\"12\" font-weight=\"600\" fill=\"#0f172a\">kernel</text>\n"
    ));
    ly += 20.0;
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&marker(k, lx + 7.0, ly - 4.0, COLORS[i], false));
        s.push_str(&format!(
            "\n<text x=\"{:.1}\" y=\"{ly:.1}\" font-size=\"12\" fill=\"#334155\">{k}</text>\n",
            lx + 22.0
        ));
        ly += 22.0;
    }
    ly += 10.0;
    s.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{ly:.1}\" font-size=\"12\" font-weight=\"600\" fill=\"#0f172a\">mask</text>\n"
    ));
    ly += 20.0;
    s.push_str(&marker("naive", lx + 7.0, ly - 4.0, "#334155", false));
    s.push_str(&format!(
        "\n<text x=\"{:.1}\" y=\"{ly:.1}\" font-size=\"12\" fill=\"#334155\">full</text>\n",
        lx + 22.0
    ));
    ly += 22.0;
    s.push_str(&marker("naive", lx + 7.0, ly - 4.0, "#334155", true));
    s.push_str(&format!(
        "\n<text x=\"{:.1}\" y=\"{ly:.1}\" font-size=\"12\" fill=\"#334155\">causal</text>\n",
        lx + 22.0
    ));

    s.push_str("</svg>\n");
    s
}

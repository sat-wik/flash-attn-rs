//! Attention kernels, from a naive baseline to a tiled ("flash") kernel to an
//! AVX2-vectorized kernel. The point of this crate is not to beat cuDNN; it is
//! to make the *hardware reason* for each speedup measurable and explainable.
//!
//! Problem: single-head scaled dot-product attention.
//!   scores = Q @ K^T / sqrt(d)     [n x n]
//!   probs  = softmax(scores)       [n x n]  (row-wise)
//!   out    = probs @ V             [n x d]
//!
//! Q, K, V are row-major [n x d] matrices. `n` = sequence length, `d` = head dim.

pub mod avx512;
pub mod multihead;
pub mod naive;
pub mod roofline;
pub mod simd;
pub mod tiled;
pub mod vexp;

/// A row-major matrix. Kept deliberately simple so the kernels below read like
/// the math, not like a matrix library.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline(always)]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Deterministic pseudo-random fill so benchmarks and correctness tests are
/// reproducible without pulling in the `rand` crate.
pub fn filled(rows: usize, cols: usize, seed: u64) -> Mat {
    let mut m = Mat::zeros(rows, cols);
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    for v in m.data.iter_mut() {
        // xorshift* -> [-1, 1)
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let x = (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u32 << 24) as f32;
        *v = x * 2.0 - 1.0;
    }
    m
}

/// Max absolute difference between two matrices — used to check that the fast
/// kernels agree with the naive reference.
pub fn max_abs_diff(a: &Mat, b: &Mat) -> f32 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Wall-clock result for one timed kernel.
///
/// Report throughput from `best`, not `median`. Interference on a shared
/// machine is one-sided — a descheduled slice, a neighbour evicting your cache,
/// a stolen timeslice can only ever make a run *slower*, never faster — so the
/// fastest observed run is the least-contaminated estimate of how long the
/// kernel actually takes. The median is an estimate of what the machine was
/// doing to you, which is a different question.
///
/// It also keeps this consistent with the roofline ceilings, which take the
/// best of several probes. Comparing best-case ceilings against median-case
/// operating points would understate every "percentage of peak" on the chart.
#[derive(Clone, Copy)]
pub struct Timing {
    /// Fastest seconds observed. Use this for throughput.
    pub best: f64,
    /// Median seconds per run, kept for reference against `best`.
    pub median: f64,
    /// Interquartile range as a percentage of the median.
    ///
    /// This is the answer to "how repeatable is that number?", which a single
    /// median cannot give you. IQR rather than min/max because one descheduled
    /// run on a shared machine should not define the error bar, and reporting it
    /// at all is what lets a reader tell a real effect from timing noise.
    pub spread_pct: f64,
}

fn timing_from(s: &mut [f64]) -> Timing {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = s[s.len() / 2];
    let q1 = s[s.len() / 4];
    let q3 = s[(3 * s.len()) / 4];
    Timing {
        best: s[0],
        median,
        spread_pct: if median > 0.0 {
            (q3 - q1) / median * 100.0
        } else {
            0.0
        },
    }
}

/// Median and spread over `iters` timed runs, after three warm-up runs.
///
/// Use [`time_interleaved`] instead whenever the result will be *compared* with
/// another kernel's — see the note there.
pub fn time_stats<F: FnMut()>(mut f: F, iters: usize) -> Timing {
    use std::time::Instant;
    for _ in 0..3 {
        f();
    }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        s.push(t.elapsed().as_secs_f64());
    }
    timing_from(&mut s)
}

/// Time several closures round-robin instead of one exhaustively after another.
///
/// Timing kernel A fifty times and *then* kernel B fifty times measures the two
/// under different conditions. On a shared machine the available throughput
/// drifts between the blocks, and the ratio — which is the headline number —
/// absorbs that drift whole. Measured this way on an oversubscribed vCPU, the
/// same speedup came out anywhere from 4.2x to 8.3x across two back-to-back
/// runs, and almost all of the movement was in the baseline rather than the
/// optimized kernel.
///
/// Interleaving turns it into a paired comparison: inside one round every
/// kernel sees the same neighbours, the same clock and the same contention, so
/// drift moves them together and largely cancels in the ratio. The starting
/// position rotates each round, so no kernel systematically pays for running
/// first into a cold cache.
pub fn time_interleaved(fs: &mut [&mut dyn FnMut()], rounds: usize) -> Vec<Timing> {
    use std::time::Instant;
    let k = fs.len();
    for f in fs.iter_mut() {
        for _ in 0..3 {
            f();
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); k];
    for r in 0..rounds {
        for off in 0..k {
            let i = (r + off) % k;
            let t = Instant::now();
            fs[i]();
            samples[i].push(t.elapsed().as_secs_f64());
        }
    }
    samples.iter_mut().map(|s| timing_from(s)).collect()
}

/// FLOPs for one attention pass, for turning wall-clock time into GFLOP/s.
/// QK^T: 2*n*n*d, softmax: ~3*n*n (cheap), PV: 2*n*n*d. Dominant term ~4*n^2*d.
pub fn attention_flops(n: usize, d: usize, causal: bool) -> f64 {
    let pairs = if causal {
        // Lower triangle incl. diagonal: n*(n+1)/2 query-key pairs.
        (n as f64) * (n as f64 + 1.0) / 2.0
    } else {
        (n as f64) * (n as f64)
    };
    4.0 * pairs * (d as f64)
}

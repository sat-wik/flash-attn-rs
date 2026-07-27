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

/// Median wall-clock seconds over `iters` timed runs, after three warm-up runs.
/// Median rather than mean so one descheduled run doesn't skew the result.
pub fn time_median<F: FnMut()>(mut f: F, iters: usize) -> f64 {
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
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
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

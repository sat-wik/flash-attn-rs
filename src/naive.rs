//! Baseline: the textbook three-pass formulation. It materializes the full
//! [n x n] score matrix, which is O(n^2) memory traffic and blows past cache as
//! `n` grows. This is the thing every later kernel is measured against.
//!
//! `causal` masks out future positions (j > i), as in a decoder / GPT-style
//! model: query i may only attend to keys 0..=i.

use crate::Mat;

pub fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    let n = q.rows;
    let d = q.cols;
    let scale = 1.0 / (d as f32).sqrt();

    // 1) scores = Q @ K^T / sqrt(d)   -- full n x n matrix in memory.
    let mut scores = Mat::zeros(n, n);
    for i in 0..n {
        let qi = q.row(i);
        let jmax = if causal { i + 1 } else { n };
        for j in 0..jmax {
            let kj = k.row(j);
            let mut acc = 0.0f32;
            for t in 0..d {
                acc += qi[t] * kj[t];
            }
            scores.row_mut(i)[j] = acc * scale;
        }
        // Masked positions stay at -inf so softmax gives them zero weight.
        if causal {
            for j in (i + 1)..n {
                scores.row_mut(i)[j] = f32::NEG_INFINITY;
            }
        }
    }

    // 2) row-wise softmax, in place.
    for i in 0..n {
        let row = scores.row_mut(i);
        let mut m = f32::NEG_INFINITY;
        for &x in row.iter() {
            m = m.max(x);
        }
        let mut sum = 0.0f32;
        for x in row.iter_mut() {
            *x = if x.is_finite() { (*x - m).exp() } else { 0.0 };
            sum += *x;
        }
        let inv = 1.0 / sum;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }

    // 3) out = probs @ V
    let mut out = Mat::zeros(n, d);
    for i in 0..n {
        let pi = scores.row(i);
        let oi = out.row_mut(i);
        let jmax = if causal { i + 1 } else { n };
        for (j, &p) in pi.iter().take(jmax).enumerate() {
            let vj = v.row(j);
            for t in 0..d {
                oi[t] += p * vj[t];
            }
        }
    }
    out
}

//! Tiled ("flash") attention. The key idea: never materialize the [n x n] score
//! matrix. Instead, walk the KV sequence in blocks and maintain a *running*
//! softmax (max `m` and normalizer `l`) per query row, rescaling the partial
//! output as new blocks arrive. Memory traffic drops from O(n^2) to O(n*d).
//!
//! Causal masking here is where tiling earns real speed: a whole KV block whose
//! smallest index already exceeds the largest query index can be skipped, so we
//! touch only the lower-triangular blocks — roughly half the work. Within the
//! diagonal block we apply the per-element j <= i mask.

use crate::Mat;

pub const BLOCK: usize = 64;

pub fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
    let n = q.rows;
    let d = q.cols;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = Mat::zeros(n, d);

    let mut m = vec![f32::NEG_INFINITY; n];
    let mut l = vec![0.0f32; n];

    let mut q0 = 0;
    while q0 < n {
        let q1 = (q0 + BLOCK).min(n);

        let mut kv0 = 0;
        while kv0 < n {
            let kv1 = (kv0 + BLOCK).min(n);

            // Causal block skip: if the whole KV block sits strictly after the
            // whole query block, every (i, j) in it is masked. Skip it.
            if causal && kv0 > q1 - 1 {
                break; // KV blocks only increase; nothing after this qualifies.
            }
            // Blocks straddling the diagonal still contain masked entries; the
            // per-element `j <= i` mask is applied by clamping `jhi` below.
            for i in q0..q1 {
                let qi = q.row(i);

                let mut block_scores = [0.0f32; BLOCK];
                let mut block_max = f32::NEG_INFINITY;
                let jhi = if causal { kv1.min(i + 1) } else { kv1 };
                for (bj, j) in (kv0..jhi).enumerate() {
                    let kj = k.row(j);
                    let mut acc = 0.0f32;
                    for t in 0..d {
                        acc += qi[t] * kj[t];
                    }
                    acc *= scale;
                    block_scores[bj] = acc;
                    if acc > block_max {
                        block_max = acc;
                    }
                }
                let valid = jhi.saturating_sub(kv0);
                if valid == 0 {
                    continue;
                } // fully masked for this query row

                let m_old = m[i];
                let m_new = m_old.max(block_max);
                let correction = if m_old == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m_old - m_new).exp()
                };

                if correction != 1.0 {
                    for o in out.row_mut(i).iter_mut() {
                        *o *= correction;
                    }
                }
                l[i] *= correction;

                let oi = out.row_mut(i);
                for (bj, j) in (kv0..jhi).enumerate() {
                    let p = (block_scores[bj] - m_new).exp();
                    l[i] += p;
                    let vj = v.row(j);
                    for t in 0..d {
                        oi[t] += p * vj[t];
                    }
                }
                m[i] = m_new;
            }
            kv0 = kv1;
        }
        q0 = q1;
    }

    for (i, &li) in l.iter().enumerate() {
        let inv = 1.0 / if li == 0.0 { 1.0 } else { li };
        for o in out.row_mut(i).iter_mut() {
            *o *= inv;
        }
    }
    out
}

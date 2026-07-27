//! Multi-head attention — the outer loop that turns a kernel into a layer.
//!
//! Real attention runs `h` heads over the same sequence, each with its own
//! projections of Q, K and V. The heads never interact until the concatenation
//! at the end, so this is `h` completely independent instances of the
//! single-head problem.
//!
//! That independence is the interesting part, because it forces a scheduling
//! decision: parallelize *across* heads, or *within* one head?
//!
//! Across heads wins here, and the roofline says why. Each head is already
//! compute-bound at `d = 64` (see `roofline`), so the cores are not waiting on
//! memory and there is nothing to gain from splitting a head's working set
//! between them. Splitting within a head would mean sharing the running softmax
//! state `m` and `l` across threads — either synchronizing on every KV block or
//! keeping per-thread partial accumulators and merging them, which is the same
//! rescaling dance the online softmax already does, now with a barrier in it.
//! Splitting across heads needs none of that: separate inputs, separate
//! outputs, no shared mutable state, no synchronization beyond the join.
//!
//! The one thing across-heads gives up is granularity. With fewer heads than
//! cores some cores idle, and that is the regime where within-head splitting
//! would start to be worth its complexity.

use crate::{simd, Mat};

/// All heads, one after another.
pub fn attention(q: &[Mat], k: &[Mat], v: &[Mat], causal: bool) -> Vec<Mat> {
    assert_eq!(q.len(), k.len());
    assert_eq!(q.len(), v.len());
    (0..q.len())
        .map(|h| simd::attention(&q[h], &k[h], &v[h], causal))
        .collect()
}

/// All heads, spread over `threads` OS threads.
///
/// Uses `std::thread::scope` so the worker closures can borrow the input slices
/// directly — no `Arc`, no cloning of Q/K/V, and no dependency on a thread-pool
/// crate. Heads are handed out in contiguous chunks and results are collected in
/// head order, so the output is identical to [`attention`] regardless of how the
/// work was split.
pub fn attention_parallel(
    q: &[Mat],
    k: &[Mat],
    v: &[Mat],
    causal: bool,
    threads: usize,
) -> Vec<Mat> {
    let h = q.len();
    assert_eq!(h, k.len());
    assert_eq!(h, v.len());
    if threads <= 1 || h <= 1 {
        return attention(q, k, v, causal);
    }

    let threads = threads.min(h);
    let chunk = h.div_ceil(threads);
    let mut out = Vec::with_capacity(h);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for c in 0..threads {
            let lo = c * chunk;
            if lo >= h {
                break;
            }
            let hi = (lo + chunk).min(h);
            handles.push(s.spawn(move || {
                (lo..hi)
                    .map(|i| simd::attention(&q[i], &k[i], &v[i], causal))
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            out.extend(handle.join().expect("attention head panicked"));
        }
    });

    out
}

/// `h` heads of reproducible test data, each seeded differently so no two heads
/// are secretly the same problem.
pub fn filled_heads(heads: usize, n: usize, d: usize, seed: u64) -> Vec<Mat> {
    (0..heads)
        .map(|h| crate::filled(n, d, seed.wrapping_add(h as u64 * 0x9E37)))
        .collect()
}

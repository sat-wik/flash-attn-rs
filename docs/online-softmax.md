# The online softmax, derived

The whole tiled kernel rests on one claim: **you can compute a softmax without
ever holding the full row in memory.** Everything else — the block loop, the
causal skip, the O(n·d) traffic — follows from that. This is the derivation,
because "flash attention rescales as it goes" is a description, not a reason.

## The problem

Softmax over a row `x` of length `n` is

```text
softmax(x)_j = exp(x_j) / Σ_t exp(x_t)
```

Computed naively this overflows: attention scores routinely exceed 88, and
`exp(89f32)` is `inf`. The standard fix subtracts the row max, which is
mathematically free because it cancels:

```text
exp(x_j - m) / Σ_t exp(x_t - m)   where m = max_t x_t
```

The numerator and denominator are each scaled by `exp(-m)`, so the quotient is
unchanged, and every exponent is now ≤ 0.

That fix is also the problem. You need `m` before you can exponentiate anything,
and you only know `m` after seeing the whole row. That is what forces the naive
kernel into three passes over an `[n × n]` matrix — find the max, exponentiate
and sum, then normalize — and it is why the score matrix has to exist.

## The fix: carry the max with you

Process the row in blocks. After block `b`, keep two running scalars per query
row:

- `m` — the largest score seen so far
- `l` — the sum of `exp(score - m)` over scores seen so far, **with the current
  `m`**

The second one is the subtle part. `l` is not a plain running sum; it is a sum
expressed relative to a normalizer that keeps changing. When a new block arrives
with a larger max, every term already in `l` was scaled by the wrong constant.

So fix them. Suppose the running state is `(m_old, l_old)` and the new block has
max `m_blk`. The new running max is

```text
m_new = max(m_old, m_blk)
```

Every existing term was `exp(x_t - m_old)` and should now be `exp(x_t - m_new)`.
Since

```text
exp(x_t - m_new) = exp(x_t - m_old) · exp(m_old - m_new)
```

the correction is the *same constant* for every term, so it factors straight out
of the sum:

```text
l_new = l_old · exp(m_old - m_new) + Σ_{j ∈ block} exp(x_j - m_new)
```

One multiply repairs the entire history. That single constant, `exp(m_old -
m_new)`, is the whole trick. Note it is always ≤ 1, so the rescale never
overflows.

## Carrying the output too

The same argument applies to the accumulated output. The unnormalized output
after some prefix is

```text
o = Σ_t exp(x_t - m) · V_t
```

which is scaled by exactly the same constant when `m` moves, because it appears
once in every term:

```text
o_new = o_old · exp(m_old - m_new) + Σ_{j ∈ block} exp(x_j - m_new) · V_j
```

So the per-block update is:

1. score the block, find `m_blk`
2. `m_new = max(m_old, m_blk)`, `c = exp(m_old - m_new)`
3. `o ← o · c`, `l ← l · c`  ← the rescale, `d + 1` multiplies
4. accumulate the block's `exp(x_j - m_new) · V_j` into `o`, and its
   `exp(x_j - m_new)` into `l`

and after the last block, `out = o / l`. That final division is the deferred
normalization: it never needed to happen early.

In [`src/tiled.rs`](../src/tiled.rs) step 3 is the `correction` variable, and
the guard `if correction != 1.0` skips the rescale in the common case where the
new block did not raise the max — which, once a row has seen a few blocks, is
most of the time.

## Why the answer is identical, not approximate

Each step is exact real arithmetic: `exp(a-c) = exp(a-b)·exp(b-c)` holds
exactly, so at every point `(m, l, o)` are precisely what a two-pass softmax over
the prefix would have produced. Induct over blocks and the final state equals the
full-row computation. There is no approximation anywhere in the algorithm.

Floating point does introduce a difference, but a small and *favourable* one:
the running formulation performs its sums in a different order, and every
intermediate is bounded because `exp(x - m) ≤ 1` by construction. That is why
[`tests/correctness.rs`](../tests/correctness.rs) holds `tiled` to `1e-4`
against the naive reference. The looser `2e-3` on the SIMD paths is nothing to
do with this — it is the polynomial `exp` in
[`src/vexp.rs`](../src/vexp.rs), which trades a few bits for eight lanes at a
time.

## What it buys

The score matrix never exists. The kernel holds one query row, one block of K
and V, and two scalars per row — so traffic drops from O(n²) to O(n·d), and
memory use stops growing quadratically with sequence length.

Whether that *helps* is a separate question, and this repo's answer is: at
`d = 64` on a CPU, mostly not. See the roofline in the [README](../README.md) —
every kernel here is compute-bound, so an optimization that only removes memory
traffic has nothing to buy. The online softmax is what makes the memory win
*possible*; the roofline is what tells you whether it is the win you need. Flash
Attention was designed for a regime where it is.

## Where the causal mask comes in

Blocking has a second payoff that has nothing to do with memory. Under a causal
mask, query `i` may only attend to keys `j ≤ i`, so any KV block lying entirely
past the query block is fully masked — every score in it would be `-inf` and
contribute `exp(-inf) = 0`.

The naive formulation cannot exploit this cheaply. It materializes the full row
including the masked half, writes `-inf` across it, and its softmax pass walks
all `n` entries regardless. The measurements bear this out: naive's causal
throughput runs *below* its full-mask throughput even after the FLOP count is
adjusted for the halved work.

The tiled kernel just skips those blocks — the `break` in the KV loop, since
blocks only increase. Only the diagonal block needs the per-element `j ≤ i`
test, handled by clamping the block's upper bound. Measured directly, that is
worth almost exactly the 2× you would predict from touching half the query-key
pairs.

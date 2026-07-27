# flash-attn-rs

[![CI](https://github.com/sat-wik/flash-attn-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sat-wik/flash-attn-rs/actions/workflows/ci.yml)

Hand-optimized single-head attention kernels in Rust, built to make the
*hardware reason* for each speedup measurable and explainable — not to beat
cuDNN. Three implementations of the same math, with causal masking and a
vectorized softmax, benchmarked against each other.

1. **`naive`** — textbook three-pass attention. Materializes the full `[n×n]`
   score matrix. O(n²) memory traffic; the baseline.
2. **`tiled`** — the Flash Attention algorithm on CPU: tiles over the KV
   sequence with a running (online) softmax, never materializing `[n×n]`.
   Under a causal mask it *skips* whole KV blocks above the diagonal.
3. **`simd`** — the tiled kernel with AVX2 + FMA intrinsics (8 f32 lanes) on the
   Q·Kᵀ dot product and p·V accumulate, **plus a vectorized `exp`** so softmax
   exponentials are computed 8-wide instead of one scalar libm call per score.
   Runtime feature detection with a scalar fallback.

An **AVX-512** kernel (16-wide) is included, `cfg`-gated for nightly / Rust 1.89+
so the default stable build stays green.

All kernels are verified bit-close to the naive reference, for both masks
(`cargo test`).

## Results

**AVX2 with a vectorized softmax runs 4.8–6.4× faster than the naive baseline.**
Tiling on its own is worth another 1.2–1.5×, and causal block-skipping roughly
halves wall-clock time. But the fastest kernel still reaches only ~17% of the
machine's measured compute ceiling, and the roofline below shows why: at
`d = 64` this workload is compute-bound at every size tested, so the
memory-traffic win that tiling exists for never gets to pay.

Single core of an x86_64 machine with AVX2 + FMA (a shared VPS vCPU — hence
modest absolute throughput; the *ratios* are the point), `head_dim = 64`, stable
Rust with `-C target-cpu=native`. GFLOP/s counts only unmasked query–key pairs,
so the two masks are directly comparable. Every number here and in the figure
comes from one run of the committed generator.

**Full (bidirectional) mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 2.26  | 2.67  | 10.76 | **4.76×**    |
| 256  | 2.15  | 3.18  | 12.04 | **5.61×**    |
| 512  | 2.34  | 3.08  | 13.62 | **5.83×**    |
| 1024 | 2.10  | 2.85  | 10.72 | **5.09×**    |

**Causal mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 2.01  | 2.70  | 11.10 | **5.53×**    |
| 256  | 1.90  | 2.62  | 11.81 | **6.20×**    |
| 512  | 1.69  | 2.60  | 10.47 | **6.19×**    |
| 1024 | 1.79  | 2.71  | 11.45 | **6.41×**    |

![Roofline: attention kernels against measured compute and bandwidth ceilings](docs/roofline.svg)

```
cargo run --release --bin roofline   # regenerates the figure above + its JSON
cargo run --release --bin bench      # zero-dep, stable Rust, both masks
cargo bench                          # criterion, with plots
cargo test --release                 # correctness vs naive, both masks
```
Reproduce with `RUSTFLAGS="-C target-cpu=native"`. The roofline binary measures
both ceilings on whatever machine runs it, stamps the figure with that machine,
and refuses to be quiet about it if the host was too contended to trust.

## What the numbers say

**Everything here is compute-bound, and that is the whole story.** The ridge
point on this machine — where the bandwidth ceiling crosses the compute ceiling
— sits at 5.25 FLOP/byte. Every kernel at every size lands to the *right* of it,
between 8 and 30 FLOP/byte. Nothing in this workload is waiting on memory, so
moving fewer bytes cannot be the lever that makes it faster. That single fact
predicts the rest of the table.

**Which is why tiling wins so little.** Flash Attention exists to avoid
materializing the `[n×n]` score matrix, cutting traffic from O(n²) to O(n·d).
The figure shows it doing exactly that — the tiled points sit at roughly twice
the arithmetic intensity of the naive ones, shifted a full step right. And it
buys 1.2–1.5×. On hardware where this problem were memory-bound, that horizontal
shift would translate straight into throughput; here it moves the point sideways
along a flat ceiling and the modest win that remains comes from doing fewer
passes over the scores, not from bandwidth. **This is the honest negative
result, and it is the interesting part of the project**: the optimization is
correctly implemented and simply mis-targeted at this size and `d`. Flash's real
win needs larger `n`, smaller cache, or the HBM-bound GPU regime it was designed
for.

**Vectorization is the lever that actually applies.** Being compute-bound means
the way up is issuing better instructions, and that is what AVX2 does — 4.8–6.4×,
moving the points vertically rather than horizontally. Most of it is the
8-wide dot product and accumulate; the last stretch came from replacing the
per-score scalar `exp` with an 8-wide polynomial (`src/vexp.rs`), which had been
the binding constraint on the softmax.

**And there is still 5× left on the table.** The best kernel reaches 13.6 of
78 GFLOP/s — about 17% of measured peak. The gap is the horizontal reduction at
the tail of every dot product, the scalar softmax bookkeeping between blocks,
and per-block loop overhead, none of which vectorize. That is a roofline
argument for where to look next, not a mystery.

**Causal masking is worth roughly 2× in wall-clock.** The tiled kernels skip any
KV block lying entirely above the diagonal, touching only the lower-triangular
half of the score space. Derived from the committed measurements, full-mask time
divided by causal time runs 1.5–2.1× across sizes (the spread is run-to-run
noise on a shared vCPU), against the 2.0× you would predict from halving the
query–key pairs. Note this does *not* show up in the GFLOP/s tables, which
already normalize it away by counting only unmasked pairs.

## AVX-512 (nightly)

`src/avx512.rs` implements 16-wide `dot`/`axpy` with a hardware
`_mm512_reduce_add_ps`. Build:

```
RUSTFLAGS="-C target-cpu=native --cfg avx512" cargo +nightly build --release
```

Expected gain over AVX2 is **sub-2×**, not 2×: the reduction and scalar softmax
tail don't widen, and some parts down-clock under sustained AVX-512 load. That
frequency-throttling crossover — where wider vectors stop paying — is the thing
worth measuring on the actual target part rather than assuming.

## Next steps

- Multi-head batching (the outer loop that makes this a real layer).
- Wire the AVX-512 kernel into the runtime dispatch, and measure it on a part
  that actually has AVX-512.
- Sweep `BLOCK` per cache level.
- Attack the 17%-of-peak gap: the per-dot-product horizontal reduction is the
  obvious first target.

## Layout

```
src/lib.rs       Mat type, reproducible fill, causal-aware FLOP counting
src/naive.rs     baseline (full + causal)
src/tiled.rs     online-softmax flash kernel with causal block-skipping
src/simd.rs      AVX2+FMA intrinsics + vectorized exp + runtime dispatch
src/vexp.rs      8-wide exp approximation
src/avx512.rs    16-wide kernel, cfg-gated for nightly / Rust 1.89+
src/roofline.rs  traffic model, measured machine ceilings, SVG writer
src/bin/bench.rs standalone benchmark (both masks)
src/bin/roofline.rs  regenerates docs/roofline.{svg,json}
benches/         criterion harness
tests/           correctness vs naive, both masks
docs/            generated roofline figure + the data behind it
```

## License

MIT — see [LICENSE](LICENSE).

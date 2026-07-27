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

**AVX2 with a vectorized softmax runs roughly 4–5.5× faster than the naive
baseline**, and **causal block-skipping is worth 2.0× wall-clock** against a
theoretical ideal of exactly 2.0×. The fastest kernel still reaches only ~22% of
the machine's measured compute ceiling, and the roofline below shows why: at
`d = 64` this workload is compute-bound at every size tested, so the
memory-traffic win that tiling exists for never gets to pay.

Single core of an x86_64 AMD EPYC with AVX2 + FMA (a shared VPS vCPU, pinned
with `taskset` — hence modest absolute throughput; the *ratios* are the point),
`head_dim = 64`, stable Rust with `-C target-cpu=native`. GFLOP/s counts only
unmasked query–key pairs, so the two masks are directly comparable. The tables
and the figure are one run of the committed generator, and `docs/roofline.json`
is its raw output.

**Full (bidirectional) mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 3.47  | 2.86  | 17.19 | **4.95×**    |
| 256  | 2.88  | 2.70  | 16.22 | **5.63×**    |
| 512  | 2.93  | 2.40  | 15.38 | **5.26×**    |
| 1024 | 2.77  | 2.24  | 14.11 | **5.10×**    |

**Causal mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 3.61  | 3.06  | 16.77 | **4.65×**    |
| 256  | 3.42  | 2.67  | 17.06 | **4.98×**    |
| 512  | 2.40  | 1.84  | 11.25 | **4.70×**    |
| 1024 | 2.58  | 2.02  | 12.62 | **4.89×**    |

**On reproducibility, since this is a shared vCPU.** The compute ceiling repeats
to within 2.7% across seven independent runs (76.8–78.9 GFLOP/s), and the causal
speedup lands on 2.01× mean against a 2.00× ideal. The kernel throughputs are
looser: individual configurations move by up to ~20% run to run, so the speedup
is stated as a range rather than to two decimals. Quoting "5.63×" as *the*
number would not survive a rerun, and the range is the honest claim.

![Roofline: attention kernels against measured compute and bandwidth ceilings](docs/roofline.svg)

```
cargo run --release --bin roofline   # regenerates the figure above + its JSON
cargo run --release --bin roofline -- --from-json docs/roofline.json   # re-render only
cargo run --release --bin blocksweep # tile size vs throughput
cargo run --release --bin bench      # both masks, plus measured causal speedup
cargo bench                          # criterion, with plots
cargo test --release                 # correctness vs naive, both masks
```
Reproduce with `RUSTFLAGS="-C target-cpu=native"`. The roofline binary measures
both ceilings on whatever machine runs it, stamps the figure with that machine,
and refuses to be quiet about it if the host was too contended to trust.

**A note on how the timing works, because it changed the answer.** Kernels are
not timed one exhaustively after another — they are interleaved round-robin, all
six combinations of kernel and mask together, with the starting position rotating
each round. Measured the old way, the baseline and the optimized kernel were
sampled during different seconds of wall-clock, so on a shared vCPU the drift
between those blocks landed entirely in their ratio: two back-to-back runs put
the same speedup anywhere from 4.2× to 8.3×, with almost all the movement in the
baseline rather than in `simd`. Interleaving makes it a paired comparison, and
the causal measurement promptly converged from a physically impossible 1.45–3.10×
onto 2.01× mean. Throughput is reported from the fastest run rather than the
median, because interference is one-sided — it can only ever make a run slower.
Every point also carries its interquartile spread, per point in the JSON and
worst-case on the figure.

## What the numbers say

**Everything here is compute-bound, and that is the whole story.** The ridge
point on this machine — where the bandwidth ceiling crosses the compute ceiling
— sits at 4.84 FLOP/byte. Every kernel at every size lands to the *right* of it,
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
the way up is issuing better instructions, and that is what AVX2 does — roughly
4–5.5×, moving the points vertically rather than horizontally. Most of it is the
8-wide dot product and accumulate. Two further steps came from following the
roofline rather than guessing: replacing the per-score scalar `exp` with an
8-wide polynomial (`src/vexp.rs`), which had been the binding constraint on the
softmax, and then amortizing the dot-product reduction (`dot4_avx2`) — see
below.

**And there is still ~4.5× left on the table — but the roofline says where.**
The best kernel reaches 17.2 of 78.9 GFLOP/s, about 22% of measured peak. That
ceiling is worth trusting: a dependency-free FMA probe, repeatable to 2.7% across
seven runs, working out to ~100% of what two 256-bit FMA ports can retire at this
core's clock. So the gap is real, and it is not in the FMAs — it is in everything
around them that does not vectorize.

The largest single item was the **horizontal reduction**. An 8-wide dot product
ends with `extractf128 + add + hadd + hadd`, which is serial, uses no FMA unit,
and costs the same whether the vector body was 8 elements or 64. At `d = 64`
that tail ran once per query-key pair. `dot4_avx2` now scores four keys against
the same query with four accumulators and collapses them in three `hadd`s plus
one 128-bit fold — because `hadd` interleaves two sources, `hadd(hadd(a,b),
hadd(c,d))` leaves all four partial sums in known lanes. Roughly a quarter of the
reduction work per key.

What remains is the scalar softmax bookkeeping between blocks and the `axpy`
accumulate, which re-reads the output row once per key rather than holding it in
registers across the block. Both are visible in the same way the reduction was:
as work that scales with query-key pairs but never touches an FMA port.

**Causal block-skipping pays, and the GFLOP/s tables cannot show it.** Those
tables divide by a causal-aware FLOP count, which normalizes the halved work
straight back out — so the only way to see what the block-skip is worth is a
stopwatch on the same kernel under both masks. `cargo run --release --bin bench`
does exactly that and prints the ratio against the ideal 2.0× you would predict
from the causal mask leaving n(n+1)/2 of n² query–key pairs. The per-point
median wall-clock is also in `docs/roofline.json`, so the ratio is checkable
without rerunning anything.

**The naive baseline cannot exploit the mask, and the data shows it.** Naive's
causal throughput runs 11–28% *below* its own full-mask throughput — look at the
orange points, which sit lower under the causal mask despite the FLOP count
already accounting for the halved work. That is structural, not noise. The
softmax pass in `src/naive.rs` walks the entire n-wide row whatever the mask,
and the masking step writes `-inf` across the whole upper triangle, so the
kernel pays O(n²) even when only half the pairs matter. Materializing the score
matrix does not just cost memory traffic; it forces you to touch entries you
have already decided to throw away. The tiled kernels skip those blocks and
never pay for them, which is the clearest single argument in the project for the
flash formulation.

## Multi-head: the outer loop that makes this a layer

Real attention runs `h` heads over the same sequence. They never interact until
the concatenation, so it is `h` independent instances of the single-head problem
— which forces a scheduling choice: parallelize *across* heads, or *within* one?

Across heads, and the roofline is the argument. Each head is already
compute-bound at `d = 64`, so cores are not waiting on memory and there is
nothing to win by splitting one head's working set between them. Splitting
within a head would mean sharing the running softmax state `m` and `l` across
threads — either synchronizing every KV block, or keeping per-thread partial
accumulators and merging them, which is the same rescaling dance the online
softmax already does, now with a barrier in it. Across heads needs none of that:
separate inputs, separate outputs, no shared mutable state. `std::thread::scope`
lets the workers borrow the inputs directly, so it costs no dependency and no
copying.

Scaling, `n = 512`, on an M1 Pro with 8 logical cores (a different machine from
the tables above — this measures thread scaling, not kernel throughput):

| heads | serial | parallel | speedup | of ideal |
|------:|-------:|---------:|--------:|---------:|
| 1     | 9.08ms | 9.16ms   | 0.99×   | 99%      |
| 2     | 18.40  | 9.29     | 1.98×   | 99%      |
| 4     | 36.33  | 9.48     | 3.83×   | 96%      |
| 8     | 72.82  | 16.43    | 4.43×   | 55%      |

Near-linear to 4 heads, then it falls off a cliff — because those 8 "logical
cores" are not interchangeable. This part has 6 performance cores and 2
efficiency cores, so the 7th and 8th heads land on cores that are several times
slower, and the join waits for them. That is the granularity cost of
across-heads parallelism, and it is the regime where within-head splitting would
start to earn its complexity.

## Tile size

`BLOCK` decides the inner loop's working set: `2 × BLOCK × d × 4` bytes of K and
V resident at once. `cargo run --release --bin blocksweep` measures the curve
across five const-generic instantiations rather than assuming it — the sizes are
compile-time constants in each, so it measures cache behaviour and not the cost
of a dynamic bound.

Worth running before trusting the default. On the M1 the curve is still climbing
at `BLOCK = 256` (128 KB, which fits that part's unusually large 128 KB L1d),
so the default 64 is leaving a few percent on the table there. `BLOCK = 64` is
sized for a conventional 32–48 KB L1d.

## AVX-512 (nightly)

`src/avx512.rs` carries the full 16-wide kernel — the same online-softmax
bookkeeping and causal block-skipping as the AVX2 path, over `_mm512_*`
primitives with a hardware `_mm512_reduce_add_ps` and a 16-wide `exp`
(`vexp::exp16`). It is wired into `simd::attention`'s runtime dispatch ahead of
the AVX2 arm, so a binary built with the cfg picks AVX-512 when the CPU has it
and falls through cleanly when it doesn't.

```
RUSTFLAGS="-C target-cpu=native --cfg avx512" cargo +nightly build --release
```

The module is gated because `_mm512_*` only stabilized in Rust 1.89 and the
default build targets an older floor — nothing outside the cfg needs a new
toolchain.

**No performance numbers here, because I have not run this on a part with
AVX-512.** The kernel is compiled and lint-clean under the cfg, and CI builds
and tests it on nightly, but every machine I have access to tops out at AVX2, so
the correctness test skips itself rather than pretend to cover something it
didn't.

What I'd *expect*, and would want to check against measurement: **sub-2×** over
AVX2, not the 2× the lane count suggests. The per-dot-product reduction and the
scalar softmax bookkeeping between blocks don't widen, and they're already the
binding constraint at 17% of peak. Against that, some parts down-clock under
sustained AVX-512 load, which can erase the gain entirely. That crossover —
where wider vectors stop paying — is the measurement worth making, and it is
explicitly projected, not observed.

## Next steps

- Measure the AVX-512 kernel on a part that actually has AVX-512.
- Keep closing the gap to peak: with the dot-product reduction amortized four
  ways, the next candidates are the scalar softmax bookkeeping between blocks
  and the `axpy` accumulate, which re-reads the output row per key.
- Within-head parallelism, for the case where heads are fewer than cores.
- A packed `[n, h*d]` layout, so multi-head does not need one `Mat` per head.

## Layout

```
src/lib.rs       Mat type, reproducible fill, causal-aware FLOP counting
src/naive.rs     baseline (full + causal)
src/tiled.rs     online-softmax flash kernel with causal block-skipping
src/simd.rs      AVX2+FMA intrinsics + vectorized exp + runtime dispatch
src/vexp.rs      8-wide exp approximation
src/avx512.rs    16-wide kernel, cfg-gated for nightly / Rust 1.89+
src/multihead.rs multi-head batching, serial and across-heads parallel
src/roofline.rs  traffic model, measured machine ceilings, SVG writer, JSON reader
src/bin/bench.rs both masks, measured causal speedup, multi-head scaling
src/bin/roofline.rs  regenerates docs/roofline.{svg,json}; --from-json re-renders
src/bin/blocksweep.rs  tile size vs throughput across cache levels
benches/         criterion harness
tests/           correctness vs naive: both masks, all tile sizes, multi-head
docs/            roofline figure + its data, and the online-softmax derivation
```

## Reading further

[`docs/online-softmax.md`](docs/online-softmax.md) derives the running-softmax
update from scratch — why one multiply repairs the entire accumulated history
when the max moves, why the result is exact rather than approximate, and why the
causal block-skip falls out of the same structure.

## License

MIT — see [LICENSE](LICENSE).

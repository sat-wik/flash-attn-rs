# flash-attn-rs

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

`head_dim = 64`, single core, `target-cpu=native` (AVX2 + FMA), stable Rust.
GFLOP/s (causal counts only the ~n²/2 unmasked pairs, so it's comparable):

**Full (bidirectional) mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 3.59  | 3.46  | 18.08 | **5.03×**    |
| 256  | 3.59  | 3.40  | 17.58 | **4.90×**    |
| 512  | 3.19  | 3.48  | 17.69 | **5.55×**    |
| 1024 | 3.54  | 3.70  | 18.61 | **5.25×**    |

**Causal mask**

| n    | naive | tiled | simd  | simd speedup |
|-----:|------:|------:|------:|-------------:|
| 128  | 3.58  | 3.74  | 16.13 | **4.50×**    |
| 256  | 3.49  | 3.47  | 17.60 | **5.04×**    |
| 512  | 3.39  | 3.54  | 18.46 | **5.45×**    |
| 1024 | 3.39  | 3.40  | 18.47 | **5.44×**    |

**Causal block-skipping, raw wall-clock (SIMD kernel):**
`n=512: 1.91× faster`, `n=1024: 1.96× faster` than the full mask — roughly the
2× you'd predict from doing half the query-key pairs.

```
cargo run --release --bin bench      # zero-dep, stable Rust, both masks
cargo bench                          # criterion, with plots
cargo test --release                 # correctness vs naive, both masks
```
Reproduce with `RUSTFLAGS="-C target-cpu=native"`.

## What the numbers say

**Vectorized exp bought the last ~0.7×.** The baseline SIMD kernel (scalar
`exp`) topped out around 4.5×; moving the softmax exponentials into an 8-wide
polynomial approximation (`src/vexp.rs`) lifted the full-mask speedup to
~5.0–5.5×. The remaining gap to the 8× lane-count ceiling is the horizontal
reduction at the tail of each dot product plus loop/bookkeeping overhead — a
roofline story, not a bug.

**Causal masking is the real algorithmic win: ~1.9× wall-clock.** The tiled
kernels skip any KV block that lies entirely above the diagonal, so they touch
only the lower-triangular half of the score space. This is also where tiling
finally beats naive on raw GFLOP/s at several sizes (e.g. n=128: 3.74 vs 3.58) —
the block-skip gives the flash formulation a structural advantage the naive
triple-loop can't get, because naive still walks masked columns.

**When tiling *doesn't* help (the honest part).** On the full mask at these
sizes and `d=64`, the naive `[n×n]` scores still fit in L2, so we're
compute-bound, not memory-bound — below the roofline ridge point, tiling's
online-softmax rescaling is pure overhead and buys nothing. Flash's memory-traffic
win shows up at larger `n`, smaller cache, or on bandwidth-bound hardware (the
HBM/GPU regime it was designed for and the silicon this targets). Knowing *when*
an optimization pays is the actual skill.

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
- A roofline plot: arithmetic intensity vs GFLOP/s, marking the compute-bound /
  memory-bound crossover so it's visible, not just argued.
- Sweep `BLOCK` per cache level; wire the AVX-512 kernel into the dispatch on a
  nightly CI job.

## Layout

```
src/lib.rs       Mat type, reproducible fill, causal-aware FLOP counting
src/naive.rs     baseline (full + causal)
src/tiled.rs     online-softmax flash kernel with causal block-skipping
src/simd.rs      AVX2+FMA intrinsics + vectorized exp + runtime dispatch
src/vexp.rs      8-wide exp approximation
src/avx512.rs    16-wide kernel, cfg-gated for nightly / Rust 1.89+
src/bin/bench.rs standalone benchmark (both masks)
benches/         criterion harness
tests/           correctness vs naive, both masks
```

## License

MIT — see [LICENSE](LICENSE).

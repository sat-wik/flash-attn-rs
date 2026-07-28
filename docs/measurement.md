# How these numbers were measured

The kernels here run on a shared VPS vCPU. Getting numbers off it that mean
anything took more work than writing some of the kernels did, and two of the
findings in the README only exist because the measurement was wrong first.

## Interleaving, not blocks

Kernels are timed **round-robin** — all six combinations of kernel and mask
together, one run each per round, with the starting position rotating so nobody
systematically eats the cold-cache cost of going first.

The obvious alternative, timing kernel A fifty times and then kernel B fifty
times, measures the two during different seconds of wall-clock. On a contended
host the available throughput drifts between those blocks, and the *ratio* —
which is the headline number — absorbs the drift whole.

That is not hypothetical. Measured in blocks, two back-to-back pinned runs put
the same AVX2 speedup anywhere from **4.2× to 8.3×**, and the per-kernel
breakdown showed why: `simd` moved 1–2% between runs while the baseline moved
31%. All the instability was in the denominator. Best-of-N does not help,
because it picks the cleanest sample *within* a block, not across blocks.

Interleaving makes it a paired comparison: inside one round every kernel sees
the same neighbours, clock and contention, so drift moves them together and
largely cancels on division. The causal measurement converged immediately, from
a physically impossible 1.45–3.10× onto 2.0× mean against a 2.00× ideal.

## Fastest run, not median

Throughput is reported from the **fastest** observed run. Interference on a
shared machine is one-sided — a stolen timeslice or an evicted cache line can
only ever make a run slower, never faster — so the minimum is the
least-contaminated estimate of how long the kernel actually takes. The median
estimates what the machine was doing to you, which is a different question.

It also keeps the operating points consistent with the ceilings, which take the
best of several probes. Comparing best-case ceilings against median-case points
would understate every percentage-of-peak on the figure.

Median and interquartile spread are still recorded per point in
`docs/roofline.json`, and the worst-case spread is printed on the figure itself.
A median with no spread beside it cannot tell you whether a difference is real.

## Measured ceilings, and whether the machine held still

Both roofline ceilings are measured on whatever machine runs the tool, never
read off a spec sheet:

- **Compute** — a dependency-free FMA chain that touches no memory, 64
  independent lanes. Width matters twice over: too narrow and the compiler will
  not vectorize the portable version, so it measures scalar throughput and the
  "ceiling" lands *below* the kernels it is meant to bound; and with fewer than
  8 independent AVX2 accumulators it measures FMA latency rather than
  throughput, reporting roughly half the real peak.
- **Bandwidth** — a STREAM triad over 64 MB buffers, past any last-level cache.
  Two warm-up passes, not one: timing the pass that faults 192 MB of pages in
  made a quiet machine look 33% contended.

The tool refuses to be quiet when the host is too noisy to trust. The compute
probe warns past 10% spread — it is a pure register loop, so on a core it owns
it repeats to a percent or two, which makes it a sharp detector of stolen time.
Bandwidth contends with the page cache and everything else, where 10–20% swing
is ordinary, so it only warns past 25%.

The compute ceiling is also **re-measured after all the kernel timings** and the
drift reported. The points are plotted against the ceiling, so if the machine
moved between the two, the figure compares numbers taken under different
conditions. On the committed run that drift is 1.4%.

## What the numbers actually reproduce to

Across ten runs on this host:

| quantity | reproducibility |
|---|---|
| compute ceiling | 74–79 GFLOP/s, a few percent |
| causal block-skip | 2.0× mean, ideal 2.00× |
| `simd` throughput | within 2% at 5 of 8 configurations, 16% worst |
| `simd` vs `naive` | stated as a range; a two-decimal figure would not survive a rerun |
| `tiled` vs `naive` | 0.65×–1.48×, i.e. not resolvable |

The multi-head benchmark prints a one-head row as a control: with a single head
the parallel path falls back to the serial one, so both columns run *identical
code*. It reads 0.89×, which puts the noise floor for that measurement at ~11%.

## Three claims this project got wrong

Worth recording, because the corrections are more informative than the numbers.

**Tiling won 1.2–1.5×.** From a single unpinned, non-interleaved run — the same
setup that put the AVX2 speedup between 4.2× and 8.3×.

**Then tiling lost 0.65–1.02×.** From two pinned, interleaved runs that agreed.
Also wrong: two runs agreeing is not evidence when the spread on each
measurement is 9–23%.

**Now: no reliable difference.** Ten runs put the ratio anywhere from 0.65× to
1.48× with no change to either kernel's source. The effect is smaller than the
noise around it, and the null result is what the roofline predicted from the
start.

An open question left over: the sign of that ratio tracks which commit was
built, though the commits in question touched only `simd.rs` and never
`naive.rs` or `tiled.rs`. Four runs is not enough to call it and there is no
mechanism to point at — but if it is real, the interleaved sampler is perturbing
the measurements it exists to protect.

## Reproducing

```
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin roofline
```

On a shared host, pin it: `taskset -c 0 env RUSTFLAGS=...`. Run it twice and
compare the ceiling lines; if they disagree by more than a few percent, the box
is too contended and the figure is not worth committing. `--from-json` re-renders
an existing dataset without measuring anything, which is what you want when the
committed figure came from hardware you do not have.

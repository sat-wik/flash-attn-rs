//! Criterion harness, for per-kernel distribution statistics.
//!
//! Note what this is *not* good for. Criterion benchmarks each kernel
//! independently, one after another, which is exactly the measurement shape
//! `docs/measurement.md` documents going wrong on a shared host: the ratio
//! between two kernels measured in separate blocks absorbs whatever the machine
//! did between them. Use `--bin bench` or `--bin roofline` for anything where
//! the comparison is the point — those interleave and pair.
//!
//! What criterion gives that the hand-rolled tooling does not is a proper
//! sampling distribution per kernel, outlier classification and HTML reports.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flash_attn_rs::{attention_flops, filled, naive, simd, tiled};

fn bench(c: &mut Criterion) {
    let d = 64;
    let seq_lengths = [128usize, 256, 512, 1024];
    for &causal in &[false, true] {
        let tag = if causal { "causal" } else { "full" };
        let mut group = c.benchmark_group(format!("attention-{tag}"));
        for &n in &seq_lengths {
            let q = filled(n, d, 1);
            let k = filled(n, d, 2);
            let v = filled(n, d, 3);
            group.throughput(Throughput::Elements(attention_flops(n, d, causal) as u64));
            group.bench_with_input(BenchmarkId::new("naive", n), &n, |b, _| {
                b.iter(|| naive::attention(&q, &k, &v, causal))
            });
            group.bench_with_input(BenchmarkId::new("tiled", n), &n, |b, _| {
                b.iter(|| tiled::attention(&q, &k, &v, causal))
            });
            group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
                b.iter(|| simd::attention(&q, &k, &v, causal))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);

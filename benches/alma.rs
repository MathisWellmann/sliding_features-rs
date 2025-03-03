use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rng, Rng};
use sliding_features::{pure_functions::Echo, sliding_windows::Alma, View};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 1_000_000;
    let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));

    let mut group = c.benchmark_group("alma_1M");
    group.bench_function("f32", |b| {
        b.iter(|| {
            let mut alma = Alma::new(Echo::new(), 1024);
            for v in vals.iter() {
                alma.update(*v);
                let _ = black_box(alma.last());
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

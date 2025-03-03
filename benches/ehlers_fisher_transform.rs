use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rng, Rng};
use sliding_features::{
    pure_functions::Echo,
    sliding_windows::{Alma, EhlersFisherTransform, Ema},
    View,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("ehlers_fisher_transform_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut alma = EhlersFisherTransform::<f64, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in vals.iter() {
                alma.update(*v);
                let _ = black_box(alma.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut alma = EhlersFisherTransform::<f32, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in vals.iter() {
                alma.update(*v);
                let _ = black_box(alma.last());
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

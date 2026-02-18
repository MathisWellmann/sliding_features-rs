use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use rand::{
    Rng,
    rng,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::{
        Ema,
        PolarizedFractalEfficiency,
    },
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("polarized_fractal_efficiency_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = PolarizedFractalEfficiency::<f64, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), NonZeroUsize::new(1024).unwrap()),
                NonZeroUsize::new(1024).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = PolarizedFractalEfficiency::<f32, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), NonZeroUsize::new(1024).unwrap()),
                NonZeroUsize::new(1024).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

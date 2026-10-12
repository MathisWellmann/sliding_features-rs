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
    RngExt,
    rng,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::{
        RollingQuantile,
        Skewness,
    },
};

const N: usize = 100_000;

fn rolling_quantile(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("rolling_quantile_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view =
                RollingQuantile::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap(), 0.25);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view =
                RollingQuantile::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap(), 0.25);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn skewness(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("skewness_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Skewness::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Skewness::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, rolling_quantile, skewness);
criterion_main!(benches);

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
        ConditionalValueAtRisk,
        ValueAtRisk,
    },
};

const N: usize = 100_000;

fn value_at_risk(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("value_at_risk_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random::<f64>() - 0.5));
        b.iter(|| {
            let mut view =
                ValueAtRisk::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap(), 0.95);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random::<f32>() - 0.5));
        b.iter(|| {
            let mut view =
                ValueAtRisk::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap(), 0.95);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn conditional_value_at_risk(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("conditional_value_at_risk_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random::<f64>() - 0.5));
        b.iter(|| {
            let mut view = ConditionalValueAtRisk::<f64, _>::new(
                Echo::new(),
                NonZeroUsize::new(1024).unwrap(),
                0.95,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random::<f32>() - 0.5));
        b.iter(|| {
            let mut view = ConditionalValueAtRisk::<f32, _>::new(
                Echo::new(),
                NonZeroUsize::new(1024).unwrap(),
                0.95,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, value_at_risk, conditional_value_at_risk);
criterion_main!(benches);

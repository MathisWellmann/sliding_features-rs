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
        AdaptiveRsi,
        AutocorrelationPeriodogram,
        EvenBetterSinewave,
    },
};

const N: usize = 100_000;

fn autocorrelation_periodogram(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("autocorrelation_periodogram_100k");
    group.sample_size(10);
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = AutocorrelationPeriodogram::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = AutocorrelationPeriodogram::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn adaptive_rsi(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("adaptive_rsi_100k");
    group.sample_size(10);
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = AdaptiveRsi::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = AdaptiveRsi::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn even_better_sinewave(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("even_better_sinewave_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view =
                EvenBetterSinewave::<f64, _>::new(Echo::new(), NonZeroUsize::new(40).unwrap());
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
                EvenBetterSinewave::<f32, _>::new(Echo::new(), NonZeroUsize::new(40).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(
    benches,
    autocorrelation_periodogram,
    even_better_sinewave,
    adaptive_rsi
);
criterion_main!(benches);

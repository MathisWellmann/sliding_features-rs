// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use num::Float;
use rand::{
    RngExt,
    rng,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::{
        Autocorrelation,
        HurstExponent,
        OuHalfLife,
        PermutationEntropy,
        SampleEntropy,
        ShannonEntropy,
        VarianceRatio,
    },
};

const N: usize = 100_000;

/// Random values in `[0, 1)`.
fn random_vals<T: Float>() -> Vec<T> {
    let mut rng = rng();
    Vec::from_iter((0..N).map(|_| T::from(rng.random::<f64>()).expect("can convert")))
}

/// Feed all `vals` into a fresh view, reading `last` after every update like a live consumer.
fn run<T: Float, V: View<T>>(mut view: V, vals: &[T]) {
    for v in vals.iter() {
        view.update(*v);
        let _ = black_box(view.last());
    }
}

fn shannon_entropy(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let bins = NonZeroUsize::new(32).unwrap();
    let mut group = c.benchmark_group("shannon_entropy_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| {
            run(
                ShannonEntropy::new(Echo::new(), window, bins, 0.0, 1.0),
                &vals,
            )
        })
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| {
            run(
                ShannonEntropy::new(Echo::new(), window, bins, 0.0, 1.0),
                &vals,
            )
        })
    });
}

fn permutation_entropy(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let delay = NonZeroUsize::new(1).unwrap();
    let mut group = c.benchmark_group("permutation_entropy_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| {
            run(
                PermutationEntropy::new(Echo::new(), window, 5, delay),
                &vals,
            )
        })
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| {
            run(
                PermutationEntropy::new(Echo::new(), window, 5, delay),
                &vals,
            )
        })
    });
}

fn sample_entropy(c: &mut Criterion) {
    // O(n^2) per update, so a small window keeps the bench fast.
    let window = NonZeroUsize::new(32).unwrap();
    let m = NonZeroUsize::new(2).unwrap();
    let mut group = c.benchmark_group("sample_entropy_100k");
    group.sample_size(10);
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| run(SampleEntropy::new(Echo::new(), window, m, 0.2), &vals))
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| run(SampleEntropy::new(Echo::new(), window, m, 0.2), &vals))
    });
}

fn hurst_exponent(c: &mut Criterion) {
    // O(n log n) per update, so a moderate window keeps the bench fast.
    let window = NonZeroUsize::new(128).unwrap();
    let mut group = c.benchmark_group("hurst_exponent_100k");
    group.sample_size(10);
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| run(HurstExponent::new(Echo::new(), window), &vals))
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| run(HurstExponent::new(Echo::new(), window), &vals))
    });
}

fn variance_ratio(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let mut group = c.benchmark_group("variance_ratio_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| run(VarianceRatio::new(Echo::new(), window, 8), &vals))
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| run(VarianceRatio::new(Echo::new(), window, 8), &vals))
    });
}

fn autocorrelation(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let lag = NonZeroUsize::new(1).unwrap();
    let mut group = c.benchmark_group("autocorrelation_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| run(Autocorrelation::new(Echo::new(), window, lag), &vals))
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| run(Autocorrelation::new(Echo::new(), window, lag), &vals))
    });
}

fn ou_half_life(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let mut group = c.benchmark_group("ou_half_life_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| run(OuHalfLife::new(Echo::new(), window), &vals))
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| run(OuHalfLife::new(Echo::new(), window), &vals))
    });
}

criterion_group!(
    benches,
    shannon_entropy,
    permutation_entropy,
    sample_entropy,
    hurst_exponent,
    variance_ratio,
    autocorrelation,
    ou_half_life
);
criterion_main!(benches);

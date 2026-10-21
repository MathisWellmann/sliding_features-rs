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
use rand::{
    RngExt,
    rng,
};
use sliding_features::{
    View,
    pure_functions::{
        Echo,
        InverseFisherTransform,
    },
    sliding_windows::{
        EhlersStochastic,
        ReversionIndex,
        UniversalOscillator,
    },
};

fn bench_inverse_fisher_transform(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("inverse_fisher_transform_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = InverseFisherTransform::new(Echo::new(), 1.5_f64);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = InverseFisherTransform::new(Echo::new(), 1.5_f32);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn bench_ehlers_stochastic(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("ehlers_stochastic_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = EhlersStochastic::new(
                Echo::new(),
                NonZeroUsize::new(48).unwrap(),
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(10).unwrap(),
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
            let mut view = EhlersStochastic::new(
                Echo::new(),
                NonZeroUsize::new(48).unwrap(),
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(10).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn bench_universal_oscillator(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("universal_oscillator_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(20).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(20).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn bench_reversion_index(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("reversion_index_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = ReversionIndex::new(
                Echo::new(),
                NonZeroUsize::new(20).unwrap(),
                NonZeroUsize::new(8).unwrap(),
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
            let mut view = ReversionIndex::new(
                Echo::new(),
                NonZeroUsize::new(20).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(
    benches,
    bench_inverse_fisher_transform,
    bench_ehlers_stochastic,
    bench_universal_oscillator,
    bench_reversion_index
);
criterion_main!(benches);

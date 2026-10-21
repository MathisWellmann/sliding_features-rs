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
use sliding_features::{
    View,
    pure_functions::Echo,
    rolling::{
        AlphaBetaFilter,
        HuberEma,
        Kalman1d,
        OneEuroFilter,
    },
    sliding_windows::{
        HampelFilter,
        LmsFilter,
        NlmsFilter,
        RlsFilter,
    },
};

fn benchmark_one_euro(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(1);
    const N: usize = 100_000;

    let mut group = c.benchmark_group("one_euro_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = OneEuroFilter::<f64, _>::new(Echo::new(), 1.0, 0.1, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = OneEuroFilter::<f32, _>::new(Echo::new(), 1.0, 0.1, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_alpha_beta(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(2);
    const N: usize = 100_000;

    let mut group = c.benchmark_group("alpha_beta_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = AlphaBetaFilter::<f64, _>::new(Echo::new(), 0.5, 0.2);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = AlphaBetaFilter::<f32, _>::new(Echo::new(), 0.5, 0.2);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_kalman1d(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(3);
    const N: usize = 100_000;

    let mut group = c.benchmark_group("kalman1d_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = Kalman1d::<f64, _>::new(Echo::new(), 0.01, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = Kalman1d::<f32, _>::new(Echo::new(), 0.01, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_lms(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(4);
    const N: usize = 100_000;
    let order = NonZeroUsize::new(8).expect("non zero");

    let mut group = c.benchmark_group("lms_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = LmsFilter::<f64, _>::new(Echo::new(), order, 0.01);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = LmsFilter::<f32, _>::new(Echo::new(), order, 0.01);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_nlms(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(5);
    const N: usize = 100_000;
    let order = NonZeroUsize::new(8).expect("non zero");

    let mut group = c.benchmark_group("nlms_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = NlmsFilter::<f64, _>::new(Echo::new(), order, 0.5, 1e-6);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = NlmsFilter::<f32, _>::new(Echo::new(), order, 0.5, 1e-6);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_rls(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(6);
    const N: usize = 100_000;
    let order = NonZeroUsize::new(8).expect("non zero");

    let mut group = c.benchmark_group("rls_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = RlsFilter::<f64, _>::new(Echo::new(), order, 0.995, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = RlsFilter::<f32, _>::new(Echo::new(), order, 0.995, 1.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_hampel(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(8);
    const N: usize = 100_000;
    let window_len = NonZeroUsize::new(11).expect("non zero");

    let mut group = c.benchmark_group("hampel_filter_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = HampelFilter::<f64, _>::new(Echo::new(), window_len, 3.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = HampelFilter::<f32, _>::new(Echo::new(), window_len, 3.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn benchmark_huber_ema(c: &mut Criterion) {
    let rng = romu::Rng::from_seed_with_64bit(7);
    const N: usize = 100_000;

    let mut group = c.benchmark_group("huber_ema_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.f64()));
        b.iter(|| {
            let mut view = HuberEma::<f64, _>::new(Echo::new(), 0.1, 0.1, 3.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.f64() as f32));
        b.iter(|| {
            let mut view = HuberEma::<f32, _>::new(Echo::new(), 0.1, 0.1, 3.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_one_euro,
    benchmark_alpha_beta,
    benchmark_kalman1d,
    benchmark_lms,
    benchmark_nlms,
    benchmark_rls,
    benchmark_huber_ema,
    benchmark_hampel
);
criterion_main!(benches);

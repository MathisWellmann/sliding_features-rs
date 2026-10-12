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
        BandPass,
        HighPass,
        HighPassPoles,
        UltimateSmoother,
    },
};

const N: usize = 100_000;

/// Benchmarks `N` updates of the view built by `$make`, for `f64` and `f32`.
/// A macro because the view type depends on the float type.
/// Inputs are positive, as some indicators divide by the price.
macro_rules! bench_view {
    ($c:expr, $name:expr, $make:expr) => {{
        let mut rng = rng();
        let mut group = $c.benchmark_group($name);
        group.bench_function("f64", |b| {
            let vals = Vec::<f64>::from_iter((0..N).map(|_| 1.0 + rng.random::<f64>()));
            b.iter(|| {
                let mut view = $make;
                for v in vals.iter() {
                    view.update(*v);
                    let _ = black_box(view.last());
                }
            })
        });
        group.bench_function("f32", |b| {
            let vals = Vec::<f32>::from_iter((0..N).map(|_| 1.0 + rng.random::<f32>()));
            b.iter(|| {
                let mut view = $make;
                for v in vals.iter() {
                    view.update(*v);
                    let _ = black_box(view.last());
                }
            })
        });
    }};
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("benchmark parameters are non-zero")
}

fn high_pass(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("high_pass_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = HighPass::<f64, _>::new(
                Echo::new(),
                NonZeroUsize::new(48).unwrap(),
                HighPassPoles::Two,
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
            let mut view = HighPass::<f32, _>::new(
                Echo::new(),
                NonZeroUsize::new(48).unwrap(),
                HighPassPoles::Two,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn band_pass(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("band_pass_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view =
                BandPass::<f64, _>::new(Echo::new(), NonZeroUsize::new(20).unwrap(), 0.3);
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
                BandPass::<f32, _>::new(Echo::new(), NonZeroUsize::new(20).unwrap(), 0.3);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn ultimate_smoother(c: &mut Criterion) {
    bench_view!(
        c,
        "ultimate_smoother_100k",
        UltimateSmoother::new(Echo::new(), nz(20))
    );
}

criterion_group!(benches, high_pass, band_pass, ultimate_smoother);
criterion_main!(benches);

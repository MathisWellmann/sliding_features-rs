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
        Butterworth,
        ButterworthPoles,
        Decycler,
        DecyclerOscillator,
        EhlersFilter,
        GaussianFilter,
        GaussianPoles,
        HighPass,
        HighPassPoles,
        UltimateBands,
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

fn ultimate_bands(c: &mut Criterion) {
    bench_view!(
        c,
        "ultimate_bands_100k",
        UltimateBands::new(Echo::new(), nz(20), 1.0)
    );
}

fn decycler(c: &mut Criterion) {
    bench_view!(
        c,
        "decycler_100k",
        Decycler::new(Echo::new(), nz(125), HighPassPoles::Two)
    );
}

fn decycler_oscillator(c: &mut Criterion) {
    bench_view!(
        c,
        "decycler_oscillator_100k",
        DecyclerOscillator::new(Echo::new(), nz(125), 1.0)
    );
}

fn gaussian_filter(c: &mut Criterion) {
    bench_view!(
        c,
        "gaussian_filter_100k",
        GaussianFilter::new(Echo::new(), nz(20), GaussianPoles::Four)
    );
}

fn butterworth(c: &mut Criterion) {
    bench_view!(
        c,
        "butterworth_100k",
        Butterworth::new(Echo::new(), nz(20), ButterworthPoles::Three)
    );
}

fn ehlers_filter(c: &mut Criterion) {
    bench_view!(
        c,
        "ehlers_filter_100k",
        EhlersFilter::new(Echo::new(), nz(15))
    );
}

criterion_group!(
    benches,
    high_pass,
    band_pass,
    ultimate_smoother,
    ultimate_bands,
    decycler,
    decycler_oscillator,
    gaussian_filter,
    butterworth,
    ehlers_filter
);
criterion_main!(benches);

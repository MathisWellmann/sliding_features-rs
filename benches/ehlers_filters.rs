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
    },
};

const N: usize = 100_000;

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

criterion_group!(benches, high_pass, band_pass);
criterion_main!(benches);

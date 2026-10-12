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
    rolling::{
        Cusum,
        PageHinkley,
        PageHinkleyDirection,
    },
    sliding_windows::Adwin,
};

const N: usize = 100_000;

fn cusum(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("cusum_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Cusum::<f64, _>::new(Echo::new(), 0.5, 0.3, 0.5, 5.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Cusum::<f32, _>::new(Echo::new(), 0.5, 0.3, 0.5, 5.0);
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn page_hinkley(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("page_hinkley_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = PageHinkley::<f64, _>::new(
                Echo::new(),
                PageHinkleyDirection::Both,
                NonZeroUsize::new(30).unwrap(),
                0.005,
                50.0,
                1.0 - 0.0001,
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
            let mut view = PageHinkley::<f32, _>::new(
                Echo::new(),
                PageHinkleyDirection::Both,
                NonZeroUsize::new(30).unwrap(),
                0.005,
                50.0,
                1.0 - 0.0001,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn adwin(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("adwin_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Adwin::<f64, _>::new(
                Echo::new(),
                0.002,
                NonZeroUsize::new(5).unwrap(),
                NonZeroUsize::new(32).unwrap(),
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
            let mut view = Adwin::<f32, _>::new(
                Echo::new(),
                0.002,
                NonZeroUsize::new(5).unwrap(),
                NonZeroUsize::new(32).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, cusum, page_hinkley, adwin);
criterion_main!(benches);

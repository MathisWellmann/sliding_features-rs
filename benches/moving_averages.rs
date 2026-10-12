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
        Dema,
        Hma,
        Rma,
        T3,
        Tema,
        Wma,
        Zlema,
    },
};

const N: usize = 100_000;

fn wma(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("wma_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Wma::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Wma::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn rma(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("rma_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Rma::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Rma::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn dema(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("dema_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Dema::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Dema::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn tema(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("tema_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Tema::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Tema::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn t3(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("t3_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = T3::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = T3::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn zlema(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("zlema_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Zlema::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Zlema::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn hma(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("hma_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Hma::<f64, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Hma::<f32, _>::new(Echo::new(), NonZeroUsize::new(1024).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, wma, rma, dema, tema, t3, zlema, hma);
criterion_main!(benches);

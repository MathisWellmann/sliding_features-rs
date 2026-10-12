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
        Apo,
        ConnorsRsi,
        Coppock,
        Dpo,
        Macd,
        Ppo,
        Qqe,
        Rsx,
        SchaffTrendCycle,
        StochRsi,
        Trix,
        Tsi,
    },
};

fn macd_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("macd_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Macd::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Macd::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn ppo_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("ppo_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Ppo::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Ppo::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn apo_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("apo_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Apo::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Apo::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn trix_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("trix_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Trix::<f64, _>::new(Echo::new(), NonZeroUsize::new(16).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Trix::<f32, _>::new(Echo::new(), NonZeroUsize::new(16).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn tsi_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("tsi_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Tsi::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Tsi::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn stoch_rsi_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("stoch_rsi_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = StochRsi::<f64, _>::new(
                Echo::new(),
                NonZeroUsize::new(14).unwrap(),
                NonZeroUsize::new(14).unwrap(),
                NonZeroUsize::new(3).unwrap(),
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
            let mut view = StochRsi::<f32, _>::new(
                Echo::new(),
                NonZeroUsize::new(14).unwrap(),
                NonZeroUsize::new(14).unwrap(),
                NonZeroUsize::new(3).unwrap(),
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn rsx_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("rsx_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Rsx::<f64, _>::new(Echo::new(), NonZeroUsize::new(14).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Rsx::<f32, _>::new(Echo::new(), NonZeroUsize::new(14).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn schaff_trend_cycle_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("schaff_trend_cycle_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = SchaffTrendCycle::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = SchaffTrendCycle::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn qqe_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("qqe_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Qqe::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Qqe::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn connors_rsi_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("connors_rsi_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = ConnorsRsi::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = ConnorsRsi::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn dpo_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("dpo_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Dpo::<f64, _>::new(Echo::new(), NonZeroUsize::new(20).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Dpo::<f32, _>::new(Echo::new(), NonZeroUsize::new(20).unwrap());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn coppock_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("coppock_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Coppock::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = Coppock::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(
    benches,
    macd_benchmark,
    ppo_benchmark,
    apo_benchmark,
    trix_benchmark,
    tsi_benchmark,
    stoch_rsi_benchmark,
    rsx_benchmark,
    schaff_trend_cycle_benchmark,
    qqe_benchmark,
    connors_rsi_benchmark,
    dpo_benchmark,
    coppock_benchmark
);
criterion_main!(benches);

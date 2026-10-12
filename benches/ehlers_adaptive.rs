use std::hint::black_box;

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
    sliding_windows::AutocorrelationPeriodogram,
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

criterion_group!(benches, autocorrelation_periodogram);
criterion_main!(benches);

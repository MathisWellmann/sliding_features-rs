use std::hint::black_box;

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    rolling::OneEuroFilter,
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

criterion_group!(benches, benchmark_one_euro);
criterion_main!(benches);

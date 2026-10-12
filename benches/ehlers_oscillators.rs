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
    pure_functions::{
        Echo,
        InverseFisherTransform,
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

criterion_group!(benches, bench_inverse_fisher_transform);
criterion_main!(benches);

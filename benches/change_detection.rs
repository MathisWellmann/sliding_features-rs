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
    rolling::Cusum,
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

criterion_group!(benches, cusum);
criterion_main!(benches);

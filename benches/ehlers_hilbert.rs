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
    sliding_windows::HomodyneDominantCycle,
};

const N: usize = 100_000;

fn homodyne_dominant_cycle(c: &mut Criterion) {
    let mut rng = rng();
    let mut group = c.benchmark_group("homodyne_dominant_cycle_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = HomodyneDominantCycle::<f64, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = HomodyneDominantCycle::<f32, _>::new(Echo::new());
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, homodyne_dominant_cycle);
criterion_main!(benches);

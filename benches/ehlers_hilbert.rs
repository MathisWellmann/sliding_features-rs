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
    sliding_windows::{
        HilbertPhasor,
        HomodyneDominantCycle,
    },
};

const N: usize = 100_000;

/// Benchmark 100k updates of the views built by `new_f64` and `new_f32` on random values.
fn bench_100k<V64, V32>(
    c: &mut Criterion,
    name: &str,
    new_f64: impl Fn() -> V64,
    new_f32: impl Fn() -> V32,
) where
    V64: View<f64>,
    V32: View<f32>,
{
    let mut rng = rng();
    let mut group = c.benchmark_group(format!("{name}_100k"));
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = new_f64();
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = new_f32();
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

fn homodyne_dominant_cycle(c: &mut Criterion) {
    bench_100k(
        c,
        "homodyne_dominant_cycle",
        || HomodyneDominantCycle::<f64, _>::new(Echo::new()),
        || HomodyneDominantCycle::<f32, _>::new(Echo::new()),
    );
}

fn hilbert_phasor(c: &mut Criterion) {
    bench_100k(
        c,
        "hilbert_phasor",
        || HilbertPhasor::<f64, _>::new(Echo::new()),
        || HilbertPhasor::<f32, _>::new(Echo::new()),
    );
}

criterion_group!(benches, homodyne_dominant_cycle, hilbert_phasor);
criterion_main!(benches);

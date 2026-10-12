use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use num::Float;
use rand::{
    RngExt,
    rng,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::{
        PermutationEntropy,
        ShannonEntropy,
    },
};

const N: usize = 100_000;

/// Random values in `[0, 1)`.
fn random_vals<T: Float>() -> Vec<T> {
    let mut rng = rng();
    Vec::from_iter((0..N).map(|_| T::from(rng.random::<f64>()).expect("can convert")))
}

/// Feed all `vals` into a fresh view, reading `last` after every update like a live consumer.
fn run<T: Float, V: View<T>>(mut view: V, vals: &[T]) {
    for v in vals.iter() {
        view.update(*v);
        let _ = black_box(view.last());
    }
}

fn shannon_entropy(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let bins = NonZeroUsize::new(32).unwrap();
    let mut group = c.benchmark_group("shannon_entropy_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| {
            run(
                ShannonEntropy::new(Echo::new(), window, bins, 0.0, 1.0),
                &vals,
            )
        })
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| {
            run(
                ShannonEntropy::new(Echo::new(), window, bins, 0.0, 1.0),
                &vals,
            )
        })
    });
}

fn permutation_entropy(c: &mut Criterion) {
    let window = NonZeroUsize::new(1024).unwrap();
    let delay = NonZeroUsize::new(1).unwrap();
    let mut group = c.benchmark_group("permutation_entropy_100k");
    group.bench_function("f64", |b| {
        let vals = random_vals::<f64>();
        b.iter(|| {
            run(
                PermutationEntropy::new(Echo::new(), window, 5, delay),
                &vals,
            )
        })
    });
    group.bench_function("f32", |b| {
        let vals = random_vals::<f32>();
        b.iter(|| {
            run(
                PermutationEntropy::new(Echo::new(), window, 5, delay),
                &vals,
            )
        })
    });
}

criterion_group!(benches, shannon_entropy, permutation_entropy);
criterion_main!(benches);

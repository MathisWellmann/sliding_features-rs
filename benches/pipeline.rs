//! A pipeline against the same feature as a hand-written generic chain, and
//! a feature set as one pipeline against one pipeline per feature.

use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use sliding_features::{
    View,
    pipeline::{
        Feature,
        Pipeline,
    },
    pure_functions::{
        Echo,
        Subtract,
    },
    sliding_windows::{
        Ema,
        Sma,
        ZScoreStandardization,
    },
};

const N: usize = 100_000;

fn random_walk() -> Vec<f64> {
    let rng = romu::Rng::from_seed_with_64bit(1);
    let mut price = 100.0;
    Vec::from_iter((0..N).map(|_| {
        price += rng.f64() - 0.5;
        price
    }))
}

fn feature(source: &str) -> Feature {
    let pipeline: Pipeline = source.parse().expect("compiles");
    Feature::try_from(pipeline).expect("one input and one output")
}

/// Push every value through a view and read it, as a feature loop does.
fn drive(view: &mut impl View<f64>, vals: &[f64]) {
    for v in vals {
        view.update(*v);
        let _ = black_box(view.last());
    }
}

/// One feature: the cost of interpreting the program against code the
/// compiler monomorphized.
fn one_feature(c: &mut Criterion) {
    let vals = random_walk();
    let source = "zscore(ema(x, 10) - sma(x, 50), 200)";
    let window = |n| NonZeroUsize::new(n).expect("non zero");
    let mut group = c.benchmark_group("pipeline_one_feature_100k");
    group.bench_function("static_chain", |b| {
        b.iter(|| {
            let mut view = ZScoreStandardization::new(
                Subtract::new(
                    Ema::new(Echo::new(), window(10)),
                    Sma::new(Echo::new(), window(50)),
                ),
                window(200),
            );
            drive(&mut view, &vals);
        })
    });
    group.bench_function("pipeline", |b| {
        b.iter(|| drive(&mut feature(source), &vals))
    });
    group.finish();
}

/// Mostly stateless nodes: the per-operation overhead of the interpreter.
fn stateless_heavy(c: &mut Criterion) {
    let vals = random_walk();
    let source = "tanh(abs(-(-(-(-(x * 2 + 1)))) - ema(x, 5)) * 0.01)";
    let mut group = c.benchmark_group("pipeline_stateless_heavy_100k");
    group.bench_function("pipeline", |b| {
        b.iter(|| drive(&mut feature(source), &vals))
    });
    group.finish();
}

/// A feature set with shared subexpressions: one pipeline computes each of
/// them once, one pipeline per feature computes them per feature.
fn feature_set(c: &mut Criterion) {
    let vals = random_walk();
    let sources = [
        "std(ln_return(x), 20) / std(ln_return(x), 100)",
        "std(ln_return(x), 20)",
        "zscore(ema(x, 10) - ema(x, 50), 200)",
        "mad(ema(x, 10) - ema(x, 50), 100)",
        "roc(ema(x, 10), 10)",
        "rsi(x, 14)",
        "ema(rsi(x, 14), 5)",
        "tanh(zscore(ln_return(x), 100))",
    ];
    let mut group = c.benchmark_group("pipeline_feature_set_100k");
    group.bench_function("pipeline_per_feature", |b| {
        b.iter(|| {
            let mut features = Vec::from_iter(sources.iter().map(|s| feature(s)));
            for v in &vals {
                for feature in &mut features {
                    feature.update(*v);
                    let _ = black_box(feature.last());
                }
            }
        })
    });
    group.bench_function("one_pipeline", |b| {
        let source = sources.join("\n");
        b.iter(|| {
            let mut pipeline: Pipeline = source.parse().expect("compiles");
            for v in &vals {
                pipeline.update(&[*v]);
                for out in pipeline.outputs() {
                    let _ = black_box(out);
                }
            }
        })
    });
    group.finish();
}

criterion_group!(benches, one_feature, stateless_heavy, feature_set);
criterion_main!(benches);

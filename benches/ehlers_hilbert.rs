// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
        HilbertTrendMode,
        HomodyneDominantCycle,
        InstantaneousTrendline,
        Mama,
        SineWave,
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

fn sine_wave(c: &mut Criterion) {
    bench_100k(
        c,
        "sine_wave",
        || SineWave::<f64, _>::new(Echo::new()),
        || SineWave::<f32, _>::new(Echo::new()),
    );
}

fn instantaneous_trendline(c: &mut Criterion) {
    bench_100k(
        c,
        "instantaneous_trendline",
        || InstantaneousTrendline::<f64, _>::new(Echo::new()),
        || InstantaneousTrendline::<f32, _>::new(Echo::new()),
    );
}

fn hilbert_trend_mode(c: &mut Criterion) {
    bench_100k(
        c,
        "hilbert_trend_mode",
        || HilbertTrendMode::<f64, _>::new(Echo::new()),
        || HilbertTrendMode::<f32, _>::new(Echo::new()),
    );
}

fn mama(c: &mut Criterion) {
    bench_100k(
        c,
        "mama",
        || Mama::<f64, _>::new(Echo::new()),
        || Mama::<f32, _>::new(Echo::new()),
    );
}

criterion_group!(
    benches,
    homodyne_dominant_cycle,
    hilbert_phasor,
    sine_wave,
    instantaneous_trendline,
    hilbert_trend_mode,
    mama
);
criterion_main!(benches);

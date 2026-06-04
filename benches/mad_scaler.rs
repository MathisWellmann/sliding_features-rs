use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use rand::{
    RngExt,
    SeedableRng,
    rngs::SmallRng,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::MadScaler,
};

const N: usize = 100_000;
const WINDOW_LENS: &[usize] = &[128, 256, 512, 1024, 2048, 4092, 8192];

fn standard_normal(rng: &mut SmallRng) -> f64 {
    // Box-Muller transform. Clamp u1 away from zero so ln(u1) is finite.
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn realistic_market_prices_f64() -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut price: f64 = 100.0;
    let mut volatility: f64 = 0.012;

    (0..N)
        .map(|i| {
            // Slowly varying volatility with clustering after large shocks.
            volatility = 0.985 * volatility + 0.015 * 0.012;
            let shock = standard_normal(&mut rng);
            if shock.abs() > 2.5 {
                volatility = (volatility * 1.35).min(0.08);
            }

            // Occasional gap/jump events, as seen in real market data.
            let jump = if rng.random::<f64>() < 0.002 {
                standard_normal(&mut rng) * 0.06
            } else {
                0.0
            };

            // Mild trend plus a daily-ish seasonality component.
            let drift = 0.00002;
            let seasonality = 0.0015 * ((i as f64 / 390.0) * std::f64::consts::TAU).sin();
            let log_return = drift + seasonality + volatility * shock + jump;

            price *= log_return.exp();
            price
        })
        .collect()
}

fn realistic_market_prices_f32() -> Vec<f32> {
    realistic_market_prices_f64()
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mad_scaler_100k");
    group.throughput(Throughput::Elements(N as u64));

    for &window_len in WINDOW_LENS {
        group.bench_function(
            BenchmarkId::new("realistic_market_prices_f64", window_len),
            |b| {
                let vals = realistic_market_prices_f64();
                b.iter(|| {
                    let mut view = MadScaler::<f64, _>::new(
                        Echo::new(),
                        NonZeroUsize::new(window_len).unwrap(),
                    );
                    for v in vals.iter() {
                        view.update(*v);
                        let _ = black_box(view.last());
                    }
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("realistic_market_prices_f32", window_len),
            |b| {
                let vals = realistic_market_prices_f32();
                b.iter(|| {
                    let mut view = MadScaler::<f32, _>::new(
                        Echo::new(),
                        NonZeroUsize::new(window_len).unwrap(),
                    );
                    for v in vals.iter() {
                        view.update(*v);
                        let _ = black_box(view.last());
                    }
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

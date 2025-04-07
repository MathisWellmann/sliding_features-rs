use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rng, rngs::SmallRng};
use sliding_features::{
    View,
    pure_functions::Echo,
    sliding_windows::{EhlersFisherTransform, Ema},
};
use time_series_generator::generate_standard_normal;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rng();
    const N: usize = 100_000;

    let mut group = c.benchmark_group("ehlers_fisher_transform_100k");
    group.bench_function("f64", |b| {
        let vals = Vec::<f64>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = EhlersFisherTransform::<f64, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("f32", |b| {
        let vals = Vec::<f32>::from_iter((0..N).map(|_| rng.random()));
        b.iter(|| {
            let mut view = EhlersFisherTransform::<f32, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in vals.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });

    let mut rng = SmallRng::seed_from_u64(0);
    group.bench_function("brownian_motion_f64", |b| {
        let motion = generate_standard_normal(&mut rng, N, 1000.0);
        b.iter(|| {
            let mut view = EhlersFisherTransform::<f64, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in motion.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
    group.bench_function("brownian_motion_f32", |b| {
        let motion = generate_standard_normal(&mut rng, N, 1000.0);
        b.iter(|| {
            let mut view = EhlersFisherTransform::<f32, _, _>::new(
                Echo::new(),
                Ema::new(Echo::new(), 1024),
                1024,
            );
            for v in motion.iter() {
                view.update(*v);
                let _ = black_box(view.last());
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

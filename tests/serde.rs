// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Views saved and restored with serde continue exactly where they stopped.

#![cfg(feature = "serde")]

use std::num::NonZeroUsize;

use serde::{
    Serialize,
    de::DeserializeOwned,
};
use sliding_features::{
    View,
    pure_functions::Echo,
    rolling::Bocpd,
    sliding_windows::{
        Ema,
        HilbertTrendMode,
        RollingMedian,
        Sma,
    },
};

/// A random walk, so that the values have no short decimal form.
fn random_walk(seed: u64, len: usize) -> Vec<f64> {
    let rng = romu::Rng::from_seed_with_64bit(seed);
    let mut price = 100.0;
    Vec::from_iter((0..len).map(|_| {
        price += rng.f64() - 0.5;
        price
    }))
}

/// Save `view` with bincode, which stores every `f64` exactly, and restore it.
fn round_trip<V: Serialize + DeserializeOwned>(view: &V) -> V {
    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(view, config).expect("serializable");
    let (restored, read) =
        bincode::serde::decode_from_slice(&bytes, config).expect("deserializable");
    assert_eq!(read, bytes.len(), "the whole state is read");
    restored
}

/// Feed `values` to a view that is saved and restored every `step` values,
/// and to one that runs without a break: they give the same bits.
fn assert_restored_views_continue<V>(make: impl Fn() -> V, values: &[f64])
where
    V: View<f64> + Serialize + DeserializeOwned,
{
    for step in [1, 7, 61] {
        let mut whole = make();
        let mut restored = make();
        for (i, v) in values.iter().enumerate() {
            if i % step == 0 {
                restored = round_trip(&restored);
            }
            whole.update(*v);
            restored.update(*v);
            assert_eq!(
                whole.last().map(f64::to_bits),
                restored.last().map(f64::to_bits),
                "every {step}, value {i}"
            );
        }
        assert!(whole.last().is_some(), "the test needs a ready view");
    }
}

fn window(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("positive")
}

#[test]
fn a_chain_of_views_continues_after_a_restore() {
    let values = random_walk(1, 500);
    assert_restored_views_continue(
        || Sma::new(Ema::new(Echo::new(), window(10)), window(20)),
        &values,
    );
    assert_restored_views_continue(|| RollingMedian::new(Echo::new(), window(31)), &values);
}

#[test]
fn views_with_long_arrays_continue_after_a_restore() {
    let values = random_walk(2, 500);
    assert_restored_views_continue(|| HilbertTrendMode::new(Echo::new()), &values);
}

#[test]
fn a_change_point_detector_continues_after_a_restore() {
    let returns = Vec::from_iter(random_walk(3, 400).windows(2).map(|w| w[1] - w[0]));
    assert_restored_views_continue(
        || Bocpd::new(Echo::new(), 100.0, 0.0, 1.0, 1.0, 1.0, 200),
        &returns,
    );
}

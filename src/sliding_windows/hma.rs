// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! HMA - Hull Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Wma,
};

/// HMA - Hull Moving Average (Alan Hull, 2005): `WMA(2*WMA(n/2) - WMA(n),
/// round(sqrt(n)))`.
///
/// `n/2` and `sqrt(n)` are rounded to the nearest integer (round half away
/// from zero, `num::Float::round`); some implementations instead floor the
/// half length. `n >= 2` is required so the half length is at least 1.
///
/// Reduces lag relative to a plain WMA of the same length while staying
/// smoother than the raw input, at the cost of overshoot on reversals.
///
/// `last()` is `None` until the final WMA stage is ready, which needs about
/// `max(n, round(n/2)) + round(sqrt(n)) - 1` input values; output is
/// unbounded, following the input.
///
/// Complexity: O(1) per update (three WMA updates, each O(1) amortized);
/// O(n) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Hma<T, V> {
    view: V,
    /// The non-zero window length `n`.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The rounded half window length, `round(n / 2)`.
    #[getset(get_copy = "pub")]
    half_len: NonZeroUsize,
    /// The rounded square-root window length, `round(sqrt(n))`.
    #[getset(get_copy = "pub")]
    sqrt_len: NonZeroUsize,
    wma_half: Wma<T, Echo<T>>,
    wma_full: Wma<T, Echo<T>>,
    wma_final: Wma<T, Echo<T>>,
}

impl<T, V> Hma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new HMA with a chained `View` and a window length.
    ///
    /// # Panics
    /// If `window_len` is 1: the half-length would be below 1.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 2, "Hull MA needs a window length >= 2");

        let n = T::from(window_len.get()).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let half_len = round_to_window_len(n / two);
        let sqrt_len = round_to_window_len(n.sqrt());

        Self {
            view,
            window_len,
            half_len,
            sqrt_len,
            wma_half: Wma::new(Echo::new(), half_len),
            wma_full: Wma::new(Echo::new(), window_len),
            wma_final: Wma::new(Echo::new(), sqrt_len),
        }
    }
}

/// Rounds `x` (expected to be a small positive value) to the nearest
/// non-zero window length.
fn round_to_window_len<T: Float>(x: T) -> NonZeroUsize {
    let rounded = x.round();
    debug_assert!(rounded >= T::one(), "a window length rounds to at least 1");
    let len = rounded
        .to_usize()
        .expect("rounded window lengths used here are small positive integers");
    NonZeroUsize::new(len).expect("rounds to at least 1")
}

impl<T, V> View<T> for Hma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.wma_half.update(val);
        self.wma_full.update(val);
        let (Some(half), Some(full)) = (self.wma_half.last(), self.wma_full.last()) else {
            return;
        };
        let two = T::from(2.0).expect("can convert");
        let diff = two * half - full;
        debug_assert!(diff.is_finite(), "value must be finite");
        self.wma_final.update(diff);
    }

    fn last(&self) -> Option<T> {
        self.wma_final.last().inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    #[test]
    fn hma_warms_up() {
        let mut hma = Hma::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        assert_eq!(hma.last(), None);
        for v in &TEST_DATA {
            hma.update(*v);
        }
        assert!(hma.last().is_some());
    }

    #[test]
    fn hma_constant_input_is_the_constant() {
        let mut hma = Hma::new(Echo::new(), NonZeroUsize::new(9).unwrap());
        for _ in 0..30 {
            hma.update(6.0);
        }
        assert!((hma.last().expect("ready") - 6.0).abs() < 1e-9);
    }

    #[test]
    fn hma_derives_half_and_sqrt_lengths() {
        let hma = Hma::<f64, _>::new(Echo::new(), NonZeroUsize::new(9).unwrap());
        assert_eq!(hma.half_len(), NonZeroUsize::new(5).unwrap()); // round(9/2) = round(4.5) = 5
        assert_eq!(hma.sqrt_len(), NonZeroUsize::new(3).unwrap()); // round(sqrt(9)) = 3
    }

    /// Naive reference: independent WMAs over a manual sliding window,
    /// combined as `wma(2*wma_half - wma_full, sqrt_len)`.
    #[test]
    fn hma_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(66);
        let window_len = 16usize;
        let half_len = 8usize;
        let sqrt_len = 4usize;
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 50.0).collect();

        let naive_wma = |window: &[f64]| -> f64 {
            let denom: f64 = (1..=window.len()).sum::<usize>() as f64;
            window
                .iter()
                .zip(1..)
                .map(|(v, i)| v * i as f64)
                .sum::<f64>()
                / denom
        };

        let mut hma = Hma::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let mut diffs: Vec<f64> = Vec::new();
        for (i, v) in vals.iter().enumerate() {
            hma.update(*v);
            if i + 1 < window_len {
                assert_eq!(hma.last(), None);
                continue;
            }
            let half = naive_wma(&vals[i + 1 - half_len..=i]);
            let full = naive_wma(&vals[i + 1 - window_len..=i]);
            diffs.push(2.0 * half - full);
            if diffs.len() < sqrt_len {
                assert_eq!(hma.last(), None);
                continue;
            }
            let expected = naive_wma(&diffs[diffs.len() - sqrt_len..]);
            let got = hma.last().expect("ready");
            assert!(
                (got - expected).abs() < 1e-9,
                "step {i}: {got} vs {expected}"
            );
        }
    }

    /// HMA follows a ramp with less lag than a plain EMA of the same length.
    #[test]
    fn hma_lags_less_than_ema_on_a_ramp() {
        let window_len = NonZeroUsize::new(16).unwrap();
        let mut hma = Hma::new(Echo::new(), window_len);
        let mut ema = Ema::new(Echo::new(), window_len);
        let ramp: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for v in &ramp {
            hma.update(*v);
            ema.update(*v);
        }
        let i = ramp.len() - 1;
        let hma_lag = (i as f64 - hma.last().expect("ready")).abs();
        let ema_lag = i as f64 - ema.last().expect("ready");
        assert!(hma_lag < ema_lag, "hma {hma_lag} ema {ema_lag}");
    }

    #[test]
    #[should_panic(expected = "window length >= 2")]
    fn window_len_of_one_is_rejected() {
        let _ = Hma::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn hma_plot() {
        let mut hma = Hma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            hma.update(*v);
            if let Some(val) = hma.last() {
                out.push(val);
            }
        }
        let filename = "img/hma.png";
        plot_values(out, filename).unwrap();
    }
}

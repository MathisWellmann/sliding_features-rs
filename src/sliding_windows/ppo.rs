// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PPO - Percentage Price Oscillator

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::Ema;
use crate::{
    View,
    pure_functions::Echo,
};

/// PPO - Percentage Price Oscillator.
///
/// `PPO = 100 * (EMA(fast_period) - EMA(slow_period)) / EMA(slow_period)`.
/// The `signal` line is an EMA of the PPO line over `signal_period`, and the
/// `histogram` is `PPO - signal`.
///
/// Deviation from TA-Lib: TA-Lib's `PPO` defaults to SMA for the two moving
/// averages (`optInMAType = 0`); this implementation always uses EMA (as do
/// most charting platforms, e.g. the "Percentage Price Oscillator" on
/// StockCharts/TradingView), matching the classic MACD-derived definition.
///
/// `last()` is `None` until the slow EMA is ready (`slow_period` values
/// observed) and non-zero; `signal()` additionally needs `signal_period`
/// more PPO values.
///
/// Output range: unbounded (expressed in percent).
///
/// Complexity: O(1) per update (three chained EMAs), O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Ppo<T, V> {
    view: V,
    /// The fast EMA period.
    #[getset(get_copy = "pub")]
    fast_period: NonZeroUsize,
    /// The slow EMA period.
    #[getset(get_copy = "pub")]
    slow_period: NonZeroUsize,
    /// The signal line EMA period.
    #[getset(get_copy = "pub")]
    signal_period: NonZeroUsize,
    ema_fast: Ema<T, Echo<T>>,
    ema_slow: Ema<T, Echo<T>>,
    ema_signal: Ema<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Ppo<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new PPO with a chained `View` and custom fast, slow and
    /// signal periods.
    ///
    /// # Panics
    /// If `fast_period >= slow_period`.
    pub fn with_periods(
        view: V,
        fast_period: NonZeroUsize,
        slow_period: NonZeroUsize,
        signal_period: NonZeroUsize,
    ) -> Self {
        assert!(
            fast_period.get() < slow_period.get(),
            "the fast period must be shorter than the slow period"
        );
        Self {
            view,
            fast_period,
            slow_period,
            signal_period,
            ema_fast: Ema::new(Echo::new(), fast_period),
            ema_slow: Ema::new(Echo::new(), slow_period),
            ema_signal: Ema::new(Echo::new(), signal_period),
            out: None,
        }
    }

    /// Create a new PPO with a chained `View` and the classic default
    /// periods: 12 (fast), 26 (slow) and 9 (signal).
    pub fn new(view: V) -> Self {
        Self::with_periods(
            view,
            NonZeroUsize::new(12).expect("12 is non-zero"),
            NonZeroUsize::new(26).expect("26 is non-zero"),
            NonZeroUsize::new(9).expect("9 is non-zero"),
        )
    }

    /// The signal line: an EMA of the PPO line over `signal_period`.
    /// `None` until the signal EMA is ready.
    #[inline]
    pub fn signal(&self) -> Option<T> {
        self.ema_signal.last()
    }

    /// The PPO histogram: `PPO - signal`. `None` until `signal()` is `Some`.
    #[inline]
    pub fn histogram(&self) -> Option<T> {
        Some(self.out? - self.signal()?)
    }
}

impl<T, V> View<T> for Ppo<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.ema_fast.update(val);
        self.ema_slow.update(val);
        let (Some(fast), Some(slow)) = (self.ema_fast.last(), self.ema_slow.last()) else {
            return;
        };
        // A zero slow EMA (e.g. all-zero input) makes the percentage
        // undefined, so the PPO (and its signal) simply stay `None`.
        if slow == T::zero() {
            return;
        }
        let ppo = T::from(100.0).expect("can convert") * (fast - slow) / slow;
        debug_assert!(ppo.is_finite(), "`ppo` must be finite");
        self.ema_signal.update(ppo);
        self.out = Some(ppo);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    /// Naive reference: plain EMA recurrence, alpha = 2 / (n + 1).
    struct NaiveEma {
        period: usize,
        val: Option<f64>,
    }
    impl NaiveEma {
        fn new(period: usize) -> Self {
            Self { period, val: None }
        }
        fn update(&mut self, x: f64) -> f64 {
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let out = match self.val {
                None => x,
                Some(prev) => alpha * x + (1.0 - alpha) * prev,
            };
            self.val = Some(out);
            out
        }
    }

    #[test]
    fn ppo_plot() {
        let mut ppo = Ppo::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            ppo.update(*v);
            if let Some(val) = ppo.last() {
                out.push(val);
            }
        }
        let filename = "img/ppo.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn ppo_warms_up() {
        let mut ppo = Ppo::with_periods(
            Echo::new(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter().enumerate() {
            ppo.update(v);
            if i < 3 {
                assert_eq!(ppo.last(), None, "step {i}");
            } else {
                assert!(ppo.last().is_some());
            }
        }
        assert!(ppo.signal().is_some());
    }

    #[test]
    fn ppo_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        let (fast, slow, signal) = (5, 13, 4);
        let mut ppo = Ppo::with_periods(
            Echo::new(),
            NonZeroUsize::new(fast).unwrap(),
            NonZeroUsize::new(slow).unwrap(),
            NonZeroUsize::new(signal).unwrap(),
        );
        let mut naive_fast = NaiveEma::new(fast);
        let mut naive_slow = NaiveEma::new(slow);
        let mut naive_signal = NaiveEma::new(signal);
        for _ in 0..2_000 {
            // Keep values away from zero so the slow EMA never hits zero.
            let x = rng.f64() * 100.0 + 50.0;
            ppo.update(x);
            let (f, s) = (naive_fast.update(x), naive_slow.update(x));
            let naive_ppo = 100.0 * (f - s) / s;
            if let Some(got) = ppo.last() {
                assert!((got - naive_ppo).abs() < 1e-9);
                let naive_sig = naive_signal.update(naive_ppo);
                if let Some(got_sig) = ppo.signal() {
                    assert!((got_sig - naive_sig).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn ppo_constant_input_converges_to_zero() {
        let mut ppo = Ppo::new(Echo::new());
        for _ in 0..500 {
            ppo.update(42.0);
        }
        assert!(ppo.last().unwrap().abs() < 1e-6);
        assert!(ppo.histogram().unwrap().abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "fast period must be shorter")]
    fn ppo_fast_not_shorter_than_slow_panics() {
        let _ = Ppo::with_periods(
            Echo::<f64>::new(),
            NonZeroUsize::new(26).unwrap(),
            NonZeroUsize::new(26).unwrap(),
            NonZeroUsize::new(9).unwrap(),
        );
    }
}

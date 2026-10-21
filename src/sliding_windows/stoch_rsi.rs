// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! StochRSI - Stochastic RSI

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Max,
    Min,
    Rsi,
    Sma,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// StochRSI - Stochastic RSI (Chande & Kroll, 1994; TA-Lib `STOCHRSI` fastk
/// line).
///
/// Applies the stochastic oscillator formula to the RSI instead of price:
/// `%K = (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n))`, where `n` is
/// `stoch_period`. `%D` is an SMA of `%K` over `d_period`.
///
/// Deviation: TA-Lib's `STOCHRSI` multiplies by 100 (range `[0, 100]`); this
/// implementation keeps the raw `[0, 1]` ratio, documented here.
///
/// Flat-window behavior: if `max(RSI) == min(RSI)` over the window (e.g. a
/// constant RSI), `%K` is defined as `0` rather than `NaN`.
///
/// `last()` is `%K`, `None` until `rsi_period + stoch_period - 1` values were
/// observed; `d()` additionally needs `d_period` more `%K` values.
///
/// Output range: `[0, 1]`.
///
/// Complexity: O(1) amortized per update (RSI, the monotonic min/max deques
/// and the SMA are all O(1) amortized), O(stoch_period + d_period) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct StochRsi<T, V> {
    view: V,
    /// The RSI period.
    #[getset(get_copy = "pub")]
    rsi_period: NonZeroUsize,
    /// The stochastic window length applied to the RSI.
    #[getset(get_copy = "pub")]
    stoch_period: NonZeroUsize,
    /// The `%D` (SMA of `%K`) smoothing period.
    #[getset(get_copy = "pub")]
    d_period: NonZeroUsize,
    rsi: Rsi<T, Echo<T>>,
    max: Max<T, Echo<T>>,
    min: Min<T, Echo<T>>,
    sma_d: Sma<T, Echo<T>>,
    n_rsi_observed: usize,
    out: Option<T>,
}

impl<T, V> StochRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new StochRSI with a chained `View`, an RSI period, a
    /// stochastic window length and a `%D` smoothing period.
    pub fn new(
        view: V,
        rsi_period: NonZeroUsize,
        stoch_period: NonZeroUsize,
        d_period: NonZeroUsize,
    ) -> Self {
        Self {
            view,
            rsi_period,
            stoch_period,
            d_period,
            rsi: Rsi::new(Echo::new(), rsi_period),
            max: Max::new(Echo::new(), stoch_period),
            min: Min::new(Echo::new(), stoch_period),
            sma_d: Sma::new(Echo::new(), d_period),
            n_rsi_observed: 0,
            out: None,
        }
    }

    /// `%D`: an SMA of `%K` over `d_period`. `None` until it is ready.
    #[inline]
    pub fn d(&self) -> Option<T> {
        self.sma_d.last()
    }
}

impl<T, V> View<T> for StochRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.rsi.update(val);
        let Some(rsi) = self.rsi.last() else { return };
        self.max.update(rsi);
        self.min.update(rsi);
        self.n_rsi_observed += 1;
        if self.n_rsi_observed < self.stoch_period.get() {
            return;
        }
        let hi = self.max.last().expect("at least one value was observed");
        let lo = self.min.last().expect("at least one value was observed");
        debug_assert!(hi >= lo, "the max is never below the min");

        let fastk = if hi == lo {
            T::zero()
        } else {
            (rsi - lo) / (hi - lo)
        };
        debug_assert!(fastk.is_finite(), "`fastk` must be finite");
        self.sma_d.update(fastk);
        self.out = Some(fastk);
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

    #[test]
    fn stoch_rsi_plot() {
        let mut v = StochRsi::new(
            Echo::new(),
            NonZeroUsize::new(14).unwrap(),
            NonZeroUsize::new(14).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        let mut out: Vec<f64> = Vec::new();
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        let filename = "img/stoch_rsi.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn stoch_rsi_warms_up() {
        let mut v = StochRsi::new(
            Echo::new(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        let mut seen_k = false;
        let mut seen_d = false;
        for x in [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0] {
            v.update(x);
            if v.last().is_some() {
                seen_k = true;
            }
            if v.d().is_some() {
                seen_d = true;
            }
        }
        assert!(seen_k);
        assert!(seen_d);
    }

    #[test]
    fn stoch_rsi_range() {
        let mut v = StochRsi::new(
            Echo::new(),
            NonZeroUsize::new(14).unwrap(),
            NonZeroUsize::new(14).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        for x in &TEST_DATA {
            v.update(*x);
            // A tiny epsilon absorbs floating point round-off from the
            // running sum in `Sma`, which can push an exact 0 or 1 input
            // a few ULPs outside of [0, 1].
            const EPS: f64 = 1e-12;
            if let Some(k) = v.last() {
                assert!((-EPS..=1.0 + EPS).contains(&k), "k = {k}");
            }
            if let Some(d) = v.d() {
                assert!((-EPS..=1.0 + EPS).contains(&d), "d = {d}");
            }
        }
    }

    #[test]
    fn stoch_rsi_constant_input_is_zero() {
        let mut v = StochRsi::new(
            Echo::new(),
            NonZeroUsize::new(5).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        for _ in 0..50 {
            v.update(10.0);
        }
        // RSI is flat (no gains or losses), so max == min and %K is 0.
        assert_eq!(v.last(), Some(0.0));
    }

    /// Naive reference: rescans the trailing RSI window for min/max.
    #[test]
    fn stoch_rsi_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(29);
        let rsi_period = 6;
        let stoch_period = 5;
        let mut v = StochRsi::new(
            Echo::new(),
            NonZeroUsize::new(rsi_period).unwrap(),
            NonZeroUsize::new(stoch_period).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        let mut naive_rsi = Rsi::new(Echo::new(), NonZeroUsize::new(rsi_period).unwrap());
        let mut rsi_window: Vec<f64> = Vec::new();
        for _ in 0..2_000 {
            let x = rng.f64() * 100.0;
            v.update(x);
            naive_rsi.update(x);
            if let Some(rsi) = naive_rsi.last() {
                rsi_window.push(rsi);
                if rsi_window.len() > stoch_period {
                    rsi_window.remove(0);
                }
            }
            if let Some(got) = v.last() {
                let hi = rsi_window.iter().copied().fold(f64::MIN, f64::max);
                let lo = rsi_window.iter().copied().fold(f64::MAX, f64::min);
                let expected = if hi == lo {
                    0.0
                } else {
                    (*rsi_window.last().unwrap() - lo) / (hi - lo)
                };
                assert!((got - expected).abs() < 1e-9);
            }
        }
    }
}

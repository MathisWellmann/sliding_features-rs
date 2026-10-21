// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sharpe ratio over a sliding window of returns.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Sharpe ratio of the last `window_len` returns: the mean excess return per unit of
/// standard deviation.
///
/// Reference: W. F. Sharpe (1966), "Mutual Fund Performance", Journal of Business 39;
/// ex-post form of W. F. Sharpe (1994), "The Sharpe Ratio".
///
/// `S = (mean(r) - r_f) / std(r)`, with the sample standard deviation (`n - 1` in the
/// denominator) of the returns in the window; `std(r - r_f) = std(r)` for a constant `r_f`.
///
/// Input: returns per period (e.g. simple or log returns), not prices.
/// Parameter `risk_free`: the risk-free rate per period (same period as the returns).
/// Output: the per-period Sharpe ratio; it is not annualized (multiply by
/// `sqrt(periods per year)` for i.i.d. returns). `None` while the standard deviation is
/// zero, i.e. when the sample variance is at most `4 eps` times the mean square deviation
/// from the internal shift, below which it is rounding noise (e.g. constant returns).
///
/// Warm-up: `None` until `window_len` returns were observed.
/// Complexity: O(1) per update with running sums of the deviations from a shift value
/// (a value of the window, which avoids catastrophic cancellation). The sums are
/// recomputed from the window every `window_len` updates to bound floating point drift
/// (amortized O(1)). O(n) memory, preallocated.
#[derive(Debug, Clone, CopyGetters)]
pub struct SharpeRatio<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    risk_free: T,
    window: VecDeque<T>,
    shift: T,
    /// Sum of `r - shift` over the window.
    sum: T,
    /// Sum of `(r - shift)^2` over the window.
    sum_sq: T,
    evictions: usize,
}

impl<T, V> SharpeRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Sharpe ratio with a chained `View` of returns, the window length and
    /// the risk-free rate per period.
    ///
    /// # Panics
    /// If `window_len < 2` (no sample standard deviation) or `risk_free` is not finite.
    pub fn new(view: V, window_len: NonZeroUsize, risk_free: T) -> Self {
        assert!(window_len.get() >= 2, "window_len must be >= 2");
        assert!(risk_free.is_finite(), "risk_free must be finite");
        Self {
            view,
            window_len,
            risk_free,
            window: VecDeque::with_capacity(window_len.get() + 1),
            shift: T::zero(),
            sum: T::zero(),
            sum_sq: T::zero(),
            evictions: 0,
        }
    }

    /// The risk-free rate per period.
    #[inline]
    pub fn risk_free(&self) -> T {
        self.risk_free
    }

    /// Re-center the sums on the newest value and recompute them from the window.
    fn recompute(&mut self) {
        self.shift = *self.window.back().expect("window is not empty");
        let (sum, sum_sq) = shifted_sums(self.window.iter().copied(), self.shift);
        self.sum = sum;
        self.sum_sq = sum_sq;
    }
}

/// Sum and sum of squares of `vals - shift`.
fn shifted_sums<T: Float>(vals: impl Iterator<Item = T>, shift: T) -> (T, T) {
    vals.fold((T::zero(), T::zero()), |(s, sq), v| {
        let d = v - shift;
        (s + d, sq + d * d)
    })
}

impl<T, V> View<T> for SharpeRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.window.is_empty() {
            self.shift = val;
        }
        self.window.push_back(val);
        let d = val - self.shift;
        self.sum = self.sum + d;
        self.sum_sq = self.sum_sq + d * d;
        if self.window.len() > self.window_len.get() {
            let old = self.window.pop_front().expect("window is not empty") - self.shift;
            self.sum = self.sum - old;
            self.sum_sq = self.sum_sq - old * old;
            self.evictions += 1;
            if self.evictions == self.window_len.get() {
                self.evictions = 0;
                self.recompute();
            }
        }
        debug_assert!(
            self.window.len() <= self.window_len.get(),
            "window is bounded"
        );
    }

    fn last(&self) -> Option<T> {
        if self.window.len() < self.window_len.get() {
            return None;
        }
        let n = T::from(self.window.len()).expect("can convert");
        let dof = n - T::one();
        let var = (self.sum_sq - self.sum * self.sum / n) / dof;
        let four = T::from(4).expect("can convert");
        if var <= four * T::epsilon() * self.sum_sq.max(T::zero()) / dof {
            return None;
        }
        let mean = self.shift + self.sum / n;
        let out = (mean - self.risk_free) / var.sqrt();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        rolling::LnReturn,
        test_data::TEST_DATA,
    };

    fn sharpe(window_len: usize, risk_free: f64) -> SharpeRatio<f64, Echo<f64>> {
        SharpeRatio::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            risk_free,
        )
    }

    /// Two-pass Sharpe ratio with the sample standard deviation.
    fn naive_sharpe(w: &[f64], risk_free: f64) -> f64 {
        let n = w.len() as f64;
        let mean = w.iter().sum::<f64>() / n;
        let var = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean - risk_free) / var.sqrt()
    }

    #[test]
    fn sharpe_ratio_plot() {
        let mut v = SharpeRatio::new(
            LnReturn::new(Echo::new()),
            NonZeroUsize::new(32).unwrap(),
            0.0,
        );
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/sharpe_ratio.png").unwrap();
    }

    #[test]
    fn sharpe_ratio_warmup_and_hand_computed() {
        let mut v = sharpe(3, 0.01);
        assert_eq!(v.window_len().get(), 3);
        assert_eq!(v.risk_free(), 0.01);
        v.update(0.01);
        v.update(0.03);
        assert_eq!(v.last(), None);
        v.update(0.05);
        // mean 0.03, sample std 0.02 -> (0.03 - 0.01) / 0.02 = 1
        assert!((v.last().unwrap() - 1.0).abs() < 1e-12);
        v.update(-0.03);
        // [0.03, 0.05, -0.03]: mean 0.01667, std 0.04163
        assert!((v.last().unwrap() - naive_sharpe(&[0.03, 0.05, -0.03], 0.01)).abs() < 1e-12);
    }

    #[test]
    fn sharpe_ratio_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(151);
        for (window_len, risk_free) in [(2, 0.0), (3, 0.001), (20, 0.0), (250, -0.002)] {
            // A drifting level, so the shift value matters.
            let returns: Vec<f64> = (0..20_000)
                .map(|i| 0.5 * (i as f64 / 5_000.0).sin() + (rng.f64() - 0.5) * 0.04)
                .collect();
            let mut v = sharpe(window_len, risk_free);
            for (i, r) in returns.iter().enumerate() {
                v.update(*r);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let expected = naive_sharpe(&returns[i + 1 - window_len..=i], risk_free);
                let got = v.last().unwrap();
                assert!(
                    (got - expected).abs() <= 1e-8 * expected.abs().max(1.0),
                    "step {i}, n {window_len}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn sharpe_ratio_risk_free_is_a_shift() {
        let rng = romu::Rng::from_seed_with_64bit(152);
        let (mut a, mut b) = (sharpe(30, 0.002), sharpe(30, 0.0));
        for _ in 0..500 {
            let r = (rng.f64() - 0.45) * 0.02;
            a.update(r);
            b.update(r - 0.002);
            if let (Some(x), Some(y)) = (a.last(), b.last()) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn sharpe_ratio_zero_std_is_none() {
        let mut v = sharpe(4, 0.0);
        for _ in 0..10 {
            v.update(0.003);
            assert_eq!(v.last(), None);
        }
        v.update(0.005);
        assert!(v.last().unwrap() > 0.0);
        // Constant again after the different value left the window (and after recomputes).
        for _ in 0..20 {
            v.update(-0.1);
        }
        assert_eq!(v.last(), None);
    }

    #[test]
    #[should_panic(expected = "window_len must be >= 2")]
    fn sharpe_ratio_window_one() {
        let _ = sharpe(1, 0.0);
    }

    #[test]
    #[should_panic(expected = "risk_free must be finite")]
    fn sharpe_ratio_nan_risk_free() {
        let _ = sharpe(10, f64::NAN);
    }
}

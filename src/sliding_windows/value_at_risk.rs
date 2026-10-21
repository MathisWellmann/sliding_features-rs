// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Historical Value at Risk over a sliding window of returns.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Historical (non-parametric) Value at Risk (VaR) of the last `window_len` returns.
///
/// Reference: P. Jorion, "Value at Risk: The New Benchmark for Managing Financial Risk";
/// historical simulation method.
///
/// `VaR_c = -Q(1 - c)`, where `Q` is the empirical quantile of the returns in the window
/// with linear interpolation between order statistics (Hyndman & Fan 1996 type 7, the
/// default of R and numpy).
///
/// Input: returns (e.g. simple or log returns per period), not prices.
/// Parameter `confidence` (`c`) in `(0, 1)`, e.g. `0.95` or `0.99`.
/// Output sign convention: a positive value is a loss, e.g. `0.02` means that with
/// confidence `c` the loss of one period does not exceed 2%. It is negative if even the
/// `1 - c` quantile of the returns is a gain.
///
/// Warm-up: `None` until `window_len` returns were observed.
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a sorted `Vec`
/// (fast for typical window sizes); O(1) for `last()`; O(n) memory.
#[derive(Debug, Clone)]
pub struct ValueAtRisk<T, V> {
    view: V,
    confidence: T,
    window: SortedWindow<T>,
}

impl<T, V> ValueAtRisk<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new historical VaR with a chained `View` of returns, a sliding window
    /// length and the confidence level.
    ///
    /// # Panics
    /// If `confidence` is not in `(0, 1)`.
    pub fn new(view: V, window_len: NonZeroUsize, confidence: T) -> Self {
        assert!(confidence > T::zero(), "confidence must be > 0");
        assert!(confidence < T::one(), "confidence must be < 1");
        Self {
            view,
            confidence,
            window: SortedWindow::new(window_len),
        }
    }

    /// The confidence level in `(0, 1)`.
    #[inline]
    pub fn confidence(&self) -> T {
        self.confidence
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }
}

impl<T, V> View<T> for ValueAtRisk<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let _evicted = self.window.push(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.window.is_full() {
            return None;
        }
        let q = self
            .window
            .quantile(T::one() - self.confidence)
            .expect("a full window is not empty");
        let out = -q;
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

    fn var(window_len: usize, confidence: f64) -> ValueAtRisk<f64, Echo<f64>> {
        ValueAtRisk::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            confidence,
        )
    }

    /// Naive type 7 quantile: sort a copy of the window.
    fn naive_quantile(window: &[f64], q: f64) -> f64 {
        let mut s = window.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = (s.len() - 1) as f64 * q;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(s.len() - 1);
        s[lo] + (h - lo as f64) * (s[hi] - s[lo])
    }

    #[test]
    fn value_at_risk_plot() {
        let mut v = ValueAtRisk::new(
            LnReturn::new(Echo::new()),
            NonZeroUsize::new(64).unwrap(),
            0.95,
        );
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/value_at_risk.png").unwrap();
    }

    #[test]
    fn value_at_risk_warmup_and_hand_computed() {
        let mut v = var(5, 0.75);
        assert_eq!(v.window_len().get(), 5);
        assert_eq!(v.confidence(), 0.75);
        for r in [0.01, -0.02, 0.03, -0.04] {
            v.update(r);
            assert_eq!(v.last(), None);
        }
        v.update(0.0);
        // sorted [-0.04, -0.02, 0, 0.01, 0.03], h = 4 * 0.25 = 1 -> Q = -0.02
        assert_eq!(v.last(), Some(0.02));
        v.update(0.05);
        // sorted [-0.04, -0.02, 0, 0.03, 0.05] -> Q = -0.02
        assert_eq!(v.last(), Some(0.02));
        v.update(0.06);
        // sorted [-0.04, 0, 0.03, 0.05, 0.06] -> Q = 0
        assert_eq!(v.last(), Some(-0.0));
    }

    #[test]
    fn value_at_risk_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(121);
        for window_len in [1, 2, 20, 250] {
            for confidence in [0.5, 0.9, 0.95, 0.99] {
                let returns: Vec<f64> = (0..1_500).map(|_| (rng.f64() - 0.5) * 0.1).collect();
                let mut v = var(window_len, confidence);
                for (i, r) in returns.iter().enumerate() {
                    v.update(*r);
                    if i + 1 < window_len {
                        assert_eq!(v.last(), None);
                        continue;
                    }
                    let w = &returns[i + 1 - window_len..=i];
                    let expected = -naive_quantile(w, 1.0 - confidence);
                    assert_eq!(v.last(), Some(expected), "step {i}");
                    // VaR is between the negated best and worst return of the window.
                    let worst = w.iter().copied().fold(f64::MAX, f64::min);
                    assert!(v.last().unwrap() <= -worst);
                }
            }
        }
    }

    #[test]
    fn value_at_risk_higher_confidence_is_larger() {
        let rng = romu::Rng::from_seed_with_64bit(122);
        let (mut lo, mut hi) = (var(100, 0.9), var(100, 0.99));
        for _ in 0..1_000 {
            let r = (rng.f64() - 0.5) * 0.1;
            lo.update(r);
            hi.update(r);
            if let (Some(a), Some(b)) = (lo.last(), hi.last()) {
                assert!(b >= a);
            }
        }
    }

    #[test]
    fn value_at_risk_constant_returns() {
        let mut v = var(10, 0.95);
        for _ in 0..20 {
            v.update(-0.01);
        }
        assert_eq!(v.last(), Some(0.01));
    }

    #[test]
    #[should_panic(expected = "confidence must be > 0")]
    fn value_at_risk_zero_confidence() {
        let _ = var(10, 0.0);
    }

    #[test]
    #[should_panic(expected = "confidence must be < 1")]
    fn value_at_risk_confidence_one() {
        let _ = var(10, 1.0);
    }

    #[test]
    #[should_panic(expected = "confidence must be > 0")]
    fn value_at_risk_nan_confidence() {
        let _ = var(10, f64::NAN);
    }
}

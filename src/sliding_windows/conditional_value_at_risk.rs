//! Historical Conditional Value at Risk (expected shortfall) over a sliding window of returns.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Historical Conditional Value at Risk (CVaR), also called expected shortfall, of the
/// last `window_len` returns.
///
/// Reference: R. T. Rockafellar, S. Uryasev (2002), "Conditional value-at-risk for general
/// loss distributions", Journal of Banking & Finance 26; historical simulation method.
///
/// `CVaR_c = -mean(r_i | r_i <= Q(1 - c))`, where `Q` is the empirical type 7 quantile of
/// the window (Hyndman & Fan 1996, as in [`ValueAtRisk`](super::ValueAtRisk)).
/// Tail-inclusion convention: the tail holds all returns at or below the quantile
/// (`<=`, ties included). It is never empty, as `Q(1 - c)` is at least the minimum.
/// As the mean of values `<= Q` is `<= Q`, `CVaR_c >= VaR_c` always holds.
///
/// Input: returns (e.g. simple or log returns per period), not prices.
/// Parameter `confidence` (`c`) in `(0, 1)`, e.g. `0.95` or `0.975`.
/// Output sign convention: a positive value is a loss, the average loss of the worst
/// `1 - c` fraction of periods.
///
/// Warm-up: `None` until `window_len` returns were observed.
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a sorted `Vec`
/// and an O(n (1 - c)) sum over the tail; O(1) for `last()`; O(n) memory.
#[derive(Debug, Clone)]
pub struct ConditionalValueAtRisk<T, V> {
    view: V,
    confidence: T,
    window: SortedWindow<T>,
    out: Option<T>,
}

impl<T, V> ConditionalValueAtRisk<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new historical CVaR with a chained `View` of returns, a sliding window
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
            out: None,
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

/// `-mean` of the `tail_len` smallest values of the ascending slice `sorted`.
fn negative_tail_mean<T: Float>(sorted: &[T], tail_len: usize) -> T {
    assert!(tail_len > 0, "the tail is never empty");
    assert!(tail_len <= sorted.len(), "the tail is part of the window");
    let sum = sorted[..tail_len].iter().fold(T::zero(), |acc, v| acc + *v);
    -sum / T::from(tail_len).expect("can convert")
}

impl<T, V> View<T> for ConditionalValueAtRisk<T, V>
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
        if !self.window.is_full() {
            return;
        }
        let q = self
            .window
            .quantile(T::one() - self.confidence)
            .expect("a full window is not empty");
        let tail_len = self.window.count_less_equal(q);
        let out = negative_tail_mean(self.window.sorted(), tail_len);
        debug_assert!(out.is_finite(), "value must be finite");
        // Rounding of the mean can undercut `-q` by a few ulps when the tail is all ties;
        // the clamp keeps `CVaR >= VaR` exact.
        debug_assert!(
            out >= -q - T::epsilon().sqrt() * q.abs().max(T::one()),
            "CVaR >= VaR up to rounding"
        );
        self.out = Some(out.max(-q));
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        rolling::LnReturn,
        sliding_windows::ValueAtRisk,
        test_data::TEST_DATA,
    };

    fn cvar(window_len: usize, confidence: f64) -> ConditionalValueAtRisk<f64, Echo<f64>> {
        ConditionalValueAtRisk::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            confidence,
        )
    }

    /// Naive CVaR: sort a copy, type 7 quantile, mean of the values <= quantile.
    fn naive_cvar(window: &[f64], confidence: f64) -> f64 {
        let mut s = window.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = (s.len() - 1) as f64 * (1.0 - confidence);
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(s.len() - 1);
        let q = s[lo] + (h - lo as f64) * (s[hi] - s[lo]);
        let tail: Vec<f64> = s.iter().copied().filter(|v| *v <= q).collect();
        (-tail.iter().sum::<f64>() / tail.len() as f64).max(-q)
    }

    #[test]
    fn conditional_value_at_risk_plot() {
        let mut v = ConditionalValueAtRisk::new(
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
        plot_values(out, "img/conditional_value_at_risk.png").unwrap();
    }

    #[test]
    fn conditional_value_at_risk_warmup_and_hand_computed() {
        let mut v = cvar(5, 0.75);
        assert_eq!(v.window_len().get(), 5);
        assert_eq!(v.confidence(), 0.75);
        for r in [0.01, -0.02, 0.03, -0.04] {
            v.update(r);
            assert_eq!(v.last(), None);
        }
        v.update(0.0);
        // sorted [-0.04, -0.02, 0, 0.01, 0.03], Q(0.25) = -0.02, tail [-0.04, -0.02]
        assert!((v.last().unwrap() - 0.03).abs() < 1e-15);
        v.update(0.05);
        v.update(0.06);
        // sorted [-0.04, 0, 0.03, 0.05, 0.06], Q(0.25) = 0, tail [-0.04, 0]
        assert!((v.last().unwrap() - 0.02).abs() < 1e-15);
    }

    #[test]
    fn conditional_value_at_risk_vs_naive_and_var() {
        let rng = romu::Rng::from_seed_with_64bit(131);
        for window_len in [1, 2, 20, 250] {
            for confidence in [0.5, 0.9, 0.95, 0.99] {
                // Continuous returns and frequent ties.
                let returns: Vec<f64> = (0..1_500)
                    .map(|_| match rng.range_usize(0..3) {
                        0 => rng.range_usize(0..5) as f64 * 0.01 - 0.02,
                        _ => (rng.f64() - 0.5) * 0.1,
                    })
                    .collect();
                let mut v = cvar(window_len, confidence);
                let mut var = ValueAtRisk::new(
                    Echo::new(),
                    NonZeroUsize::new(window_len).unwrap(),
                    confidence,
                );
                for (i, r) in returns.iter().enumerate() {
                    v.update(*r);
                    var.update(*r);
                    if i + 1 < window_len {
                        assert_eq!(v.last(), None);
                        continue;
                    }
                    let w = &returns[i + 1 - window_len..=i];
                    let got = v.last().unwrap();
                    assert!((got - naive_cvar(w, confidence)).abs() < 1e-12, "step {i}");
                    assert!(got >= var.last().unwrap(), "CVaR >= VaR at step {i}");
                    let worst = w.iter().copied().fold(f64::MAX, f64::min);
                    assert!(got <= -worst + 1e-15, "CVaR <= worst loss at step {i}");
                }
            }
        }
    }

    #[test]
    fn conditional_value_at_risk_constant_returns() {
        let mut v = cvar(10, 0.95);
        for _ in 0..20 {
            v.update(-0.01);
        }
        assert_eq!(v.last(), Some(0.01));
    }

    #[test]
    #[should_panic(expected = "confidence must be > 0")]
    fn conditional_value_at_risk_negative_confidence() {
        let _ = cvar(10, -0.5);
    }

    #[test]
    #[should_panic(expected = "confidence must be < 1")]
    fn conditional_value_at_risk_confidence_one() {
        let _ = cvar(10, 1.0);
    }
}

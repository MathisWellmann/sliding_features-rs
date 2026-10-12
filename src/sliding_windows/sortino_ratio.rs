//! Sortino ratio over a sliding window of returns.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Sortino ratio of the last `window_len` returns: the mean excess return over a target
/// per unit of downside deviation.
///
/// Reference: F. A. Sortino, L. N. Price (1994), "Performance Measurement in a Downside
/// Risk Framework", Journal of Investing 3.
///
/// `S = (mean(r) - target) / DD`, with the downside deviation
/// `DD = sqrt(sum(min(0, r_i - target)^2) / n)` over the whole window of `n` returns
/// (returns above the target count as zero deviation, as in Sortino & Price, instead of
/// averaging only over the returns below the target).
///
/// Input: returns per period (e.g. simple or log returns), not prices.
/// Parameter `target`: the minimum acceptable return per period (e.g. `0` or the
/// risk-free rate).
/// Output: the per-period Sortino ratio, not annualized. `None` while no return of the
/// window is below the target (zero downside deviation; the ratio is undefined, or
/// infinite for a positive excess mean), and while the downside deviation rounds to
/// zero (below about `1e-162`, whose square underflows).
///
/// Warm-up: `None` until `window_len` returns were observed.
/// Complexity: O(1) per update with running sums of `r - target` and of the squared
/// downside deviations, and a count of returns below the target, so the downside sum is
/// exactly zero without downside. The sums are recomputed from the window every
/// `window_len` updates to bound floating point drift (amortized O(1)). O(n) memory,
/// preallocated.
#[derive(Debug, Clone, CopyGetters)]
pub struct SortinoRatio<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    target: T,
    window: VecDeque<T>,
    /// Sum of `r - target` over the window.
    sum_excess: T,
    /// Sum of `min(0, r - target)^2` over the window.
    sum_downside_sq: T,
    /// Number of returns below the target in the window.
    num_downside: usize,
    evictions: usize,
}

impl<T, V> SortinoRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Sortino ratio with a chained `View` of returns, the window length and
    /// the target (minimum acceptable) return per period.
    ///
    /// # Panics
    /// If `target` is not finite.
    pub fn new(view: V, window_len: NonZeroUsize, target: T) -> Self {
        assert!(target.is_finite(), "target must be finite");
        Self {
            view,
            window_len,
            target,
            window: VecDeque::with_capacity(window_len.get() + 1),
            sum_excess: T::zero(),
            sum_downside_sq: T::zero(),
            num_downside: 0,
            evictions: 0,
        }
    }

    /// The target (minimum acceptable) return per period.
    #[inline]
    pub fn target(&self) -> T {
        self.target
    }

    /// The downside deviation of the window, `None` during warm-up.
    pub fn downside_deviation(&self) -> Option<T> {
        if self.window.len() < self.window_len.get() {
            return None;
        }
        let n = T::from(self.window.len()).expect("can convert");
        Some((self.sum_downside_sq.max(T::zero()) / n).sqrt())
    }
}

/// The excess return of `r` over `target` and its squared downside part.
#[inline]
fn excess_and_downside<T: Float>(r: T, target: T) -> (T, T) {
    let excess = r - target;
    let downside = excess.min(T::zero());
    (excess, downside * downside)
}

impl<T, V> View<T> for SortinoRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.window.push_back(val);
        let (excess, down_sq) = excess_and_downside(val, self.target);
        self.sum_excess = self.sum_excess + excess;
        self.sum_downside_sq = self.sum_downside_sq + down_sq;
        self.num_downside += usize::from(val < self.target);
        if self.window.len() > self.window_len.get() {
            let old = self.window.pop_front().expect("window is not empty");
            let (excess, down_sq) = excess_and_downside(old, self.target);
            self.sum_excess = self.sum_excess - excess;
            self.sum_downside_sq = self.sum_downside_sq - down_sq;
            self.num_downside -= usize::from(old < self.target);
            self.evictions += 1;
            if self.evictions == self.window_len.get() {
                self.evictions = 0;
                let target = self.target;
                (self.sum_excess, self.sum_downside_sq) = self
                    .window
                    .iter()
                    .map(|r| excess_and_downside(*r, target))
                    .fold((T::zero(), T::zero()), |(a, b), (e, d)| (a + e, b + d));
            }
        }
        if self.num_downside == 0 {
            // Exactly zero instead of a rounding residual, which `sqrt` would amplify.
            self.sum_downside_sq = T::zero();
        }
        debug_assert!(
            self.window.len() <= self.window_len.get(),
            "window is bounded"
        );
        debug_assert!(
            self.num_downside <= self.window.len(),
            "downside count is bounded"
        );
    }

    fn last(&self) -> Option<T> {
        if self.num_downside == 0 {
            return None;
        }
        let dd = self.downside_deviation()?;
        debug_assert!(dd >= T::zero(), "a deviation is non-negative");
        // Downside deviations below about 1e-162 square to zero.
        if dd == T::zero() {
            return None;
        }
        let n = T::from(self.window.len()).expect("can convert");
        let out = self.sum_excess / n / dd;
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

    fn sortino(window_len: usize, target: f64) -> SortinoRatio<f64, Echo<f64>> {
        SortinoRatio::new(Echo::new(), NonZeroUsize::new(window_len).unwrap(), target)
    }

    /// Direct Sortino ratio, `None` without downside.
    fn naive_sortino(w: &[f64], target: f64) -> Option<f64> {
        let n = w.len() as f64;
        let mean = w.iter().sum::<f64>() / n;
        let dd2 = w.iter().map(|r| (r - target).min(0.0).powi(2)).sum::<f64>() / n;
        if !w.iter().any(|r| *r < target) {
            return None;
        }
        Some((mean - target) / dd2.sqrt())
    }

    #[test]
    fn sortino_ratio_plot() {
        let mut v = SortinoRatio::new(
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
        plot_values(out, "img/sortino_ratio.png").unwrap();
    }

    /// Deviations below about 1e-162 square to 0, so the downside deviation
    /// vanishes although a return is below the target.
    #[test]
    fn sortino_ratio_tiny_downside_is_not_ready() {
        let mut v = sortino(2, 0.0);
        for r in [-1e-200, 1e-200, -1e-200] {
            v.update(r);
            assert!(v.last().is_none_or(f64::is_finite), "{:?}", v.last());
        }
        assert_eq!(v.last(), None);
    }

    #[test]
    fn sortino_ratio_warmup_and_hand_computed() {
        let mut v = sortino(4, 0.0);
        assert_eq!(v.window_len().get(), 4);
        assert_eq!(v.target(), 0.0);
        for r in [0.02, -0.01, 0.03] {
            v.update(r);
            assert_eq!(v.last(), None);
            assert_eq!(v.downside_deviation(), None);
        }
        v.update(-0.02);
        // mean = 0.005, DD = sqrt((0.0001 + 0.0004) / 4) = sqrt(0.000125)
        let dd = 0.000125_f64.sqrt();
        assert!((v.downside_deviation().unwrap() - dd).abs() < 1e-15);
        assert!((v.last().unwrap() - 0.005 / dd).abs() < 1e-12);
    }

    #[test]
    fn sortino_ratio_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(161);
        for (window_len, target) in [(1, 0.0), (2, 0.0), (20, 0.001), (250, -0.01)] {
            let returns: Vec<f64> = (0..20_000)
                .map(|i| 0.02 * (i as f64 / 3_000.0).sin() + (rng.f64() - 0.5) * 0.04)
                .collect();
            let mut v = sortino(window_len, target);
            for (i, r) in returns.iter().enumerate() {
                v.update(*r);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let w = &returns[i + 1 - window_len..=i];
                match (v.last(), naive_sortino(w, target)) {
                    (None, None) => {}
                    (Some(got), Some(expected)) => assert!(
                        (got - expected).abs() <= 1e-8 * expected.abs().max(1.0),
                        "step {i}, n {window_len}: {got} vs {expected}"
                    ),
                    (got, expected) => panic!("step {i}: {got:?} vs {expected:?}"),
                }
            }
        }
    }

    #[test]
    fn sortino_ratio_no_downside_is_none() {
        let mut v = sortino(3, 0.0);
        for r in [0.01, 0.0, 0.02, 0.03] {
            v.update(r);
            assert_eq!(v.last(), None);
        }
        assert_eq!(v.downside_deviation(), Some(0.0));
        v.update(-0.01);
        assert!(v.last().unwrap() > 0.0);
        // The downside return leaves the window again.
        for r in [0.01, 0.02, 0.01] {
            v.update(r);
        }
        assert_eq!(v.last(), None);
        assert_eq!(v.downside_deviation(), Some(0.0));
    }

    #[test]
    fn sortino_ratio_all_below_target() {
        // All returns below the target: DD is the RMS of the shortfalls.
        let mut v = sortino(2, 0.01);
        v.update(0.0);
        v.update(-0.01);
        // mean excess -0.015, DD = sqrt((0.0001 + 0.0004) / 2)
        let expected = -0.015 / 0.00025_f64.sqrt();
        assert!((v.last().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "target must be finite")]
    fn sortino_ratio_nan_target() {
        let _ = sortino(10, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "target must be finite")]
    fn sortino_ratio_infinite_target() {
        let _ = sortino(10, f64::INFINITY);
    }
}

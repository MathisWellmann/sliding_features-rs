//! Rolling sample skewness over a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::statistics_moments::StatisticsMoments;
use crate::View;

/// Rolling sample skewness over a sliding window.
///
/// Computes the adjusted Fisher-Pearson standardized moment coefficient
/// `G1 = (sqrt(n (n - 1)) / (n - 2)) * g1`, with the population (biased)
/// skewness `g1 = m3 / m2^1.5` (`m2`, `m3` the second and third central
/// moments of the window). `G1` is the convention used by Excel's `SKEW`,
/// SciPy's `skew(..., bias=False)` and pandas' `Series.skew` (Joanes &
/// Gill, 1998, "Comparing measures of sample skewness and kurtosis", The
/// Statistician 47(1), 183-189); chosen over the plain population estimator
/// `g1` because it corrects its small-sample bias and matches what users of
/// those tools expect.
///
/// `last()` is `None` until `window_len` values were observed, and
/// `Some(0)` if the window has zero variance (no asymmetry is defined
/// then). Symmetric distributions give values near 0; a positive value
/// indicates a right (positive) tail, a negative value a left tail.
///
/// # Complexity
/// O(1) amortized per update (see the crate-private `StatisticsMoments`
/// helper); O(1) for `last()`.
#[derive(Debug, Clone)]
pub struct Skewness<T, V> {
    view: V,
    moments: StatisticsMoments<T>,
}

impl<T, V> Skewness<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling skewness with a chained `View` and a sliding
    /// window length.
    ///
    /// # Panics
    /// If `window_len` is below 3: the bias adjustment divides by `n - 2`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 3, "window_len must be >= 3");
        Self {
            view,
            moments: StatisticsMoments::new(window_len),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.moments.window_len()
    }
}

impl<T, V> View<T> for Skewness<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        self.moments.push(val);
    }

    fn last(&self) -> Option<T> {
        if !self.moments.is_full() {
            return None;
        }
        let m2 = self.moments.central_moment2();
        if m2 <= T::zero() {
            return Some(T::zero());
        }
        let m3 = self.moments.central_moment3();
        let g1 = m3 / m2.powf(T::from(1.5).expect("can convert"));

        let n = T::from(self.window_len().get()).expect("can convert");
        let one = T::one();
        let two = one + one;
        let adjust = (n * (n - one)).sqrt() / (n - two);
        let out = adjust * g1;
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
        test_data::TEST_DATA,
    };

    fn skew(window_len: usize) -> Skewness<f64, Echo<f64>> {
        Skewness::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Naive adjusted Fisher-Pearson skewness computed directly from a slice.
    fn naive_skewness(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let m2 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return 0.0;
        }
        let g1 = m3 / m2.powf(1.5);
        ((n * (n - 1.0)).sqrt() / (n - 2.0)) * g1
    }

    #[test]
    fn skewness_plot() {
        let mut v = skew(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/skewness.png").unwrap();
    }

    #[test]
    fn skewness_warmup() {
        let mut v = skew(4);
        for x in [1.0, 2.0, 3.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(4.0);
        assert!(v.last().is_some());
    }

    #[test]
    fn skewness_symmetric_is_near_zero() {
        let mut v = skew(200);
        // Symmetric values around 0.
        for i in -100..=100 {
            v.update(i as f64);
        }
        assert!(v.last().unwrap().abs() < 1e-9);
    }

    #[test]
    fn skewness_constant_is_zero() {
        let mut v = skew(5);
        for _ in 0..20 {
            v.update(7.0);
        }
        assert_eq!(v.last(), Some(0.0));
    }

    #[test]
    fn skewness_vs_naive_with_offset() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        for window_len in [3, 4, 10, 50] {
            let mut v = skew(window_len);
            // Large offset to stress numerical stability.
            let vals: Vec<f64> = (0..1_000).map(|_| 1e6 + rng.f64() * 5.0).collect();
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let expected = naive_skewness(&vals[i + 1 - window_len..=i]);
                let got = v.last().expect("warm");
                // A window of only 3-4 points divides by (n - 2) = 1..2, so
                // the bias adjustment amplifies float round-off; the naive
                // reference itself subtracts the ~1e6 offset directly
                // (catastrophic cancellation), so it is the less accurate
                // side here. A loose relative tolerance only guards against
                // a gross mismatch, not float noise.
                assert!(
                    (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "window_len {window_len} step {i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be >= 3")]
    fn skewness_window_len_too_small() {
        let _ = skew(2);
    }
}

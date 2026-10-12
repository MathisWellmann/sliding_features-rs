//! Rolling coefficient of variation over a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::WelfordOnline;
use crate::{
    View,
    pure_functions::Echo,
};

/// Rolling coefficient of variation (relative standard deviation) over a
/// sliding window.
///
/// Computes `CV = sigma / |mu|`, with `mu` and `sigma` the sample mean and
/// sample (`n - 1`) standard deviation of the window (as computed by
/// [`WelfordOnline`]). The absolute value of the mean is used, rather than
/// the signed mean, so `CV` stays a non-negative dispersion measure
/// regardless of the sign of the data (the plain `sigma / mu` would flip
/// sign whenever `mu` crosses zero, which is not a meaningful change in
/// relative dispersion); this is the convention of, e.g., NumPy/SciPy's
/// `variation(..., ddof=1)` with `abs(mean)` guarding its denominator.
///
/// `last()` is `None` until `window_len` (>= 2, so a sample standard
/// deviation is defined) values were observed, and `None` for every window
/// whose mean is exactly `0` (`CV` is undefined there, since the measure
/// is relative to the mean). Output range: `[0, infinity)`.
///
/// # Complexity
/// O(1) per update (see [`WelfordOnline`]); O(1) for `last()`.
#[derive(Debug, Clone)]
pub struct CoefficientOfVariation<T: Float, V> {
    view: V,
    welford: WelfordOnline<T, Echo<T>>,
    count: usize,
}

impl<T, V> CoefficientOfVariation<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling coefficient of variation with a chained `View`
    /// and a sliding window length.
    ///
    /// # Panics
    /// If `window_len` is below 2: a sample standard deviation needs at
    /// least 2 values.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 2, "window_len must be >= 2");
        Self {
            view,
            welford: WelfordOnline::new(Echo::new(), window_len),
            count: 0,
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.welford.window_len()
    }
}

impl<T, V> View<T> for CoefficientOfVariation<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.welford.update(val);
        self.count = (self.count + 1).min(self.window_len().get());
    }

    fn last(&self) -> Option<T> {
        if self.count < self.window_len().get() {
            return None;
        }
        let mean = self.welford.mean();
        if mean == T::zero() {
            return None;
        }
        let std_dev = self.welford.variance().sqrt();
        let out = std_dev / mean.abs();
        debug_assert!(out >= T::zero(), "coefficient of variation is non-negative");
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    fn cv(window_len: usize) -> CoefficientOfVariation<f64, Echo<f64>> {
        CoefficientOfVariation::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    fn naive_cv(vals: &[f64]) -> Option<f64> {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        if mean == 0.0 {
            return None;
        }
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Some(var.sqrt() / mean.abs())
    }

    #[test]
    fn coefficient_of_variation_plot() {
        let mut v = cv(16);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/coefficient_of_variation.png").unwrap();
    }

    #[test]
    fn coefficient_of_variation_warmup() {
        let mut v = cv(4);
        for x in [1.0, 2.0, 3.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(4.0);
        assert!(v.last().is_some());
    }

    #[test]
    fn coefficient_of_variation_zero_mean_is_none() {
        let mut v = cv(3);
        for x in [-1.0, 0.0, 1.0] {
            v.update(x);
        }
        assert_eq!(v.last(), None);
    }

    #[test]
    fn coefficient_of_variation_negative_mean_uses_absolute_value() {
        // Mirror image of a positive-mean window should give the same CV.
        let mut pos = cv(3);
        let mut neg = cv(3);
        for x in [10.0, 12.0, 8.0] {
            pos.update(x);
            neg.update(-x);
        }
        assert_eq!(pos.last(), neg.last());
    }

    #[test]
    fn coefficient_of_variation_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(13);
        for window_len in [2, 3, 10, 40] {
            let mut v = cv(window_len);
            let vals: Vec<f64> = (0..500).map(|_| rng.f64() * 20.0 - 5.0).collect();
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let window = &vals[i + 1 - window_len..=i];
                let expected = naive_cv(window);
                let got = v.last();
                match expected {
                    None => assert_eq!(got, None, "window_len {window_len} step {i}"),
                    Some(expected) => {
                        let got = got.expect("warm and mean != 0");
                        assert!(
                            (got - expected).abs() < 1e-9,
                            "window_len {window_len} step {i}: got {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be >= 2")]
    fn coefficient_of_variation_window_len_too_small() {
        let _ = cv(1);
    }
}

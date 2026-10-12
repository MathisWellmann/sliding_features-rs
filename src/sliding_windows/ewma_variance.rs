//! Exponentially weighted moving variance.

use std::num::NonZeroUsize;

use num::Float;

use crate::View;

/// Exponentially weighted moving variance (EWMA variance).
///
/// Uses Finch's incremental formulation of a weighted variance (T. Finch,
/// "Incremental calculation of weighted mean and variance", University of
/// Cambridge Computing Service, 2009):
///
/// ```text
/// delta = x_t - mean_{t-1}
/// mean_t = mean_{t-1} + alpha * delta
/// var_t  = (1 - alpha) * (var_{t-1} + alpha * delta^2)
/// ```
///
/// which is the recursive, numerically stable update behind most
/// RiskMetrics-style exponentially weighted volatility estimators. This is
/// the *biased* (population) exponentially weighted variance: there is no
/// small-sample bias correction, matching `pandas.ewm(adjust=False).var(bias=True)`.
///
/// Unlike the other sliding-window indicators in this crate the estimate is
/// not confined to a hard window: every past observation contributes, with
/// a weight that decays geometrically by `(1 - alpha)` per step. `window_len`
/// only controls the default smoothing factor `alpha = 2 / (window_len + 1)`
/// (same convention as [`super::Ema`]) and the warm-up length before
/// `last()` starts returning a value.
///
/// `last()` is `None` until `window_len` values were observed. Output
/// range: `[0, infinity)`.
///
/// # Complexity
/// O(1) per update; O(1) for `last()`.
#[derive(Debug, Clone)]
pub struct EwmaVariance<T, V> {
    view: V,
    window_len: usize,
    alpha: T,
    mean: T,
    var: T,
    n_observed_values: usize,
}

impl<T, V> EwmaVariance<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `EwmaVariance` with a chained `View`, a window length
    /// and the default smoothing factor `alpha = 2 / (window_len + 1)`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self::with_alpha(
            view,
            window_len,
            T::from(2.0).expect("can convert")
                / (T::from(window_len.get()).expect("can convert") + T::one()),
        )
    }

    /// Create a new `EwmaVariance` with a chained `View`, a window length
    /// (used only for the warm-up length) and a custom smoothing factor
    /// `alpha`.
    ///
    /// # Panics
    /// If `alpha` is not in `(0, 1]`.
    pub fn with_alpha(view: V, window_len: NonZeroUsize, alpha: T) -> Self {
        assert!(alpha > T::zero(), "alpha must be > 0");
        assert!(alpha <= T::one(), "alpha must be <= 1");
        Self {
            view,
            window_len: window_len.get(),
            alpha,
            mean: T::zero(),
            var: T::zero(),
            n_observed_values: 0,
        }
    }

    /// The window length used for the default smoothing factor and the
    /// warm-up length.
    #[inline]
    pub fn window_len(&self) -> usize {
        self.window_len
    }
}

impl<T, V> View<T> for EwmaVariance<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.n_observed_values == 0 {
            self.mean = val;
            self.var = T::zero();
            self.n_observed_values = 1;
            return;
        }

        let delta = val - self.mean;
        self.mean = self.mean + self.alpha * delta;
        self.var = (T::one() - self.alpha) * (self.var + self.alpha * delta * delta);
        self.n_observed_values += 1;
    }

    fn last(&self) -> Option<T> {
        if self.n_observed_values < self.window_len {
            return None;
        }
        debug_assert!(self.var >= T::zero(), "variance is non-negative");
        debug_assert!(self.var.is_finite(), "value must be finite");
        Some(self.var)
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

    fn ewma_var(window_len: usize) -> EwmaVariance<f64, Echo<f64>> {
        EwmaVariance::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    fn naive_ewma_var(vals: &[f64], alpha: f64) -> f64 {
        let mut mean = vals[0];
        let mut var = 0.0;
        for &x in &vals[1..] {
            let delta = x - mean;
            mean += alpha * delta;
            var = (1.0 - alpha) * (var + alpha * delta * delta);
        }
        var
    }

    #[test]
    fn ewma_variance_plot() {
        let mut v = ewma_var(16);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/ewma_variance.png").unwrap();
    }

    #[test]
    fn ewma_variance_warmup() {
        let mut v = ewma_var(4);
        for x in [1.0, 2.0, 3.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(4.0);
        assert!(v.last().is_some());
    }

    #[test]
    fn ewma_variance_constant_input_is_zero() {
        let mut v = ewma_var(4);
        for _ in 0..10 {
            v.update(5.0);
        }
        assert_eq!(v.last(), Some(0.0));
    }

    #[test]
    fn ewma_variance_non_negative() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let mut v = ewma_var(8);
        for _ in 0..500 {
            v.update(rng.f64() * 100.0 - 50.0);
            if let Some(val) = v.last() {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn ewma_variance_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(21);
        let window_len = 10;
        let alpha = 2.0 / (window_len as f64 + 1.0);
        let vals: Vec<f64> = (0..200).map(|_| rng.f64() * 10.0 - 5.0).collect();
        let mut v = ewma_var(window_len);
        for (i, x) in vals.iter().enumerate() {
            v.update(*x);
            if i + 1 < window_len {
                assert_eq!(v.last(), None);
                continue;
            }
            let expected = naive_ewma_var(&vals[..=i], alpha);
            let got = v.last().expect("warm");
            assert!(
                (got - expected).abs() < 1e-9,
                "step {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "alpha must be > 0")]
    fn ewma_variance_alpha_too_small() {
        let _ = EwmaVariance::with_alpha(Echo::new(), NonZeroUsize::new(4).unwrap(), 0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be <= 1")]
    fn ewma_variance_alpha_too_large() {
        let _ = EwmaVariance::with_alpha(Echo::new(), NonZeroUsize::new(4).unwrap(), 1.1);
    }
}

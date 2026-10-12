//! Coefficient of Determination (R^2) of a sliding linear regression

use std::num::NonZeroUsize;

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Coefficient of determination `R^2` of the least-squares line fitted to
/// the last `window_len` samples of the chained view (same fit as
/// [`super::LinRegSlope`]). Measures how well a straight line explains the
/// window: `1` for a perfect line, `0` for a fit no better than the mean.
/// Output is in `[0, 1]`.
///
/// `R^2` of a simple linear regression is the squared Pearson correlation
/// coefficient between the sample index and the value. When the window has
/// zero variance (every sample equal), the usual ratio is `0/0`; this is
/// defined as `R^2 = 1`, since the fitted (constant) line then matches every
/// sample exactly.
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct RSquared<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> RSquared<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    ///
    /// # Panics
    /// Panics if `window_len < 2`, since at least two points are needed to fit a line.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window: LinRegWindow::new(window_len),
        }
    }

    /// The sliding window length.
    #[inline]
    #[must_use]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }
}

impl<T, V> View<T> for RSquared<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        self.window.update(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.window.r_squared().inspect(|out| {
            debug_assert!(out.is_finite(), "value must be finite");
            debug_assert!(
                *out >= T::zero() && *out <= T::one(),
                "R^2 must be in [0, 1]"
            );
        })
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

    /// Naive O(n) reference: squared Pearson correlation between x = 0..n-1 and y.
    fn naive_r_squared(ys: &[f64]) -> f64 {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        let sum_ysqr: f64 = ys.iter().map(|y| y.powi(2)).sum();
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = (n * sum_xsqr - sum_x * sum_x) * (n * sum_ysqr - sum_y * sum_y);
        numerator * numerator / denominator
    }

    #[test]
    fn warmup_then_ready() {
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            r2.update(v);
            assert_eq!(r2.last(), None);
        }
        r2.update(5.0);
        assert_eq!(r2.last(), Some(1.0));
    }

    #[test]
    fn perfect_line_is_one() {
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..20 {
            r2.update(f64::from(i) * -3.0 + 1.0);
        }
        assert!((r2.last().expect("warmed up") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_variance_window_is_one() {
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(6).unwrap());
        for _ in 0..12 {
            r2.update(5.0);
        }
        assert_eq!(r2.last(), Some(1.0));
    }

    #[test]
    fn output_is_bounded_in_zero_one_on_real_data() {
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            r2.update(*v);
            if let Some(out) = r2.last() {
                assert!((0.0..=1.0).contains(&out), "out of range: {out}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(77);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, v) in vals.iter().enumerate() {
            r2.update(*v);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let expected = naive_r_squared(&vals[i + 1 - WINDOW_LEN..=i]);
            let got = r2.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = RSquared::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn r_squared_plot() {
        let mut r2 = RSquared::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            r2.update(*v);
            if let Some(val) = r2.last() {
                out.push(val);
            }
        }
        let filename = "img/r_squared.png";
        plot_values(out, filename).unwrap();
    }
}

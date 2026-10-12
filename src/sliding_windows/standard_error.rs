//! Standard Error of a sliding linear regression

use std::num::NonZeroUsize;

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Standard error of the least-squares line fitted to the last `window_len`
/// samples of the chained view (same fit as [`super::LinRegSlope`]).
///
/// Reports the sample standard deviation of the residuals, using `n - 2`
/// degrees of freedom (two parameters, slope and intercept, are estimated
/// from the data): `sqrt(SSE / (window_len - 2))`. Output is unbounded and
/// non-negative, in the same units as the input; `0` means the window lies
/// exactly on the fitted line.
///
/// Reference: TA-Lib `STDDEV`-style residual error used alongside
/// `LINEARREG`, <https://ta-lib.org/functions/linearreg.html>
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct StandardError<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> StandardError<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    ///
    /// # Panics
    /// Panics if `window_len < 3`, since `n - 2` degrees of freedom must be
    /// positive to define a residual standard error.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= 3,
            "window_len must be at least 3, got {}",
            window_len.get()
        );
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

impl<T, V> View<T> for StandardError<T, V>
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
        self.window.sse().map(|sse| {
            let dof = T::from(self.window_len().get() - 2).expect("can convert");
            let out = (sse / dof).sqrt();
            debug_assert!(out.is_finite(), "value must be finite");
            debug_assert!(out >= T::zero(), "standard error must be non-negative");
            out
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

    /// Naive O(n) reference: OLS fit with x = 0..n-1, then residual standard
    /// error with n - 2 degrees of freedom.
    fn naive_standard_error(ys: &[f64]) -> f64 {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        let divisor = n * sum_xsqr - sum_x * sum_x;
        let m = (n * sum_xy - sum_x * sum_y) / divisor;
        let b = (sum_y - m * sum_x) / n;
        let sse: f64 = ys
            .iter()
            .enumerate()
            .map(|(i, y)| {
                let fitted = b + m * i as f64;
                (y - fitted).powi(2)
            })
            .sum();
        (sse / (n - 2.0)).sqrt()
    }

    #[test]
    fn warmup_then_ready() {
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            se.update(v);
            assert_eq!(se.last(), None);
        }
        se.update(5.0);
        assert!(se.last().is_some());
    }

    #[test]
    fn perfect_line_is_zero() {
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..20 {
            se.update(f64::from(i) * 2.0 - 7.0);
        }
        assert!((se.last().expect("warmed up") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn constant_input_is_zero() {
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..16 {
            se.update(4.0);
        }
        assert!((se.last().expect("warmed up") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn output_is_non_negative_on_real_data() {
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            se.update(*v);
            if let Some(out) = se.last() {
                assert!(out >= 0.0, "must be non-negative: {out}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(321);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, v) in vals.iter().enumerate() {
            se.update(*v);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let expected = naive_standard_error(&vals[i + 1 - WINDOW_LEN..=i]);
            let got = se.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 3")]
    fn window_len_two_panics() {
        let _ = StandardError::<f64, _>::new(Echo::new(), NonZeroUsize::new(2).unwrap());
    }

    #[test]
    fn standard_error_plot() {
        let mut se = StandardError::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            se.update(*v);
            if let Some(val) = se.last() {
                out.push(val);
            }
        }
        let filename = "img/standard_error.png";
        plot_values(out, filename).unwrap();
    }
}

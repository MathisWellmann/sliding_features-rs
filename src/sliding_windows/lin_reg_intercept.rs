// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Linear Regression Intercept

use std::num::NonZeroUsize;

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Linear Regression Intercept (TA-Lib `LINEARREG_INTERCEPT`).
///
/// Intercept `b` of the least-squares line `y = b + m*x` fitted to the last
/// `window_len` samples of the chained view, with `x = 0` at the oldest
/// sample in the window and `x = window_len - 1` at the newest one. Because
/// `x = 0` is the *oldest* bar, `b` is the fitted value at the oldest bar in
/// the window, not a projection to "time zero" in the usual sense. Output is
/// unbounded, in the same units as the input.
///
/// Reference: TA-Lib `LINEARREG_INTERCEPT`,
/// <https://ta-lib.org/functions/linearreg_intercept.html>
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct LinRegIntercept<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> LinRegIntercept<T, V>
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

impl<T, V> View<T> for LinRegIntercept<T, V>
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
        self.window.fit().map(|(_, b)| {
            debug_assert!(b.is_finite(), "value must be finite");
            b
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

    /// Naive O(n) reference: ordinary least squares intercept with x = 0..n-1
    /// (x = 0 at the oldest sample, matching the TA-Lib convention).
    fn naive_intercept(ys: &[f64]) -> f64 {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        let divisor = n * sum_xsqr - sum_x * sum_x;
        let m = (n * sum_xy - sum_x * sum_y) / divisor;
        (sum_y - m * sum_x) / n
    }

    #[test]
    fn warmup_then_ready() {
        let mut v = LinRegIntercept::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for x in [1.0, 2.0, 3.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(5.0);
        assert_eq!(v.last(), Some(1.0));
    }

    #[test]
    fn perfect_line_intercept_is_oldest_bar() {
        let mut v = LinRegIntercept::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        // y = 2.5*t - 3.0, oldest bar of the final window is at t = 10.
        for t in 0..20 {
            v.update(f64::from(t) * 2.5 - 3.0);
        }
        let expected = 10.0 * 2.5 - 3.0;
        let out = v.last().expect("warmed up");
        assert!(
            (out - expected).abs() < 1e-9,
            "got {out}, expected {expected}"
        );
    }

    #[test]
    fn constant_input_intercept_equals_constant() {
        let mut v = LinRegIntercept::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..16 {
            v.update(42.0);
        }
        assert!((v.last().expect("warmed up") - 42.0).abs() < 1e-9);
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(321);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut v = LinRegIntercept::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, val) in vals.iter().enumerate() {
            v.update(*val);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let expected = naive_intercept(&vals[i + 1 - WINDOW_LEN..=i]);
            let got = v.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = LinRegIntercept::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn lin_reg_intercept_plot() {
        let mut v = LinRegIntercept::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        let filename = "img/lin_reg_intercept.png";
        plot_values(out, filename).unwrap();
    }
}

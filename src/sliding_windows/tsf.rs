// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Time Series Forecast

use std::num::NonZeroUsize;

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Time Series Forecast (TA-Lib `TSF`).
///
/// Projects the least-squares line `y = b + m*x` fitted to the last
/// `window_len` samples of the chained view (same fit as
/// [`super::LinRegSlope`]) one bar beyond the newest sample: `b + m*window_len`.
/// Output is unbounded, in the same units as the input.
///
/// The fitted value at the newest bar itself (TA-Lib `LINEARREG`,
/// `b + m*(window_len - 1)`) is also exposed via [`Tsf::linreg`].
///
/// Reference: TA-Lib `TSF`, <https://ta-lib.org/functions/tsf.html>; TA-Lib
/// `LINEARREG`, <https://ta-lib.org/functions/linearreg.html>
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct Tsf<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> Tsf<T, V>
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

    /// The fitted value at the newest bar in the window (TA-Lib `LINEARREG`),
    /// or `None` during warmup.
    #[must_use]
    pub fn linreg(&self) -> Option<T> {
        self.window.fit().map(|(m, b)| {
            let n_minus_1 = T::from(self.window_len().get() - 1).expect("can convert");
            let out = b + m * n_minus_1;
            debug_assert!(out.is_finite(), "value must be finite");
            out
        })
    }
}

impl<T, V> View<T> for Tsf<T, V>
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
        self.window.fit().map(|(m, b)| {
            let n = T::from(self.window_len().get()).expect("can convert");
            let out = b + m * n;
            debug_assert!(out.is_finite(), "value must be finite");
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

    /// Naive O(n) reference: ordinary least squares fit with x = 0..n-1,
    /// then project to `x = n` (TSF) and `x = n - 1` (LINEARREG).
    fn naive_fit(ys: &[f64]) -> (f64, f64) {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        let divisor = n * sum_xsqr - sum_x * sum_x;
        let m = (n * sum_xy - sum_x * sum_y) / divisor;
        let b = (sum_y - m * sum_x) / n;
        (m, b)
    }

    #[test]
    fn warmup_then_ready() {
        let mut t = Tsf::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            t.update(v);
            assert_eq!(t.last(), None);
            assert_eq!(t.linreg(), None);
        }
        t.update(5.0);
        assert_eq!(t.last(), Some(6.0));
        assert_eq!(t.linreg(), Some(5.0));
    }

    #[test]
    fn perfect_line_forecasts_next_point() {
        let mut t = Tsf::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..20 {
            t.update(f64::from(i) * 2.5 - 3.0);
        }
        // the series is exactly y = 2.5*t - 3.0, so TSF at t=20 should be y(20).
        let expected_next = 2.5 * 20.0 - 3.0;
        let out = t.last().expect("warmed up");
        assert!((out - expected_next).abs() < 1e-9, "got {out}");

        let expected_endpoint = 2.5 * 19.0 - 3.0;
        let endpoint = t.linreg().expect("warmed up");
        assert!(
            (endpoint - expected_endpoint).abs() < 1e-9,
            "got {endpoint}"
        );
    }

    #[test]
    fn constant_input_forecasts_the_constant() {
        let mut t = Tsf::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..16 {
            t.update(11.0);
        }
        assert!((t.last().expect("warmed up") - 11.0).abs() < 1e-9);
        assert!((t.linreg().expect("warmed up") - 11.0).abs() < 1e-9);
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(246);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut t = Tsf::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, v) in vals.iter().enumerate() {
            t.update(*v);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let (m, b) = naive_fit(&vals[i + 1 - WINDOW_LEN..=i]);
            let expected_tsf = b + m * WINDOW_LEN as f64;
            let expected_linreg = b + m * (WINDOW_LEN as f64 - 1.0);
            let got_tsf = t.last().expect("warmed up");
            let got_linreg = t.linreg().expect("warmed up");
            assert!(
                (expected_tsf - got_tsf).abs() < 1e-6,
                "tsf at {i}: {expected_tsf} vs {got_tsf}"
            );
            assert!(
                (expected_linreg - got_linreg).abs() < 1e-6,
                "linreg at {i}: {expected_linreg} vs {got_linreg}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = Tsf::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn tsf_plot() {
        let mut t = Tsf::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            t.update(*v);
            if let Some(val) = t.last() {
                out.push(val);
            }
        }
        let filename = "img/tsf.png";
        plot_values(out, filename).unwrap();
    }
}

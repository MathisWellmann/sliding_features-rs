//! Linear Regression Slope

use std::num::NonZeroUsize;

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Linear Regression Slope (TA-Lib `LINEARREG_SLOPE`).
///
/// Slope `m` of the least-squares line `y = b + m*x` fitted to the last
/// `window_len` samples of the chained view, with `x = 0` at the oldest
/// sample in the window and `x = window_len - 1` at the newest one.
/// Positive values mean the window trends upward over time. Output is
/// unbounded, in units of "value change per bar".
///
/// Reference: TA-Lib `LINEARREG_SLOPE`, <https://ta-lib.org/functions/linearreg_slope.html>
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct LinRegSlope<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> LinRegSlope<T, V>
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

impl<T, V> View<T> for LinRegSlope<T, V>
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
        self.window.fit().map(|(m, _)| {
            debug_assert!(m.is_finite(), "value must be finite");
            m
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

    /// Naive O(n) reference: ordinary least squares slope with x = 0..n-1.
    fn naive_slope(ys: &[f64]) -> f64 {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        (n * sum_xy - sum_x * sum_y) / (n * sum_xsqr - sum_x * sum_x)
    }

    #[test]
    fn warmup_then_ready() {
        let mut s = LinRegSlope::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            s.update(v);
            assert_eq!(s.last(), None);
        }
        s.update(5.0);
        assert_eq!(s.last(), Some(1.0));
    }

    #[test]
    fn perfect_line() {
        let mut s = LinRegSlope::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..20 {
            s.update(f64::from(i) * 2.5 - 3.0);
        }
        let out = s.last().expect("warmed up");
        assert!((out - 2.5).abs() < 1e-9, "got {out}");
    }

    #[test]
    fn constant_input_has_zero_slope() {
        let mut s = LinRegSlope::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..16 {
            s.update(42.0);
        }
        assert!((s.last().expect("warmed up")).abs() < 1e-9);
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(123);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut s = LinRegSlope::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, v) in vals.iter().enumerate() {
            s.update(*v);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let expected = naive_slope(&vals[i + 1 - WINDOW_LEN..=i]);
            let got = s.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = LinRegSlope::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn lin_reg_slope_plot() {
        let mut s = LinRegSlope::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            s.update(*v);
            if let Some(val) = s.last() {
                out.push(val);
            }
        }
        let filename = "img/lin_reg_slope.png";
        plot_values(out, filename).unwrap();
    }
}

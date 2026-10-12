//! Linear Regression Angle

use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use num::Float;

use super::linreg_window::LinRegWindow;
use crate::View;

/// Linear Regression Angle (TA-Lib `LINEARREG_ANGLE`).
///
/// Angle, in degrees, of the slope `m` of the least-squares line fitted to
/// the last `window_len` samples of the chained view (same fit as
/// [`super::LinRegSlope`]): `atan(m) * 180 / pi`. Output is in `(-90, 90)`
/// degrees, with `0` meaning a flat window and the sign matching the trend
/// direction.
///
/// Reference: TA-Lib `LINEARREG_ANGLE`,
/// <https://ta-lib.org/functions/linearreg_angle.html>
///
/// # Complexity
/// O(1) amortized per update: the regression sums are kept incrementally
/// using a fixed-window recurrence and are periodically re-derived from the
/// raw window buffer to bound floating-point drift.
#[derive(Debug, Clone)]
pub struct LinRegAngle<T, V> {
    view: V,
    window: LinRegWindow<T>,
}

impl<T, V> LinRegAngle<T, V>
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

impl<T, V> View<T> for LinRegAngle<T, V>
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
            let degrees_per_radian = T::from(180.0 / PI).expect("can convert");
            let out = m.atan() * degrees_per_radian;
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

    /// Naive O(n) reference: ordinary least squares slope, then atan in degrees.
    fn naive_angle(ys: &[f64]) -> f64 {
        let n = ys.len() as f64;
        let sum_x: f64 = (0..ys.len()).map(|i| i as f64).sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xsqr: f64 = (0..ys.len()).map(|i| (i as f64).powi(2)).sum();
        let divisor = n * sum_xsqr - sum_x * sum_x;
        let m = (n * sum_xy - sum_x * sum_y) / divisor;
        m.atan() * 180.0 / std::f64::consts::PI
    }

    #[test]
    fn warmup_then_ready() {
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            a.update(v);
            assert_eq!(a.last(), None);
        }
        a.update(5.0);
        assert!(a.last().is_some());
    }

    #[test]
    fn perfect_45_degree_line() {
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        // slope of 1.0 is a 45 degree angle.
        for i in 0..20 {
            a.update(f64::from(i));
        }
        let out = a.last().expect("warmed up");
        assert!((out - 45.0).abs() < 1e-9, "got {out}");
    }

    #[test]
    fn constant_input_has_zero_angle() {
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..16 {
            a.update(7.0);
        }
        assert!((a.last().expect("warmed up")).abs() < 1e-9);
    }

    #[test]
    fn output_is_bounded_to_open_90_degrees() {
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(6).unwrap());
        for v in &TEST_DATA {
            a.update(*v);
            if let Some(out) = a.last() {
                assert!(out > -90.0 && out < 90.0, "out of range: {out}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(55);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, v) in vals.iter().enumerate() {
            a.update(*v);
            if i + 1 < WINDOW_LEN {
                continue;
            }
            let expected = naive_angle(&vals[i + 1 - WINDOW_LEN..=i]);
            let got = a.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = LinRegAngle::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn lin_reg_angle_plot() {
        let mut a = LinRegAngle::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            a.update(*v);
            if let Some(val) = a.last() {
                out.push(val);
            }
        }
        let filename = "img/lin_reg_angle.png";
        plot_values(out, filename).unwrap();
    }
}

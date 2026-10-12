//! Shared O(1) sliding least-squares window used by the `LinReg*` regression
//! indicators (`LinRegSlope`, `LinRegIntercept`, `LinRegAngle`, `RSquared`,
//! `Tsf`, `StandardError`).
//!
//! Fits `y = b + m*x` over the last `window_len` samples with `x = 0` for the
//! oldest sample in the window and `x = window_len - 1` for the newest one.
//! This matches the TA-Lib `LINEARREG_*` family convention: `LINEARREG_INTERCEPT`
//! reports `b`, `LINEARREG` reports `b + m*(period-1)` (the fitted value at the
//! newest bar) and `TSF` reports `b + m*period` (one bar beyond the newest bar).
//!
//! # Complexity
//! `update` is O(1) amortized. `sum_x` and `sum_xsqr` only depend on
//! `window_len`, so they are precomputed once in `LinRegWindow::new`.
//! `sum_y` and `sum_xy` are advanced in O(1) per step using the
//! fixed-window recurrence below, with a full O(n) recompute from the raw
//! window buffer every `32 * window_len` updates to bound floating-point
//! drift of the running sums (the same re-anchoring cadence TA-Lib uses for
//! its own `LINEARREG_*` running sums).
//!
//! When the window slides by one sample (oldest `y_old` leaves, `y_new`
//! enters), every retained sample keeps its value but its `x` position
//! decreases by one, so:
//! `sum_xy' = sum_xy - sum_y + y_old + (window_len - 1) * y_new`.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use num::Float;

/// Sliding least-squares window shared by the `LinReg*` regression indicators.
#[derive(Debug, Clone)]
pub(super) struct LinRegWindow<T> {
    window_len: NonZeroUsize,
    buf: VecDeque<T>,
    sum_y: T,
    sum_xy: T,
    sum_yy: T,
    /// `sum(x)` for `x in 0..window_len`, constant for a given `window_len`.
    sum_x: T,
    /// `window_len * sum(x^2) - sum_x^2`, constant for a given `window_len`.
    divisor: T,
    updates_since_recompute: usize,
}

impl<T: Float> LinRegWindow<T> {
    /// Create a new sliding least-squares window.
    ///
    /// # Panics
    /// Panics if `window_len < 2`, since at least two points are needed to fit a line.
    pub(super) fn new(window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= 2,
            "window_len must be at least 2 to fit a line, got {}",
            window_len.get()
        );
        let n = T::from(window_len.get()).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let six = T::from(6.0).expect("can convert");
        let sum_x = n * (n - T::one()) / two;
        let sum_xsqr = n * (n - T::one()) * (two * n - T::one()) / six;
        let divisor = n * sum_xsqr - sum_x * sum_x;
        debug_assert!(
            divisor > T::zero(),
            "divisor must be positive for window_len >= 2"
        );

        Self {
            window_len,
            buf: VecDeque::with_capacity(window_len.get()),
            sum_y: T::zero(),
            sum_xy: T::zero(),
            sum_yy: T::zero(),
            sum_x,
            divisor,
            updates_since_recompute: 0,
        }
    }

    /// Push a new sample, maintaining the running sums.
    pub(super) fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        let window_len = self.window_len.get();

        if self.buf.len() < window_len {
            self.buf.push_back(val);
            if self.buf.len() == window_len {
                self.recompute();
            }
            return;
        }

        let old = self.buf.pop_front().expect("buffer is at full capacity");
        let n_minus_1 = T::from(window_len - 1).expect("can convert");
        self.sum_xy = self.sum_xy - self.sum_y + old + n_minus_1 * val;
        self.sum_y = self.sum_y - old + val;
        self.sum_yy = self.sum_yy - old * old + val * val;
        self.buf.push_back(val);

        self.updates_since_recompute += 1;
        if self.updates_since_recompute >= 32 * window_len {
            self.recompute();
        }
    }

    /// Rebuild `sum_y`, `sum_xy` and `sum_yy` from the raw window buffer, to
    /// bound the floating-point drift of the O(1) running-sum recurrence.
    fn recompute(&mut self) {
        let mut sum_y = T::zero();
        let mut sum_xy = T::zero();
        let mut sum_yy = T::zero();
        for (i, y) in self.buf.iter().enumerate() {
            let x = T::from(i).expect("can convert");
            sum_y = sum_y + *y;
            sum_xy = sum_xy + x * *y;
            sum_yy = sum_yy + *y * *y;
        }
        self.sum_y = sum_y;
        self.sum_xy = sum_xy;
        self.sum_yy = sum_yy;
        self.updates_since_recompute = 0;
    }

    /// Whether the window has observed `window_len` samples yet.
    pub(super) fn is_ready(&self) -> bool {
        self.buf.len() == self.window_len.get()
    }

    /// The configured window length.
    pub(super) fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// Least-squares `(slope, intercept)` of `y = b + m*x`, or `None` during warmup.
    pub(super) fn fit(&self) -> Option<(T, T)> {
        if !self.is_ready() {
            return None;
        }
        let n = T::from(self.window_len.get()).expect("can convert");
        let m = (n * self.sum_xy - self.sum_x * self.sum_y) / self.divisor;
        let b = (self.sum_y - m * self.sum_x) / n;
        debug_assert!(m.is_finite(), "slope must be finite");
        debug_assert!(b.is_finite(), "intercept must be finite");
        Some((m, b))
    }

    /// Coefficient of determination `R^2` of the fit, in `[0, 1]`, or `None`
    /// during warmup.
    ///
    /// `R^2` of a simple linear regression equals the squared Pearson
    /// correlation coefficient between `x` and `y`:
    /// `R^2 = (n*sum_xy - sum_x*sum_y)^2 / (divisor * (n*sum_yy - sum_y^2))`.
    /// When the window has zero variance (every sample equal), the
    /// denominator is zero; the fitted line is then a constant that matches
    /// every sample exactly (zero residuals), so this trivial case is
    /// defined as `R^2 = 1` rather than left undefined.
    pub(super) fn r_squared(&self) -> Option<T> {
        let (m, _) = self.fit()?;
        let n = T::from(self.window_len.get()).expect("can convert");
        let y_variance_term = n * self.sum_yy - self.sum_y * self.sum_y;
        if y_variance_term <= T::zero() {
            return Some(T::one());
        }
        // r^2 = (m * divisor)^2 / (divisor * y_variance_term), with the
        // common `divisor` factor cancelled to reduce rounding error.
        let r_squared = (m * m * self.divisor / y_variance_term)
            .min(T::one())
            .max(T::zero());
        debug_assert!(r_squared.is_finite(), "value must be finite");
        Some(r_squared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_line_slope_and_intercept() {
        let mut w = LinRegWindow::new(NonZeroUsize::new(4).unwrap());
        assert_eq!(w.fit(), None);
        for v in [1.0_f64, 2.0, 3.0, 4.0] {
            w.update(v);
        }
        let (m, b) = w.fit().expect("ready");
        assert!((m - 1.0).abs() < 1e-9);
        assert!((b - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sliding_matches_full_recompute() {
        let r = romu::Rng::from_seed_with_64bit(7);
        let vals: Vec<f64> = (0..200).map(|_| r.f64() * 100.0).collect();
        let window_len = 10;
        let mut w = LinRegWindow::new(NonZeroUsize::new(window_len).unwrap());
        for (i, v) in vals.iter().enumerate() {
            w.update(*v);
            if i + 1 < window_len {
                continue;
            }
            let slice = &vals[i + 1 - window_len..=i];
            let (m, b) = naive_fit(slice);
            let (got_m, got_b) = w.fit().expect("ready");
            assert!(
                (m - got_m).abs() < 1e-6,
                "slope mismatch at {i}: {m} vs {got_m}"
            );
            assert!(
                (b - got_b).abs() < 1e-6,
                "intercept mismatch at {i}: {b} vs {got_b}"
            );
        }
    }

    /// Naive O(n) reference: ordinary least squares with x = 0..n-1.
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
    #[should_panic(expected = "window_len must be at least 2")]
    fn rejects_window_len_one() {
        let _ = LinRegWindow::<f64>::new(NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn r_squared_is_one_for_a_perfect_line() {
        let mut w = LinRegWindow::new(NonZeroUsize::new(5).unwrap());
        for v in [1.0_f64, 3.0, 5.0, 7.0, 9.0] {
            w.update(v);
        }
        assert!((w.r_squared().expect("ready") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn r_squared_is_one_for_zero_variance_window() {
        let mut w = LinRegWindow::new(NonZeroUsize::new(5).unwrap());
        for _ in 0..5 {
            w.update(3.0_f64);
        }
        assert_eq!(w.r_squared(), Some(1.0));
    }

    #[test]
    fn r_squared_matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(99);
        let vals: Vec<f64> = (0..200).map(|_| r.f64() * 100.0).collect();
        let window_len = 10;
        let mut w = LinRegWindow::new(NonZeroUsize::new(window_len).unwrap());
        for (i, v) in vals.iter().enumerate() {
            w.update(*v);
            if i + 1 < window_len {
                continue;
            }
            let expected = naive_r_squared(&vals[i + 1 - window_len..=i]);
            let got = w.r_squared().expect("ready");
            assert!((expected - got).abs() < 1e-6, "at {i}: {expected} vs {got}");
        }
    }

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
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Page-Hinkley change detector.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// The direction of mean shifts that a [`PageHinkley`] test detects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageHinkleyDirection {
    /// Detect increases of the mean (the classic one-sided test).
    Increase,
    /// Detect decreases of the mean.
    Decrease,
    /// Detect increases and decreases of the mean.
    Both,
}

/// Page-Hinkley test for abrupt changes of the mean of a signal.
///
/// References: E. S. Page (1954), "Continuous Inspection Schemes"; J. Gama, R. Sebastiao,
/// P. P. Rodrigues (2013), "On evaluating stream learning algorithms", Machine Learning 90;
/// the update follows `PageHinkley` of scikit-multiflow.
///
/// With the running mean `mean_t` of all values since the last reset (including `x_t`):
/// - increase statistic: `U_t = max(0, alpha U_{t-1} + (x_t - mean_t - delta))`,
/// - decrease statistic: `D_t = max(0, alpha D_{t-1} + (mean_t - x_t - delta))`.
///
/// The clamp at zero is the reflected form of the classic `m_T - min_t m_t`.
/// An alarm is raised when an active statistic exceeds `lambda`, after which the
/// mean and both statistics are reset.
///
/// Parameters:
/// - `direction`: which shifts are detected, see [`PageHinkleyDirection`].
/// - `min_instances`: the number of values after the start or a reset before alarms
///   are allowed, so the running mean is estimated first (scikit-multiflow: 30).
/// - `delta`: the magnitude of tolerated changes, in units of the input
///   (scikit-multiflow: 0.005).
/// - `lambda`: the detection threshold, in units of the input (scikit-multiflow: 50).
/// - `alpha`: the forgetting factor in `(0, 1]` for the statistics; `1` is the classic
///   test without forgetting (scikit-multiflow: `1 - 0.0001`).
///
/// Output: the statistic of the latest update, `U_t` for `Increase`, `-D_t` for `Decrease`
/// and for `Both` `U_t` if `U_t >= D_t`, else `-D_t`. After an alarm, `last()` still
/// reports the statistic that crossed `lambda`. `alarm()` reports the alarm of the latest
/// update.
///
/// Warm-up: `None` until `min_instances` values of the chained `View` were observed, then
/// always `Some` (also during the re-estimation after a reset, where alarms are suppressed).
/// Complexity: O(1) time per update and O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct PageHinkley<T, V> {
    view: V,
    /// The direction of mean shifts that are detected.
    #[getset(get_copy = "pub")]
    direction: PageHinkleyDirection,
    /// The number of values after the start or a reset before alarms are allowed.
    #[getset(get_copy = "pub")]
    min_instances: NonZeroUsize,
    delta: T,
    lambda: T,
    alpha: T,
    /// Number of values since the last reset.
    count: usize,
    mean: T,
    sum_up: T,
    sum_down: T,
    out: Option<T>,
    alarm: i8,
}

impl<T, V> PageHinkley<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Page-Hinkley test with a chained `View`, see the type docs for the
    /// meaning of the parameters.
    ///
    /// # Panics
    /// If `delta` is not finite and non-negative, `lambda` is not finite and positive,
    /// or `alpha` is not in `(0, 1]`.
    pub fn new(
        view: V,
        direction: PageHinkleyDirection,
        min_instances: NonZeroUsize,
        delta: T,
        lambda: T,
        alpha: T,
    ) -> Self {
        assert!(delta.is_finite(), "delta must be finite");
        assert!(delta >= T::zero(), "delta must be >= 0");
        assert!(lambda.is_finite(), "lambda must be finite");
        assert!(lambda > T::zero(), "lambda must be > 0");
        assert!(alpha > T::zero(), "alpha must be > 0");
        assert!(alpha <= T::one(), "alpha must be <= 1");
        Self {
            view,
            direction,
            min_instances,
            delta,
            lambda,
            alpha,
            count: 0,
            mean: T::zero(),
            sum_up: T::zero(),
            sum_down: T::zero(),
            out: None,
            alarm: 0,
        }
    }

    /// The magnitude of tolerated changes.
    #[inline]
    pub fn delta(&self) -> T {
        self.delta
    }

    /// The detection threshold.
    #[inline]
    pub fn lambda(&self) -> T {
        self.lambda
    }

    /// The forgetting factor of the statistics.
    #[inline]
    pub fn alpha(&self) -> T {
        self.alpha
    }

    /// The running mean of the values since the last reset (zero right after a reset).
    #[inline]
    pub fn mean(&self) -> T {
        self.mean
    }

    /// The alarm of the latest update: `1` for an increase, `-1` for a decrease and
    /// `0` for no alarm.
    #[inline]
    pub fn alarm(&self) -> i8 {
        self.alarm
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = T::zero();
        self.sum_up = T::zero();
        self.sum_down = T::zero();
    }
}

impl<T, V> View<T> for PageHinkley<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.count += 1;
        let n = T::from(self.count).expect("can convert");
        self.mean = self.mean + (val - self.mean) / n;
        self.sum_up = (self.alpha * self.sum_up + (val - self.mean - self.delta)).max(T::zero());
        self.sum_down =
            (self.alpha * self.sum_down + (self.mean - val - self.delta)).max(T::zero());
        debug_assert!(self.sum_up >= T::zero(), "U_t is non-negative");
        debug_assert!(self.sum_down >= T::zero(), "D_t is non-negative");

        let (up, down) = match self.direction {
            PageHinkleyDirection::Increase => (self.sum_up, T::zero()),
            PageHinkleyDirection::Decrease => (T::zero(), self.sum_down),
            PageHinkleyDirection::Both => (self.sum_up, self.sum_down),
        };
        let out = if up >= down && self.direction != PageHinkleyDirection::Decrease {
            up
        } else {
            -down
        };
        debug_assert!(out.is_finite(), "value must be finite");
        if self.out.is_some() || self.count >= self.min_instances.get() {
            self.out = Some(out);
        }

        self.alarm = if self.count < self.min_instances.get() {
            0
        } else if out > self.lambda {
            1
        } else if -out > self.lambda {
            -1
        } else {
            0
        };
        if self.alarm != 0 {
            self.reset();
        }
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out
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

    /// Standard normal sample via the Box-Muller transform on romu uniforms.
    fn gaussian(rng: &romu::Rng) -> f64 {
        let u1 = 1.0 - rng.f64(); // in (0, 1], so `ln` is finite
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn ph(
        direction: PageHinkleyDirection,
        min_instances: usize,
        delta: f64,
        lambda: f64,
        alpha: f64,
    ) -> PageHinkley<f64, Echo<f64>> {
        PageHinkley::new(
            Echo::new(),
            direction,
            NonZeroUsize::new(min_instances).unwrap(),
            delta,
            lambda,
            alpha,
        )
    }

    #[test]
    fn page_hinkley_plot() {
        let mut v = ph(PageHinkleyDirection::Both, 30, 0.5, 50.0, 1.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/page_hinkley.png").unwrap();
    }

    #[test]
    fn page_hinkley_warmup() {
        let mut v = ph(PageHinkleyDirection::Both, 30, 0.005, 50.0, 1.0);
        assert_eq!(v.min_instances().get(), 30);
        assert_eq!(v.direction(), PageHinkleyDirection::Both);
        for _ in 0..29 {
            v.update(1.0);
            assert_eq!(v.last(), None);
        }
        v.update(1.0);
        assert_eq!(v.last(), Some(0.0));
        assert_eq!(v.mean(), 1.0);
    }

    #[test]
    fn page_hinkley_hand_computed() {
        let mut v = ph(PageHinkleyDirection::Increase, 1, 0.1, 100.0, 1.0);
        assert_eq!(v.delta(), 0.1);
        assert_eq!(v.lambda(), 100.0);
        assert_eq!(v.alpha(), 1.0);
        v.update(1.0); // mean 1, U = max(0, 1 - 1 - 0.1) = 0
        assert_eq!(v.last(), Some(0.0));
        v.update(3.0); // mean 2, U = 3 - 2 - 0.1 = 0.9
        assert!((v.last().unwrap() - 0.9).abs() < 1e-12);
        v.update(5.0); // mean 3, U = 0.9 + 5 - 3 - 0.1 = 2.8
        assert!((v.last().unwrap() - 2.8).abs() < 1e-12);
        assert_eq!(v.alarm(), 0);

        // Decrease reports `-D`: mean 2, D = 2 - 1 - 0.1 = 0.9 after the second value.
        let mut v = ph(PageHinkleyDirection::Decrease, 1, 0.1, 100.0, 1.0);
        v.update(3.0);
        v.update(1.0);
        assert!((v.last().unwrap() + 0.9).abs() < 1e-12);
    }

    #[test]
    fn page_hinkley_forgetting() {
        // Four zeros and an outlier of 10: mean 2, U = 8; then a zero: mean 10 / 6,
        // U = alpha * 8 - 10 / 6.
        for (alpha, expected) in [(1.0, 8.0 - 10.0 / 6.0), (0.5, 4.0 - 10.0 / 6.0)] {
            let mut v = ph(PageHinkleyDirection::Increase, 1, 0.0, 100.0, alpha);
            for x in [0.0, 0.0, 0.0, 0.0, 10.0] {
                v.update(x);
            }
            assert!((v.last().unwrap() - 8.0).abs() < 1e-12);
            v.update(0.0);
            assert!((v.last().unwrap() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn page_hinkley_no_alarm_on_stationary_noise() {
        // `delta = 0.5, lambda = 15` on unit variance noise behaves like a CUSUM with
        // k = 0.5, h = 15, whose in-control average run length is far above 10^6.
        let rng = romu::Rng::from_seed_with_64bit(91);
        let mut v = ph(PageHinkleyDirection::Both, 30, 0.5, 15.0, 1.0);
        for _ in 0..20_000 {
            v.update(gaussian(&rng));
            assert_eq!(v.alarm(), 0);
        }
        assert!(v.mean().abs() < 0.05, "{}", v.mean());
    }

    #[test]
    fn page_hinkley_detects_mean_shift() {
        use PageHinkleyDirection::*;
        for (seed, direction, shift, expected) in [
            (92, Increase, 2.0, 1),
            (93, Both, 2.0, 1),
            (94, Decrease, -2.0, -1),
            (95, Both, -2.0, -1),
        ] {
            let rng = romu::Rng::from_seed_with_64bit(seed);
            let mut v = ph(direction, 30, 0.5, 15.0, 1.0);
            for _ in 0..2_000 {
                v.update(gaussian(&rng));
                assert_eq!(v.alarm(), 0);
            }
            // The statistic grows by about |shift| - delta = 1.5 per sample.
            let delay = (1..=30)
                .find(|_| {
                    v.update(shift + gaussian(&rng));
                    v.alarm() != 0
                })
                .expect("the shift must be detected");
            assert_eq!(v.alarm(), expected, "{direction:?}");
            assert!(delay <= 20, "delay {delay} for {direction:?}");
            // Reset after the alarm.
            assert_eq!(v.mean(), 0.0);
        }
    }

    #[test]
    fn page_hinkley_ignores_other_direction() {
        let rng = romu::Rng::from_seed_with_64bit(96);
        let mut up = ph(PageHinkleyDirection::Increase, 30, 0.5, 15.0, 1.0);
        let mut down = ph(PageHinkleyDirection::Decrease, 30, 0.5, 15.0, 1.0);
        for i in 0..2_500 {
            let shift = if i < 2_000 { 0.0 } else { 3.0 };
            let x = -shift + gaussian(&rng);
            up.update(x);
            down.update(-x);
            assert_eq!(up.alarm(), 0);
            assert_eq!(down.alarm(), 0);
            assert!(up.last().unwrap_or(0.0) >= 0.0);
            assert!(down.last().unwrap_or(0.0) <= 0.0);
        }
    }

    #[test]
    #[should_panic(expected = "delta must be >= 0")]
    fn page_hinkley_negative_delta() {
        let _ = ph(PageHinkleyDirection::Both, 30, -0.1, 50.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 0")]
    fn page_hinkley_zero_lambda() {
        let _ = ph(PageHinkleyDirection::Both, 30, 0.1, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be > 0")]
    fn page_hinkley_zero_alpha() {
        let _ = ph(PageHinkleyDirection::Both, 30, 0.1, 50.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be <= 1")]
    fn page_hinkley_alpha_above_one() {
        let _ = ph(PageHinkleyDirection::Both, 30, 0.1, 50.0, 1.0 + 1e-9);
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Two-sided tabular CUSUM change detector.

use num::Float;

use crate::View;

/// Two-sided tabular CUSUM (cumulative sum) change detector for shifts of the mean.
///
/// Reference: E. S. Page (1954), "Continuous Inspection Schemes", Biometrika 41;
/// tabular form as in D. C. Montgomery, "Introduction to Statistical Quality Control".
///
/// Each input `x` is standardized with the in-control (target) mean and standard
/// deviation, `z = (x - target_mean) / target_std`, and two one-sided statistics are
/// accumulated:
/// `S+ = max(0, S+ + z - k)` detects increases and `S- = max(0, S- - z - k)` detects decreases.
///
/// Parameters, both in units of `target_std`:
/// - `k`: the reference value (slack). Deviations smaller than `k` drain the statistics.
///   `k = delta / 2` is optimal for detecting a shift of `delta` standard deviations.
/// - `h`: the decision threshold. An alarm is raised when `S+ > h` or `S- > h`.
///   `k = 0.5, h = 5` gives an in-control average run length of about 465 samples
///   (two-sided) and detects a one standard deviation shift after about 10 samples.
///
/// The target mean and standard deviation are given as parameters instead of being
/// estimated from a warm-up period, so a change during warm-up can not corrupt the
/// reference and the detector is ready after the first value. Chain a z-score `View`
/// and use `target_mean = 0, target_std = 1` for a self-normalizing setup.
///
/// Output: the signed statistic of the latest update, `S+` if `S+ >= S-`, else `-S-`.
/// After an alarm, `last()` still reports the statistic that crossed `h`, and both
/// statistics are reset to zero, so the next update starts a fresh run.
/// `alarm()` reports the alarm of the latest update.
///
/// Warm-up: `None` until the chained `View` yields its first value.
/// Complexity: O(1) time per update and O(1) memory.
#[derive(Debug, Clone)]
pub struct Cusum<T, V> {
    view: V,
    target_mean: T,
    target_std: T,
    k: T,
    h: T,
    s_plus: T,
    s_minus: T,
    out: Option<T>,
    alarm: i8,
}

impl<T, V> Cusum<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new two-sided CUSUM with a chained `View`, the in-control `target_mean`
    /// and `target_std`, the reference value `k` and the decision threshold `h`
    /// (both in units of `target_std`).
    ///
    /// # Panics
    /// If `target_mean` is not finite, `target_std` is not finite and positive,
    /// `k` is not finite and non-negative or `h` is not finite and positive.
    pub fn new(view: V, target_mean: T, target_std: T, k: T, h: T) -> Self {
        assert!(target_mean.is_finite(), "target_mean must be finite");
        assert!(target_std.is_finite(), "target_std must be finite");
        assert!(target_std > T::zero(), "target_std must be > 0");
        assert!(k.is_finite(), "k must be finite");
        assert!(k >= T::zero(), "k must be >= 0");
        assert!(h.is_finite(), "h must be finite");
        assert!(h > T::zero(), "h must be > 0");
        Self {
            view,
            target_mean,
            target_std,
            k,
            h,
            s_plus: T::zero(),
            s_minus: T::zero(),
            out: None,
            alarm: 0,
        }
    }

    /// The in-control mean.
    #[inline]
    pub fn target_mean(&self) -> T {
        self.target_mean
    }

    /// The in-control standard deviation.
    #[inline]
    pub fn target_std(&self) -> T {
        self.target_std
    }

    /// The reference value (slack) in units of `target_std`.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }

    /// The decision threshold in units of `target_std`.
    #[inline]
    pub fn h(&self) -> T {
        self.h
    }

    /// The upper statistic `S+` after the latest update (zero after an alarm reset).
    #[inline]
    pub fn s_plus(&self) -> T {
        self.s_plus
    }

    /// The lower statistic `S-` after the latest update (zero after an alarm reset).
    #[inline]
    pub fn s_minus(&self) -> T {
        self.s_minus
    }

    /// The alarm of the latest update: `1` for an upward shift (`S+ > h`),
    /// `-1` for a downward shift (`S- > h`) and `0` for no alarm.
    #[inline]
    pub fn alarm(&self) -> i8 {
        self.alarm
    }
}

impl<T, V> View<T> for Cusum<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let z = (val - self.target_mean) / self.target_std;
        self.s_plus = (self.s_plus + z - self.k).max(T::zero());
        self.s_minus = (self.s_minus - z - self.k).max(T::zero());
        debug_assert!(self.s_plus >= T::zero(), "S+ is non-negative");
        debug_assert!(self.s_minus >= T::zero(), "S- is non-negative");

        let out = if self.s_plus >= self.s_minus {
            self.s_plus
        } else {
            -self.s_minus
        };
        debug_assert!(out.is_finite(), "value must be finite");
        self.out = Some(out);

        // At most one statistic can cross `h` per update: both are <= h before it
        // (an alarm resets them) and `z` raises at most one of them.
        debug_assert!(
            !(self.s_plus > self.h && self.s_minus > self.h),
            "only one side can alarm"
        );
        self.alarm = if self.s_plus > self.h {
            1
        } else if self.s_minus > self.h {
            -1
        } else {
            0
        };
        if self.alarm != 0 {
            self.s_plus = T::zero();
            self.s_minus = T::zero();
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
        sliding_windows::Sma,
        test_data::TEST_DATA,
    };

    /// Standard normal sample via the Box-Muller transform on romu uniforms.
    fn gaussian(rng: &romu::Rng) -> f64 {
        let u1 = 1.0 - rng.f64(); // in (0, 1], so `ln` is finite
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn cusum(k: f64, h: f64) -> Cusum<f64, Echo<f64>> {
        Cusum::new(Echo::new(), 0.0, 1.0, k, h)
    }

    #[test]
    fn cusum_plot() {
        let mut c = Cusum::new(Echo::new(), 100.0, 5.0, 0.5, 5.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            c.update(*v);
            out.push(c.last().unwrap());
        }
        plot_values(out, "img/cusum.png").unwrap();
    }

    #[test]
    fn cusum_warmup() {
        let mut c = Cusum::new(
            Sma::new(Echo::new(), std::num::NonZeroUsize::new(3).unwrap()),
            0.0,
            1.0,
            0.5,
            5.0,
        );
        assert_eq!(c.last(), None);
        c.update(1.0);
        c.update(1.0);
        assert_eq!(c.last(), None);
        c.update(1.0);
        assert_eq!(c.last(), Some(0.5));
        assert_eq!(c.alarm(), 0);
    }

    #[test]
    fn cusum_hand_computed() {
        let mut c = Cusum::new(Echo::new(), 10.0, 2.0, 0.5, 2.0);
        assert_eq!(c.target_mean(), 10.0);
        assert_eq!(c.target_std(), 2.0);
        assert_eq!(c.k(), 0.5);
        assert_eq!(c.h(), 2.0);
        // z = 1.0 -> S+ = 0.5, S- = 0
        c.update(12.0);
        assert_eq!(c.last(), Some(0.5));
        // z = 1.5 -> S+ = 1.5
        c.update(13.0);
        assert_eq!(c.last(), Some(1.5));
        assert_eq!(c.alarm(), 0);
        // z = 2.0 -> S+ = 3.0 > h -> alarm and reset
        c.update(14.0);
        assert_eq!(c.last(), Some(3.0));
        assert_eq!(c.alarm(), 1);
        assert_eq!(c.s_plus(), 0.0);
        assert_eq!(c.s_minus(), 0.0);
        // z = -2.0 -> S- = 1.5
        c.update(6.0);
        assert_eq!(c.last(), Some(-1.5));
        assert_eq!(c.alarm(), 0);
        // z = -1.5 -> S- = 2.5 > h -> alarm down
        c.update(7.0);
        assert_eq!(c.last(), Some(-2.5));
        assert_eq!(c.alarm(), -1);
        // z = 0 -> both stay at zero
        c.update(10.0);
        assert_eq!(c.last(), Some(0.0));
        assert_eq!(c.alarm(), 0);
    }

    #[test]
    fn cusum_no_alarm_on_stationary_noise() {
        // k = 0.5, h = 10: in-control average run length of roughly 7 * 10^4 (two-sided,
        // Siegmund approximation), so an alarm in 2k samples has a probability of ~3%
        // and this fixed seed has none.
        let rng = romu::Rng::from_seed_with_64bit(81);
        let mut c = cusum(0.5, 10.0);
        for _ in 0..2_000 {
            c.update(gaussian(&rng));
            assert_eq!(c.alarm(), 0);
            assert!(c.last().unwrap().abs() <= 10.0);
        }
    }

    #[test]
    fn cusum_false_alarm_rate() {
        // k = 0.5, h = 5: in-control average run length about 465 (two-sided),
        // so about 215 false alarms are expected in 100k samples.
        let rng = romu::Rng::from_seed_with_64bit(82);
        let mut c = cusum(0.5, 5.0);
        let n = 100_000;
        let mut alarms = 0;
        for _ in 0..n {
            c.update(gaussian(&rng));
            alarms += usize::from(c.alarm() != 0);
        }
        assert!(alarms > 100, "{alarms}");
        assert!(alarms < 400, "{alarms}");
    }

    #[test]
    fn cusum_detects_mean_shift() {
        for (seed, shift, expected) in [(83, 2.0, 1), (84, -2.0, -1), (85, 1.0, 1)] {
            let rng = romu::Rng::from_seed_with_64bit(seed);
            let mut c = cusum(0.5, 10.0);
            for _ in 0..1_000 {
                c.update(gaussian(&rng));
                assert_eq!(c.alarm(), 0);
            }
            let delay = (1..=60)
                .find(|_| {
                    c.update(shift + gaussian(&rng));
                    c.alarm() != 0
                })
                .expect("the shift must be detected");
            assert_eq!(c.alarm(), expected);
            // Expected delay (out-of-control run length): ~7 for 2 sigma, ~20 for 1 sigma.
            let max_delay = if shift.abs() >= 2.0 { 15 } else { 60 };
            assert!(delay <= max_delay, "delay {delay} for shift {shift}");
        }
    }

    #[test]
    #[should_panic(expected = "target_std must be > 0")]
    fn cusum_zero_std() {
        let _ = Cusum::new(Echo::<f64>::new(), 0.0, 0.0, 0.5, 5.0);
    }

    #[test]
    #[should_panic(expected = "k must be >= 0")]
    fn cusum_negative_k() {
        let _ = cusum(-0.1, 5.0);
    }

    #[test]
    #[should_panic(expected = "h must be > 0")]
    fn cusum_zero_h() {
        let _ = cusum(0.5, 0.0);
    }

    #[test]
    #[should_panic(expected = "target_mean must be finite")]
    fn cusum_nan_mean() {
        let _ = Cusum::new(Echo::<f64>::new(), f64::NAN, 1.0, 0.5, 5.0);
    }
}

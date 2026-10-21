// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::num::NonZeroUsize;

use num::Float;

use super::{
    HighPass,
    HighPassPoles,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// John Ehlers Decycler: the input minus its high-pass filtered cycles.
///
/// Removing the cycles shorter than `cutoff_period` leaves the trend
/// component with very little lag, because the high-pass filter has almost
/// no phase shift at the long periods that pass through.
///
/// ```text
/// Decycle = x - HP(x, cutoff_period)
/// ```
///
/// References (they differ in the high-pass filter, so both are offered via `poles`):
/// - [`HighPassPoles::One`]: John Ehlers, "Cycle Analytics for Traders" (2013),
///   chapter 4, written there as the equivalent
///   `Decycle = a1/2 (x + x[1]) + (1 - a1) Decycle[1]`.
/// - [`HighPassPoles::Two`]: John Ehlers, "Decyclers", Technical Analysis of
///   Stocks & Commodities, September 2015 ("Simple Decycler"), which uses the
///   two-pole high-pass with the `.707` factor. The two-pole version has zero
///   steady state lag on a linear ramp.
///
/// The high-pass is the crate's [`HighPass`], so it starts at rest
/// (the first output equals the first input) and `last()` is `None` for the
/// first `cutoff_period - 1` values.
///
/// Output range: unbounded, follows the input. Constant input yields the constant.
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone)]
pub struct Decycler<T, V> {
    view: V,
    high_pass: HighPass<T, Echo<T>>,
    /// The most recent value from the inner view.
    val: T,
}

impl<T, V> Decycler<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Decycler with a chained `View`.
    ///
    /// Ehlers uses `cutoff_period = 125` in the TASC article.
    ///
    /// # Panics
    /// If `cutoff_period` is too small for a stable high-pass filter,
    /// see [`HighPass::new`].
    pub fn new(view: V, cutoff_period: NonZeroUsize, poles: HighPassPoles) -> Self {
        Self {
            view,
            high_pass: HighPass::new(Echo::new(), cutoff_period, poles),
            val: T::zero(),
        }
    }

    /// Cycles with periods shorter than this are removed.
    #[inline]
    pub fn cutoff_period(&self) -> NonZeroUsize {
        self.high_pass.cutoff_period()
    }

    /// The number of poles of the high-pass filter.
    #[inline]
    pub fn poles(&self) -> HighPassPoles {
        self.high_pass.poles()
    }
}

impl<T, V> View<T> for Decycler<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.high_pass.update(val);
        self.val = val;
    }

    #[inline]
    fn last(&self) -> Option<T> {
        let out = self.val - self.high_pass.last()?;
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    const POLES: [HighPassPoles; 2] = [HighPassPoles::One, HighPassPoles::Two];

    fn dc(period: usize, poles: HighPassPoles) -> Decycler<f64, Echo<f64>> {
        Decycler::new(Echo::new(), NonZeroUsize::new(period).unwrap(), poles)
    }

    /// Steady state amplitude of the output for a unit sine input.
    fn sine_gain(period: usize, poles: HighPassPoles, sine_period: f64) -> f64 {
        let mut f = dc(period, poles);
        let n = (40.0 * sine_period.max(period as f64)) as usize;
        let mut max_abs: f64 = 0.0;
        for i in 0..n {
            f.update((2.0 * PI * i as f64 / sine_period).sin());
            if i > n - 4 * sine_period as usize {
                max_abs = max_abs.max(f.last().unwrap().abs());
            }
        }
        max_abs
    }

    #[test]
    fn decycler_plot() {
        let mut f = dc(60, HighPassPoles::Two);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/decycler.png").unwrap();
    }

    #[test]
    fn decycler_warmup() {
        for poles in POLES {
            let mut f = dc(10, poles);
            assert_eq!(f.cutoff_period().get(), 10);
            assert_eq!(f.poles(), poles);
            for i in 0..9 {
                f.update(i as f64);
                assert_eq!(f.last(), None);
            }
            f.update(9.0);
            assert!(f.last().is_some());
        }
    }

    #[test]
    fn decycler_one_pole_matches_cycle_analytics() {
        let p = 20.0;
        let angle = 2.0 * PI / p;
        let a1 = (angle.cos() + angle.sin() - 1.0) / angle.cos();
        let mut f = dc(20, HighPassPoles::One);
        let (mut x1, mut d) = (TEST_DATA[0], TEST_DATA[0]);
        for (i, v) in TEST_DATA.iter().enumerate() {
            if i > 0 {
                d = a1 / 2.0 * (v + x1) + (1.0 - a1) * d;
            }
            x1 = *v;
            f.update(*v);
            if let Some(got) = f.last() {
                assert!((got - d).abs() < 1e-9, "bar {i}: {got} vs {d}");
            }
        }
    }

    #[test]
    fn decycler_two_pole_matches_tasc() {
        let p = 20.0;
        let angle = 0.707 * 2.0 * PI / p;
        let a1 = (angle.cos() + angle.sin() - 1.0) / angle.cos();
        let mut f = dc(20, HighPassPoles::Two);
        let x0 = TEST_DATA[0];
        let (mut x1, mut x2, mut h1, mut h2) = (x0, x0, 0.0, 0.0);
        for (i, v) in TEST_DATA.iter().enumerate() {
            let hp = (1.0 - a1 / 2.0).powi(2) * (v - 2.0 * x1 + x2) + 2.0 * (1.0 - a1) * h1
                - (1.0 - a1).powi(2) * h2;
            (x2, x1, h2, h1) = (x1, *v, h1, hp);
            f.update(*v);
            if let Some(got) = f.last() {
                assert!((got - (v - hp)).abs() < 1e-9, "bar {i}");
            }
        }
    }

    #[test]
    fn decycler_constant_input() {
        for poles in POLES {
            let mut f = dc(20, poles);
            for _ in 0..100 {
                f.update(42.0);
            }
            assert_eq!(f.last(), Some(42.0));
        }
    }

    #[test]
    fn decycler_ramp_lag() {
        let mut one = dc(20, HighPassPoles::One);
        let mut two = dc(20, HighPassPoles::Two);
        let mut x = 0.0;
        for i in 0..2_000 {
            x = i as f64;
            one.update(x);
            two.update(x);
        }
        // A one-pole high-pass passes part of the ramp's slope, the two-pole one none of it.
        assert!(x - one.last().unwrap() > 1.0);
        assert!((x - two.last().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn decycler_frequency_response() {
        for poles in POLES {
            let slow = sine_gain(20, poles, 400.0);
            assert!((0.95..1.05).contains(&slow), "{poles:?}: slow gain {slow}");
            let fast = sine_gain(20, poles, 4.0);
            assert!(fast < 0.3, "{poles:?}: fast gain {fast}");
        }
    }

    #[test]
    #[should_panic(expected = "cutoff_period must be >= 5")]
    fn decycler_one_pole_period_too_small() {
        let _ = dc(4, HighPassPoles::One);
    }

    #[test]
    #[should_panic(expected = "cutoff_period must be >= 3")]
    fn decycler_two_pole_period_too_small() {
        let _ = dc(2, HighPassPoles::Two);
    }
}

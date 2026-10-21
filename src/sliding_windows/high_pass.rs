// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Number of poles of a [`HighPass`] filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighPassPoles {
    /// First order filter: 6 dB per octave attenuation below the cutoff period.
    One,
    /// Second order filter: 12 dB per octave attenuation below the cutoff period.
    Two,
}

/// John Ehlers high-pass filter with one or two poles.
///
/// Removes spectral components with periods longer than `cutoff_period`
/// (trend) and passes shorter periods with roughly unit gain.
///
/// Reference: John Ehlers, "Cycle Analytics for Traders" (2013), `HighPass`
/// (one pole) and "Predictive Indicators for Effective Trading Strategies",
/// <https://www.mesasoftware.com/papers/PredictiveIndicatorsForEffectiveTrading%20Strategies.pdf>
/// (two poles, as used in the [`RoofingFilter`](super::RoofingFilter)):
///
/// - One pole: `a1 = (cos(2pi/P) + sin(2pi/P) - 1) / cos(2pi/P)`,
///   `HP = (1 - a1/2) (x - x[1]) + (1 - a1) HP[1]`
/// - Two poles: `a1 = (cos(.707*2pi/P) + sin(.707*2pi/P) - 1) / cos(.707*2pi/P)`,
///   `HP = (1 - a1/2)^2 (x - 2x[1] + x[2]) + 2(1 - a1) HP[1] - (1 - a1)^2 HP[2]`
///
/// The input history is seeded with the first value, so the filter starts at
/// rest (`HP = 0`) instead of seeing a step from zero. This plays the role of
/// Ehlers' `If CurrentBar < 3 Then HP = 0` initialization.
/// `last()` is `None` for the first `cutoff_period - 1` values (the filter's
/// time scale, as in [`SuperSmoother`](super::SuperSmoother)).
///
/// Output range: unbounded, oscillates around 0. Constant input yields 0.
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct HighPass<T, V> {
    view: V,
    /// Periods longer than this are attenuated.
    #[getset(get_copy = "pub")]
    cutoff_period: NonZeroUsize,
    /// The number of poles of the filter.
    #[getset(get_copy = "pub")]
    poles: HighPassPoles,
    /// Gain applied to the differenced input.
    c_in: T,
    /// Coefficient of `x[1]` in the differenced input, `-1` or `-2`.
    d_1: T,
    /// Coefficient of `x[2]` in the differenced input, `0` or `1`.
    d_2: T,
    /// Feedback coefficient of `HP[1]`.
    c_hp_1: T,
    /// Feedback coefficient of `HP[2]`, `0` for one pole.
    c_hp_2: T,
    /// Number of values received from the inner view.
    n: usize,
    val_1: T,
    val_2: T,
    hp: T,
    hp_1: T,
}

impl<T, V> HighPass<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new high-pass filter with a chained `View`.
    ///
    /// # Panics
    /// The coefficient `a1` is only in `(0, 1)` (a stable filter) if the angle
    /// `2pi/P` (one pole) or `.707*2pi/P` (two poles) is below `pi/2`.
    /// Hence `cutoff_period` must be at least 5 for one pole and at least 3 for two poles.
    pub fn new(view: V, cutoff_period: NonZeroUsize, poles: HighPassPoles) -> Self {
        let min_period = match poles {
            HighPassPoles::One => 5,
            HighPassPoles::Two => 3,
        };
        assert!(
            cutoff_period.get() >= min_period,
            "cutoff_period must be >= {min_period} for {poles:?} pole(s)"
        );
        let two = T::from(2.0).expect("can convert");
        let pole_factor = match poles {
            HighPassPoles::One => T::one(),
            HighPassPoles::Two => T::from(0.707).expect("can convert"),
        };
        let angle = pole_factor * two * T::from(PI).expect("can convert")
            / T::from(cutoff_period.get()).expect("can convert");
        let a1 = (angle.cos() + angle.sin() - T::one()) / angle.cos();
        assert!(a1 > T::zero(), "a1 must be positive for a stable filter");
        assert!(a1 < T::one(), "a1 must be below 1 for a stable filter");

        let k = T::one() - a1 / two;
        let (c_in, d_1, d_2, c_hp_1, c_hp_2) = match poles {
            HighPassPoles::One => (k, -T::one(), T::zero(), T::one() - a1, T::zero()),
            HighPassPoles::Two => (
                k * k,
                -two,
                T::one(),
                two * (T::one() - a1),
                -(T::one() - a1) * (T::one() - a1),
            ),
        };

        Self {
            view,
            cutoff_period,
            poles,
            c_in,
            d_1,
            d_2,
            c_hp_1,
            c_hp_2,
            n: 0,
            val_1: T::zero(),
            val_2: T::zero(),
            hp: T::zero(),
            hp_1: T::zero(),
        }
    }
}

impl<T, V> View<T> for HighPass<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.n == 0 {
            // Start at rest, see struct docs.
            self.val_1 = val;
            self.val_2 = val;
        }
        let hp = self.c_in * (val + self.d_1 * self.val_1 + self.d_2 * self.val_2)
            + self.c_hp_1 * self.hp
            + self.c_hp_2 * self.hp_1;
        debug_assert!(hp.is_finite(), "filter output must be finite");
        self.hp_1 = self.hp;
        self.hp = hp;
        self.val_2 = self.val_1;
        self.val_1 = val;
        self.n = self.n.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < self.cutoff_period.get() {
            return None;
        }
        debug_assert!(self.hp.is_finite(), "value must be finite");
        Some(self.hp)
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

    const POLES: [HighPassPoles; 2] = [HighPassPoles::One, HighPassPoles::Two];

    fn hp(period: usize, poles: HighPassPoles) -> HighPass<f64, Echo<f64>> {
        HighPass::new(Echo::new(), NonZeroUsize::new(period).unwrap(), poles)
    }

    /// Steady state amplitude of the filter output for a unit sine input.
    fn sine_gain(cutoff: usize, poles: HighPassPoles, sine_period: f64) -> f64 {
        let mut f = hp(cutoff, poles);
        let n = (40.0 * sine_period.max(cutoff as f64)) as usize;
        let mut max_abs: f64 = 0.0;
        for i in 0..n {
            f.update((2.0 * PI * i as f64 / sine_period).sin());
            // Only measure the last few cycles, after the transient has decayed.
            if i > n - 4 * sine_period as usize {
                max_abs = max_abs.max(f.last().unwrap().abs());
            }
        }
        max_abs
    }

    #[test]
    fn high_pass_plot() {
        let mut f = hp(48, HighPassPoles::Two);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/high_pass.png").unwrap();
    }

    #[test]
    fn high_pass_warmup() {
        for poles in POLES {
            let mut f = hp(10, poles);
            assert_eq!(f.poles(), poles);
            assert_eq!(f.cutoff_period().get(), 10);
            for i in 0..9 {
                f.update(i as f64);
                assert_eq!(f.last(), None);
            }
            f.update(9.0);
            assert!(f.last().is_some());
        }
    }

    #[test]
    fn high_pass_hand_computed_one_pole() {
        let mut f = hp(8, HighPassPoles::One);
        let angle = 2.0 * PI / 8.0;
        let a1 = (angle.cos() + angle.sin() - 1.0) / angle.cos();
        let vals = [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 1.0, 0.0];
        let (mut x1, mut h) = (vals[0], 0.0);
        for v in vals {
            h = (1.0 - a1 / 2.0) * (v - x1) + (1.0 - a1) * h;
            x1 = v;
            f.update(v);
        }
        assert!((f.last().unwrap() - h).abs() < 1e-12);
    }

    #[test]
    fn high_pass_matches_roofing_filter_high_pass() {
        // Same formula as the high pass stage of RoofingFilter, written out explicitly.
        let mut f = hp(48, HighPassPoles::Two);
        let a = ((4.4422_f64 / 48.0).cos() + (4.4422_f64 / 48.0).sin() - 1.0)
            / (4.4422_f64 / 48.0).cos();
        let (mut x1, mut x2, mut h1, mut h2) = (TEST_DATA[0], TEST_DATA[0], 0.0, 0.0);
        for v in TEST_DATA.iter() {
            let h = (1.0 - a / 2.0).powi(2) * (v - 2.0 * x1 + x2) + 2.0 * (1.0 - a) * h1
                - (1.0 - a).powi(2) * h2;
            (x2, x1, h2, h1) = (x1, *v, h1, h);
            f.update(*v);
            // 0.707 vs 4.4422 / (2 pi) = 0.70700 differ slightly.
            assert!((f.last().unwrap_or(h) - h).abs() < 1e-3 * v.abs().max(1.0));
        }
    }

    #[test]
    fn high_pass_constant_input_is_zero() {
        for poles in POLES {
            let mut f = hp(20, poles);
            for _ in 0..100 {
                f.update(42.0);
            }
            assert_eq!(f.last(), Some(0.0));
        }
    }

    #[test]
    fn high_pass_step_decays_to_zero() {
        for poles in POLES {
            let mut f = hp(20, poles);
            for i in 0..1_000 {
                f.update(if i < 50 { 1.0 } else { 5.0 });
            }
            assert!(f.last().unwrap().abs() < 1e-9, "{poles:?}");
        }
    }

    #[test]
    fn high_pass_frequency_response() {
        for poles in POLES {
            let fast = sine_gain(48, poles, 6.0);
            assert!((0.9..1.1).contains(&fast), "{poles:?}: fast gain {fast}");
            let slow = sine_gain(48, poles, 480.0);
            let max_slow = match poles {
                HighPassPoles::One => 0.2,
                HighPassPoles::Two => 0.05,
            };
            assert!(slow < max_slow, "{poles:?}: slow gain {slow}");
        }
    }

    #[test]
    fn high_pass_min_period_valid() {
        let mut f = hp(5, HighPassPoles::One);
        let mut g = hp(3, HighPassPoles::Two);
        let rng = romu::Rng::from_seed_with_64bit(1);
        for _ in 0..10_000 {
            let v = rng.f64();
            f.update(v);
            g.update(v);
        }
        assert!(f.last().unwrap().is_finite());
        assert!(g.last().unwrap().is_finite());
        assert!(f.last().unwrap().abs() < 10.0);
        assert!(g.last().unwrap().abs() < 10.0);
    }

    #[test]
    #[should_panic(expected = "cutoff_period must be >= 5")]
    fn high_pass_one_pole_period_too_small() {
        let _ = hp(4, HighPassPoles::One);
    }

    #[test]
    #[should_panic(expected = "cutoff_period must be >= 3")]
    fn high_pass_two_pole_period_too_small() {
        let _ = hp(2, HighPassPoles::Two);
    }
}

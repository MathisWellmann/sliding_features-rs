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

/// Two-pole high-pass with a fractional cutoff period, as in the TASC code.
///
/// The crate's [`HighPass`](super::HighPass) only accepts integer periods and
/// hides its value during warm-up, both of which the oscillator needs.
#[derive(Debug, Clone)]
struct HighPass2Core<T> {
    /// `(1 - a1/2)^2`.
    c_in: T,
    /// `2 (1 - a1)`.
    c_hp_1: T,
    /// `-(1 - a1)^2`.
    c_hp_2: T,
    /// Whether the input history has been seeded.
    started: bool,
    x_1: T,
    x_2: T,
    hp_1: T,
    hp_2: T,
}

impl<T: Float> HighPass2Core<T> {
    fn new(cutoff_period: T) -> Self {
        let angle = T::from(0.707 * 2.0 * PI).expect("can convert") / cutoff_period;
        let a1 = (angle.cos() + angle.sin() - T::one()) / angle.cos();
        assert!(a1 > T::zero(), "a1 must be positive for a stable filter");
        assert!(a1 < T::one(), "a1 must be below 1 for a stable filter");
        let k = T::one() - a1 / T::from(2.0).expect("can convert");
        let d = T::one() - a1;
        Self {
            c_in: k * k,
            c_hp_1: d + d,
            c_hp_2: -d * d,
            started: false,
            x_1: T::zero(),
            x_2: T::zero(),
            hp_1: T::zero(),
            hp_2: T::zero(),
        }
    }

    fn update(&mut self, x: T) -> T {
        if !self.started {
            // Start at rest instead of seeing a step from zero.
            self.x_1 = x;
            self.x_2 = x;
            self.started = true;
        }
        let two = T::from(2.0).expect("can convert");
        let hp = self.c_in * (x - two * self.x_1 + self.x_2)
            + self.c_hp_1 * self.hp_1
            + self.c_hp_2 * self.hp_2;
        debug_assert!(hp.is_finite(), "filter output must be finite");
        self.x_2 = self.x_1;
        self.x_1 = x;
        self.hp_2 = self.hp_1;
        self.hp_1 = hp;
        hp
    }
}

/// John Ehlers Decycler Oscillator.
///
/// A two-pole high-pass of the two-pole [`Decycler`](super::Decycler), i.e.
/// a band-pass keeping cycles with periods between about `hp_period / 2` and
/// `hp_period`, expressed in percent of the input:
///
/// ```text
/// Decycle    = x - HP2(x, hp_period)
/// DecycleOsc = HP2(Decycle, hp_period / 2)
/// out        = 100 K DecycleOsc / x
/// ```
///
/// with `HP2(x, P)`: `a1 = (cos(.707 2pi/P) + sin(.707 2pi/P) - 1) / cos(.707 2pi/P)`,
/// `HP = (1 - a1/2)^2 (x - 2x[1] + x[2]) + 2(1 - a1) HP[1] - (1 - a1)^2 HP[2]`.
/// Zero crossings mark trend reversals with very little lag. Ehlers' strategy
/// trades the crossings of a fast (`hp_period = 100`, `K = 1.2`) and a slow
/// (`hp_period = 125`, `K = 1`) oscillator.
///
/// Reference: John Ehlers, "Decyclers", Technical Analysis of Stocks &
/// Commodities, September 2015 (Decycler Oscillator code in the Traders' Tips).
/// "Cycle Analytics for Traders" (2013) has a different oscillator,
/// `HP2(x, P2) - HP2(x, P1)` without the price scaling; it is not implemented here.
///
/// Both high-pass filters start at rest (input history seeded with the first
/// value), replacing EasyLanguage's zero initialization.
/// `last()` is `None` for the first `hp_period - 1` values.
///
/// Output range: unbounded, oscillates around 0, in percent of the input.
/// Constant input and linear ramps yield 0.
/// Complexity: O(1) per update, O(1) memory.
///
/// # Panics
/// `update` panics if the input is not positive, as the output divides by it.
#[derive(Debug, Clone, CopyGetters)]
pub struct DecyclerOscillator<T, V> {
    view: V,
    /// Cutoff period of the decycler; the oscillator uses half of it.
    #[getset(get_copy = "pub")]
    hp_period: NonZeroUsize,
    k: T,
    decycle_hp: HighPass2Core<T>,
    osc_hp: HighPass2Core<T>,
    /// The most recent input.
    val: T,
    /// The most recent `DecycleOsc`, before scaling.
    osc: T,
    /// Number of values received from the inner view.
    n: usize,
}

impl<T, V> DecyclerOscillator<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Decycler Oscillator with a chained `View`.
    ///
    /// `k` scales the output, Ehlers uses 1 (slow) and 1.2 (fast).
    ///
    /// # Panics
    /// - If `hp_period < 6`, as the high-pass at `hp_period / 2` would be unstable.
    /// - If `k` is not positive and finite.
    pub fn new(view: V, hp_period: NonZeroUsize, k: T) -> Self {
        assert!(hp_period.get() >= 6, "hp_period must be >= 6");
        assert!(k.is_finite(), "k must be finite");
        assert!(k > T::zero(), "k must be > 0");
        let p = T::from(hp_period.get()).expect("can convert");
        let half = T::from(0.5).expect("can convert");
        Self {
            view,
            hp_period,
            k,
            decycle_hp: HighPass2Core::new(p),
            osc_hp: HighPass2Core::new(half * p),
            val: T::one(),
            osc: T::zero(),
            n: 0,
        }
    }

    /// The output scale factor `K`.
    // Not generated by `getset`, as its impl block lacks the `T: Copy` bound.
    #[inline]
    pub fn k(&self) -> T {
        self.k
    }
}

impl<T, V> View<T> for DecyclerOscillator<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        assert!(
            val > T::zero(),
            "input must be positive, the output divides by it"
        );

        let decycle = val - self.decycle_hp.update(val);
        self.osc = self.osc_hp.update(decycle);
        self.val = val;
        self.n = self.n.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < self.hp_period.get() {
            return None;
        }
        let hundred = T::from(100.0).expect("can convert");
        let out = hundred * self.k * self.osc / self.val;
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::{
            HighPass,
            HighPassPoles,
        },
        test_data::TEST_DATA,
    };

    fn osc(period: usize, k: f64) -> DecyclerOscillator<f64, Echo<f64>> {
        DecyclerOscillator::new(Echo::new(), NonZeroUsize::new(period).unwrap(), k)
    }

    /// Steady state amplitude of the unscaled oscillator for a unit sine around 100.
    fn sine_gain(period: usize, sine_period: f64) -> f64 {
        let mut f = osc(period, 1.0);
        let n = (40.0 * sine_period.max(period as f64)) as usize;
        let mut max_abs: f64 = 0.0;
        for i in 0..n {
            let x = 100.0 + (2.0 * PI * i as f64 / sine_period).sin();
            f.update(x);
            if i > n - 4 * sine_period as usize {
                max_abs = max_abs.max((f.last().unwrap() * x / 100.0).abs());
            }
        }
        max_abs
    }

    #[test]
    fn decycler_oscillator_plot() {
        let mut f = osc(60, 1.0);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/decycler_oscillator.png").unwrap();
    }

    #[test]
    fn decycler_oscillator_warmup() {
        let mut f = osc(10, 1.2);
        assert_eq!(f.hp_period().get(), 10);
        assert_eq!(f.k(), 1.2);
        for i in 0..9 {
            f.update(1.0 + i as f64);
            assert_eq!(f.last(), None);
        }
        f.update(10.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn decycler_oscillator_matches_tasc_code() {
        // Direct transcription of the TASC code with the start-at-rest seeding.
        let (p, k) = (25.0_f64, 1.2);
        let alpha = |period: f64| {
            let a = 0.707 * 2.0 * PI / period;
            (a.cos() + a.sin() - 1.0) / a.cos()
        };
        let (a1, a2) = (alpha(p), alpha(0.5 * p));
        let mut f = osc(25, k);
        let x0 = TEST_DATA[0];
        let (mut x1, mut x2, mut h1, mut h2) = (x0, x0, 0.0, 0.0);
        let (mut d1, mut d2, mut o1, mut o2) = (x0, x0, 0.0, 0.0);
        for (i, x) in TEST_DATA.iter().enumerate() {
            let hp = (1.0 - a1 / 2.0).powi(2) * (x - 2.0 * x1 + x2) + 2.0 * (1.0 - a1) * h1
                - (1.0 - a1).powi(2) * h2;
            let d = x - hp;
            let o = (1.0 - a2 / 2.0).powi(2) * (d - 2.0 * d1 + d2) + 2.0 * (1.0 - a2) * o1
                - (1.0 - a2).powi(2) * o2;
            (x2, x1, h2, h1) = (x1, *x, h1, hp);
            (d2, d1, o2, o1) = (d1, d, o1, o);
            f.update(*x);
            if let Some(got) = f.last() {
                let want = 100.0 * k * o / x;
                assert!((got - want).abs() < 1e-9, "bar {i}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn decycler_oscillator_core_matches_high_pass() {
        let mut core = HighPass2Core::new(30.0);
        let mut hp = HighPass::new(
            Echo::new(),
            NonZeroUsize::new(30).unwrap(),
            HighPassPoles::Two,
        );
        for v in TEST_DATA {
            let c = core.update(v);
            hp.update(v);
            assert!((hp.last().unwrap_or(c) - c).abs() < 1e-12);
        }
    }

    #[test]
    fn decycler_oscillator_constant_and_ramp_are_zero() {
        let mut flat = osc(40, 1.0);
        let mut ramp = osc(40, 1.0);
        for i in 0..2_000 {
            flat.update(42.0);
            ramp.update(10.0 + i as f64);
        }
        assert_eq!(flat.last(), Some(0.0));
        assert!(ramp.last().unwrap().abs() < 1e-9);
    }

    #[test]
    fn decycler_oscillator_frequency_response() {
        let center = sine_gain(40, 30.0);
        assert!(center > 0.35, "center gain {center}");
        let slow = sine_gain(40, 400.0);
        assert!(slow < 0.1 * center, "slow gain {slow}");
        let fast = sine_gain(40, 4.0);
        assert!(fast < 0.3 * center, "fast gain {fast}");
    }

    #[test]
    fn decycler_oscillator_k_scales_output() {
        let mut a = osc(30, 1.0);
        let mut b = osc(30, 2.0);
        for v in TEST_DATA {
            a.update(v);
            b.update(v);
        }
        assert!((2.0 * a.last().unwrap() - b.last().unwrap()).abs() < 1e-12);
        assert_ne!(a.last(), Some(0.0));
    }

    #[test]
    #[should_panic(expected = "hp_period must be >= 6")]
    fn decycler_oscillator_period_too_small() {
        let _ = osc(5, 1.0);
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn decycler_oscillator_k_zero() {
        let _ = osc(20, 0.0);
    }

    #[test]
    #[should_panic(expected = "k must be finite")]
    fn decycler_oscillator_k_infinite() {
        let _ = osc(20, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "input must be positive")]
    fn decycler_oscillator_zero_input() {
        let mut f = osc(20, 1.0);
        f.update(1.0);
        f.update(0.0);
    }
}

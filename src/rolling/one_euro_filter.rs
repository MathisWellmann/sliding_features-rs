// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The 1€ Filter: a low-jitter, low-lag signal filter for noisy, jittery
//! signals such as pointer or sensor input.

use std::f64::consts::PI;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// An exponential low-pass filter that remembers both the raw value it was
/// last fed (`y`) and the smoothed value it last produced (`s`), as the 1€
/// Filter needs the raw value to estimate the derivative.
#[derive(Debug, Clone)]
struct LowPassFilter<T> {
    y: Option<T>,
    s: Option<T>,
}

impl<T: Float> LowPassFilter<T> {
    fn new() -> Self {
        Self { y: None, s: None }
    }

    /// Filter `value` with smoothing factor `alpha` in `(0, 1]`. The first
    /// call returns `value` unchanged, matching the reference implementation.
    fn filter(&mut self, value: T, alpha: T) -> T {
        let s = match self.s {
            None => value,
            Some(prev_s) => alpha * value + (T::one() - alpha) * prev_s,
        };
        self.y = Some(value);
        self.s = Some(s);
        s
    }

    fn last_raw_value(&self) -> Option<T> {
        self.y
    }
}

/// The 1€ Filter (Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based
/// Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012).
///
/// Low-pass filters both the signal and its derivative with an exponential
/// smoother whose cutoff frequency adapts to the estimated speed of change:
/// `cutoff = min_cutoff + beta * |derivative|`. This keeps jitter low while
/// standing still and lag low while moving fast, trading one knob (`beta`)
/// against the other (`min_cutoff`).
///
/// `rate` is the sampling frequency in updates per unit time; with the
/// crate's unit sample period convention (one update = one time step) the
/// default is `1.0`. `min_cutoff` and `d_cutoff` are cutoff frequencies in
/// the same units as `rate` (Hz if `rate` is in updates per second).
///
/// `last()` is `Some` from the first update onward (there is no window to
/// fill); it is the filtered value. `derivative()` returns the filtered
/// speed estimate used to adapt the cutoff.
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct OneEuroFilter<T: Float, V> {
    view: V,
    /// The minimum cutoff frequency, applied when the signal is not moving.
    #[getset(get_copy = "pub")]
    min_cutoff: T,
    /// The speed coefficient: how much the cutoff frequency grows per unit
    /// of estimated speed.
    #[getset(get_copy = "pub")]
    beta: T,
    /// The cutoff frequency used to filter the derivative itself.
    #[getset(get_copy = "pub")]
    d_cutoff: T,
    /// The sampling frequency, in updates per unit time.
    #[getset(get_copy = "pub")]
    rate: T,
    /// The last filtered derivative estimate.
    #[getset(get_copy = "pub")]
    derivative: T,
    x_filter: LowPassFilter<T>,
    dx_filter: LowPassFilter<T>,
    out: Option<T>,
}

impl<T, V> OneEuroFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new filter with a chained `View`, a minimum cutoff
    /// frequency, a speed coefficient `beta` and a cutoff frequency for the
    /// derivative, using a unit sample period (`rate = 1`).
    ///
    /// # Panics
    /// If `min_cutoff <= 0`, `beta < 0` or `d_cutoff <= 0`.
    pub fn new(view: V, min_cutoff: T, beta: T, d_cutoff: T) -> Self {
        Self::with_rate(view, min_cutoff, beta, d_cutoff, T::one())
    }

    /// Create a new filter with an explicit sampling `rate` (updates per
    /// unit time), for callers whose samples are not spaced one time unit
    /// apart.
    ///
    /// # Panics
    /// If `min_cutoff <= 0`, `beta < 0`, `d_cutoff <= 0` or `rate <= 0`.
    pub fn with_rate(view: V, min_cutoff: T, beta: T, d_cutoff: T, rate: T) -> Self {
        assert!(min_cutoff > T::zero(), "min_cutoff must be > 0");
        assert!(beta >= T::zero(), "beta must be >= 0");
        assert!(d_cutoff > T::zero(), "d_cutoff must be > 0");
        assert!(rate > T::zero(), "rate must be > 0");
        Self {
            view,
            min_cutoff,
            beta,
            d_cutoff,
            rate,
            derivative: T::zero(),
            x_filter: LowPassFilter::new(),
            dx_filter: LowPassFilter::new(),
            out: None,
        }
    }

    /// The exponential smoothing factor for a cutoff frequency, given the
    /// sample period `1 / rate`: `alpha = 1 / (1 + tau / te)` with
    /// `tau = 1 / (2 pi cutoff)`.
    fn alpha(&self, cutoff: T) -> T {
        let te = T::one() / self.rate;
        let tau = T::one() / (T::from(2.0 * PI).expect("can convert") * cutoff);
        T::one() / (T::one() + tau / te)
    }
}

impl<T, V> View<T> for OneEuroFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let dx = match self.x_filter.last_raw_value() {
            None => T::zero(),
            Some(prev_x) => (val - prev_x) * self.rate,
        };
        let d_alpha = self.alpha(self.d_cutoff);
        self.derivative = self.dx_filter.filter(dx, d_alpha);

        let cutoff = self.min_cutoff + self.beta * self.derivative.abs();
        let x_alpha = self.alpha(cutoff);
        let out = self.x_filter.filter(val, x_alpha);
        debug_assert!(out.is_finite(), "value must be finite");
        self.out = Some(out);
    }

    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::RollingQuantile,
        test_data::TEST_DATA,
    };

    fn one_euro(min_cutoff: f64, beta: f64) -> OneEuroFilter<f64, Echo<f64>> {
        OneEuroFilter::new(Echo::new(), min_cutoff, beta, 1.0)
    }

    #[test]
    fn one_euro_plot() {
        let mut f = one_euro(1.0, 0.1);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/one_euro_filter.png").unwrap();
    }

    #[test]
    fn one_euro_ready_from_first_update() {
        let mut f = one_euro(1.0, 0.0);
        assert_eq!(f.last(), None);
        f.update(3.0);
        assert_eq!(f.last(), Some(3.0));
    }

    #[test]
    fn one_euro_reduces_jitter_on_noisy_constant() {
        // Standard uniform jitter around a constant: less variance out than in.
        let rng = romu::Rng::from_seed_with_64bit(1);
        let mut f = one_euro(1.0, 0.0);
        let mut window = RollingQuantile::new(Echo::new(), NonZeroUsize::new(200).unwrap(), 0.5);
        let mut filtered = Vec::new();
        for _ in 0..500 {
            let x = 10.0 + (rng.f64() - 0.5) * 2.0;
            f.update(x);
            window.update(x);
            filtered.push(f.last().unwrap());
        }
        let mean = filtered.iter().sum::<f64>() / filtered.len() as f64;
        let filtered_var = filtered.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        let raw_var: f64 = 500.0 * (2.0 * 2.0 / 12.0); // Var[Uniform(-1, 1)] * n, approx.
        assert!(
            filtered_var < raw_var,
            "filtered variance {filtered_var} should be well below raw variance {raw_var}"
        );
    }

    #[test]
    fn one_euro_small_lag_on_fast_ramp() {
        // Reference transcription of the algorithm (Casiez et al. 2012),
        // operating directly on floats without the `View` abstraction.
        struct Reference {
            min_cutoff: f64,
            beta: f64,
            d_cutoff: f64,
            freq: f64,
            // Raw previous input, used only for the derivative estimate.
            x_raw_prev: Option<f64>,
            // Previous filtered outputs, used for the exponential smoothing itself.
            x_filt_prev: Option<f64>,
            dx_filt_prev: Option<f64>,
        }
        impl Reference {
            fn alpha(&self, cutoff: f64) -> f64 {
                let te = 1.0 / self.freq;
                let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
                1.0 / (1.0 + tau / te)
            }
            fn filter(&mut self, x: f64) -> f64 {
                let dx = match self.x_raw_prev {
                    None => 0.0,
                    Some(px) => (x - px) * self.freq,
                };
                let a_d = self.alpha(self.d_cutoff);
                let edx = match self.dx_filt_prev {
                    None => dx,
                    Some(prev) => a_d * dx + (1.0 - a_d) * prev,
                };
                self.dx_filt_prev = Some(edx);
                let cutoff = self.min_cutoff + self.beta * edx.abs();
                let a = self.alpha(cutoff);
                let s = match self.x_filt_prev {
                    None => x,
                    Some(prev_s) => a * x + (1.0 - a) * prev_s,
                };
                self.x_raw_prev = Some(x);
                self.x_filt_prev = Some(s);
                s
            }
        }
        let mut reference = Reference {
            min_cutoff: 1.0,
            beta: 0.5,
            d_cutoff: 1.0,
            freq: 1.0,
            x_raw_prev: None,
            x_filt_prev: None,
            dx_filt_prev: None,
        };
        let mut f = OneEuroFilter::new(Echo::new(), 1.0, 0.5, 1.0);
        let ramp: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        for x in &ramp {
            let want = reference.filter(*x);
            f.update(*x);
            let got = f.last().expect("ready from the first update");
            assert!(
                (got - want).abs() < 1e-9,
                "got {got}, reference {want}, x {x}"
            );
        }
        // Lag with respect to the ramp stays bounded.
        let last_x = *ramp.last().unwrap();
        let lag = last_x - f.last().unwrap();
        assert!(lag < 10.0, "lag {lag} should be small relative to the ramp");
    }

    #[test]
    #[should_panic(expected = "min_cutoff must be > 0")]
    fn one_euro_rejects_non_positive_min_cutoff() {
        let _ = one_euro(0.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "beta must be >= 0")]
    fn one_euro_rejects_negative_beta() {
        let _ = one_euro(1.0, -0.1);
    }
}

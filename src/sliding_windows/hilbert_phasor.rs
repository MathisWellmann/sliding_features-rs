// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use num::Float;

use super::ehlers_hilbert_core::HilbertCore;
use crate::View;

/// Number of inputs before the phasor is reported.
/// The Hilbert FIR gain depends on the measured period, which needs this long to settle
/// (see `HomodyneDominantCycle`). TA-Lib's `HT_PHASOR` lookback is 32 bars.
const WARM_UP: usize = 50;

/// John Ehlers' Hilbert transform phasor: the InPhase and Quadrature components of the
/// detrended, 4-bar WMA smoothed price, as computed by TA-Lib's `HT_PHASOR`.
///
/// The InPhase component is the detrender delayed by 3 bars (the group delay of the
/// Hilbert FIR), and the Quadrature component is the Hilbert transform of the detrender.
/// The FIR gain adapts to the dominant cycle period measured by the homodyne discriminator.
///
/// The output of [`View::last`] is the phasor angle `atan2(quadrature, in_phase)` in degrees,
/// in the range `(-180, 180]`. Following Ehlers' convention (`DeltaPhase = Phase[1] - Phase`),
/// the angle of a cycle *decreases* by `360 / period` degrees per bar and wraps from -180 to 180.
/// The InPhase and Quadrature components are available with [`Self::in_phase`] and
/// [`Self::quadrature`]; they are unbounded and scale with the input's cycle amplitude.
///
/// Reference: John Ehlers, "Rocket Science for Traders" (2001), chapter 7,
/// and TA-Lib's `HT_PHASOR`.
///
/// Complexity: O(1) per update. Ready after 50 inputs.
#[derive(Debug, Clone)]
pub struct HilbertPhasor<T, V> {
    view: V,
    core: HilbertCore<T>,
}

impl<T, V> HilbertPhasor<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View.
    pub fn new(view: V) -> Self {
        Self {
            view,
            core: HilbertCore::new(),
        }
    }

    fn is_ready(&self) -> bool {
        self.core.n_inputs() >= WARM_UP
    }

    /// The InPhase component, `None` while warming up.
    pub fn in_phase(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        let out = self.core.i1();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }

    /// The Quadrature component, `None` while warming up.
    pub fn quadrature(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        let out = self.core.q1();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

impl<T, V> View<T> for HilbertPhasor<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        self.core.update(val);
    }

    fn last(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        let out = self.core.q1().atan2(self.core.i1()).to_degrees();
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(
            out <= T::from(180.0).expect("can convert"),
            "angle above 180"
        );
        debug_assert!(
            out >= T::from(-180.0).expect("can convert"),
            "angle below -180"
        );
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    /// Wrap an angle difference into (-180, 180].
    fn wrap(deg: f64) -> f64 {
        let d = deg.rem_euclid(360.0);
        if d > 180.0 { d - 360.0 } else { d }
    }

    #[test]
    fn hilbert_phasor_warm_up() {
        let mut ph = HilbertPhasor::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(ph.last().is_none());
            assert!(ph.in_phase().is_none());
            assert!(ph.quadrature().is_none());
            ph.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(ph.last().is_some());
        assert!(ph.in_phase().is_some());
        assert!(ph.quadrature().is_some());
    }

    #[test]
    fn hilbert_phasor_rotation_rate_on_sine() {
        for period in [15.0, 20.0, 30.0] {
            let mut ph = HilbertPhasor::new(Echo::new());
            let mut prev: Option<f64> = None;
            let mut sum = 0.0;
            let mut n = 0;
            for i in 0..400 {
                ph.update(100.0 + (2.0 * PI * i as f64 / period).sin());
                let Some(angle) = ph.last() else { continue };
                assert!((-180.0..=180.0).contains(&angle));
                if let Some(p) = prev
                    && i >= 100
                {
                    sum += wrap(angle - p);
                    n += 1;
                }
                prev = Some(angle);
            }
            let rate = sum / n as f64;
            let expected = -360.0 / period;
            assert!(
                (rate - expected).abs() < 0.05 * expected.abs(),
                "period {period}: rate {rate}"
            );
        }
    }

    #[test]
    fn hilbert_phasor_components_in_quadrature() {
        // On a pure sine the phasor magnitude is nearly constant and I, Q are orthogonal.
        let mut ph = HilbertPhasor::new(Echo::new());
        let (mut min_mag, mut max_mag, mut dot) = (f64::MAX, 0.0_f64, 0.0);
        let (mut ii, mut qq) = (0.0, 0.0);
        for i in 0..400 {
            ph.update(100.0 + (2.0 * PI * i as f64 / 20.0).sin());
            if i < 100 {
                continue;
            }
            let (re, im) = (ph.in_phase().unwrap(), ph.quadrature().unwrap());
            let mag = re.hypot(im);
            min_mag = min_mag.min(mag);
            max_mag = max_mag.max(mag);
            dot += re * im;
            ii += re * re;
            qq += im * im;
        }
        assert!(min_mag > 0.0);
        assert!(max_mag / min_mag < 1.3, "{min_mag} {max_mag}");
        assert!(
            dot.abs() / (ii * qq).sqrt() < 0.05,
            "correlation of I and Q"
        );
    }

    #[test]
    fn hilbert_phasor_constant_input_has_no_cycle() {
        let mut ph = HilbertPhasor::new(Echo::new());
        for _ in 0..100 {
            ph.update(42.0);
        }
        assert_eq!(ph.in_phase(), Some(0.0));
        assert_eq!(ph.quadrature(), Some(0.0));
    }

    #[test]
    fn hilbert_phasor_plot() {
        let mut ph = HilbertPhasor::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            ph.update(*v);
            if let Some(val) = ph.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/hilbert_phasor.png").unwrap();
    }
}

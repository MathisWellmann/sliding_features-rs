use num::Float;

use super::{
    ehlers_hilbert_core::HilbertCore,
    ehlers_hilbert_dc_phase::DcPhase,
};
use crate::View;

/// Number of inputs before the sine wave is reported, equal to TA-Lib's `HT_SINE` lookback.
/// It covers the dominant cycle period settling (50 bars) and a full DFT history.
const WARM_UP: usize = 63;

/// John Ehlers' Hilbert Sine Wave indicator.
///
/// The dominant cycle period is measured with the homodyne discriminator
/// (see `HomodyneDominantCycle`). The dominant cycle phase is the phase of a single-bin DFT
/// of the 4-bar WMA smoothed price over one dominant cycle, corrected for the WMA lag.
/// The output of [`View::last`] is `sin(phase)`, and [`Self::lead_sine`] is
/// `sin(phase + 45 degrees)`. In cycle mode the two lines cross shortly before the cycle
/// turning points; in trend mode they meander without clean crossings.
///
/// Reference: John Ehlers, "Rocket Science for Traders" (2001), chapter 9,
/// and TA-Lib's `HT_SINE`.
///
/// Output range: `[-1, 1]` for both lines. Ready after 63 inputs.
/// Complexity: O(P) per update, where P <= 50 is the rounded dominant cycle period.
#[derive(Debug, Clone)]
pub struct SineWave<T, V> {
    view: V,
    core: HilbertCore<T>,
    dc_phase: DcPhase<T>,
}

impl<T, V> SineWave<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View.
    pub fn new(view: V) -> Self {
        Self {
            view,
            core: HilbertCore::new(),
            dc_phase: DcPhase::new(),
        }
    }

    /// The lead sine, `sin(phase + 45 degrees)` in `[-1, 1]`, `None` while warming up.
    pub fn lead_sine(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let lead = self.dc_phase.phase() + T::from(45.0).expect("can convert");
        let out = lead.to_radians().sin();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

impl<T, V> View<T> for SineWave<T, V>
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
        if !self.core.has_output() {
            return;
        }
        self.dc_phase
            .update(self.core.smooth(), self.core.smooth_period());
    }

    fn last(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let out = self.dc_phase.phase().to_radians().sin();
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
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    #[test]
    fn sine_wave_warm_up() {
        let mut sw = SineWave::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(sw.last().is_none());
            assert!(sw.lead_sine().is_none());
            sw.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(sw.last().is_some());
        assert!(sw.lead_sine().is_some());
    }

    #[test]
    fn sine_wave_tracks_pure_sine() {
        for period in [15.0, 20.0, 30.0] {
            let mut sw = SineWave::new(Echo::new());
            let (mut err, mut lead_err, mut n) = (0.0, 0.0, 0);
            for i in 0..400 {
                let theta = 2.0 * PI * i as f64 / period;
                sw.update(100.0 + 5.0 * theta.sin());
                if i < 100 {
                    continue;
                }
                err += (sw.last().unwrap() - theta.sin()).abs();
                lead_err += (sw.lead_sine().unwrap() - (theta + PI / 4.0).sin()).abs();
                n += 1;
            }
            let (err, lead_err) = (err / n as f64, lead_err / n as f64);
            assert!(err < 0.1, "period {period}: mean abs error {err}");
            assert!(
                lead_err < 0.1,
                "period {period}: mean abs lead error {lead_err}"
            );
        }
    }

    #[test]
    fn sine_wave_lead_crosses_first() {
        // The lead sine reaches each zero crossing 45 degrees (period / 8 bars) earlier.
        let period = 24.0;
        let mut sw = SineWave::new(Echo::new());
        let (mut sine_up, mut lead_up) = (Vec::new(), Vec::new());
        let (mut prev_s, mut prev_l) = (0.0, 0.0);
        for i in 0..400 {
            sw.update(100.0 + (2.0 * PI * i as f64 / period).sin());
            let (Some(s), Some(l)) = (sw.last(), sw.lead_sine()) else {
                continue;
            };
            if i > 100 && prev_s < 0.0 && s >= 0.0 {
                sine_up.push(i);
            }
            if i > 100 && prev_l < 0.0 && l >= 0.0 {
                lead_up.push(i);
            }
            (prev_s, prev_l) = (s, l);
        }
        assert!(sine_up.len() > 5);
        for s in sine_up {
            let l = lead_up
                .iter()
                .rev()
                .find(|l| **l <= s)
                .expect("lead crossed before");
            let delta = s - l;
            assert!((2..=4).contains(&delta), "lead by {delta} bars");
        }
    }

    #[test]
    fn sine_wave_range() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        let mut sw = SineWave::new(Echo::new());
        let mut price = 100.0;
        for _ in 0..10_000 {
            price += rng.f64() - 0.5;
            sw.update(price);
            if let Some(s) = sw.last() {
                assert!((-1.0..=1.0).contains(&s));
                let l = sw.lead_sine().unwrap();
                assert!((-1.0..=1.0).contains(&l));
            }
        }
    }

    #[test]
    fn sine_wave_plot() {
        let mut sw = SineWave::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            sw.update(*v);
            if let Some(val) = sw.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/sine_wave.png").unwrap();
    }
}

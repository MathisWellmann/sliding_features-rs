use num::Float;

use super::ehlers_hilbert_core::{
    HilbertCore,
    MAX_PERIOD,
    MIN_PERIOD,
    c,
};
use crate::View;

/// Number of inputs before the period is reported.
/// TA-Lib's `HT_DCPERIOD` lookback is 32 bars, but the period starts at zero and may only grow
/// 1.5x per bar (then smoothed twice), so 50 bars let it settle on cycles up to the 50 bar maximum.
const WARM_UP: usize = 50;

/// John Ehlers' Homodyne Discriminator dominant cycle period, in bars.
///
/// The price is smoothed by a 4-bar WMA, detrended and split into InPhase and Quadrature
/// components with a Hilbert transformer FIR. The homodyne discriminator multiplies the
/// phasor with the complex conjugate of its previous value, so the angle of the product is the
/// phase change per bar, and `360 / angle` is the period.
/// The period is limited to `[0.67, 1.5]` times the previous period and to `[6, 50]` bars,
/// smoothed `0.2 / 0.8` (see [`Self::period`]) and then `0.33 / 0.67` (the output).
///
/// Reference: John Ehlers, "Rocket Science for Traders" (2001), chapter 7,
/// and TA-Lib's `HT_DCPERIOD`.
///
/// Output range: `[6, 50]` bars once warmed up (50 inputs).
/// Complexity: O(1) per update.
#[derive(Debug, Clone)]
pub struct HomodyneDominantCycle<T, V> {
    view: V,
    core: HilbertCore<T>,
}

impl<T, V> HomodyneDominantCycle<T, V>
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

    /// The period before the final `0.33 / 0.67` smoothing, in `[6, 50]` bars.
    /// It reacts faster to cycle changes than the output of [`View::last`].
    pub fn period(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let out = clamp_period(self.core.period());
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

impl<T, V> View<T> for HomodyneDominantCycle<T, V>
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
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let out = clamp_period(self.core.smooth_period());
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

/// Limit a reported period to `[6, 50]` bars. The periods are smoothed from zero, so on
/// input without a cycle (e.g. a constant) they still approach 6 from below after the
/// warm-up, e.g. 5.9997 after 50 constant inputs.
#[inline(always)]
fn clamp_period<T: Float>(period: T) -> T {
    let out = period.max(c(MIN_PERIOD)).min(c(MAX_PERIOD));
    debug_assert!(out >= c(MIN_PERIOD), "period below minimum");
    debug_assert!(out <= c(MAX_PERIOD), "period above maximum");
    out
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

    /// Mean of the reported period over the bars `[skip, n)` of a sine with `period` bars.
    fn mean_period_of_sine(period: f64, n: usize, skip: usize) -> f64 {
        let mut dc = HomodyneDominantCycle::new(Echo::new());
        let mut sum = 0.0;
        for i in 0..n {
            dc.update(100.0 + (2.0 * PI * i as f64 / period).sin());
            if i >= skip {
                sum += dc.last().unwrap();
            }
        }
        sum / (n - skip) as f64
    }

    #[test]
    fn homodyne_dominant_cycle_warm_up() {
        let mut dc = HomodyneDominantCycle::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(dc.last().is_none());
            assert!(dc.period().is_none());
            dc.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(dc.last().is_some());
        assert!(dc.period().is_some());
    }

    #[test]
    fn homodyne_dominant_cycle_sine_periods() {
        for period in [10.0, 15.0, 20.0, 30.0, 40.0] {
            let measured = mean_period_of_sine(period, 400, 100);
            assert!(
                (measured - period).abs() < 0.1 * period,
                "period {period}: measured {measured}"
            );
        }
    }

    #[test]
    fn homodyne_dominant_cycle_settled_after_warm_up() {
        for period in [15.0, 20.0, 30.0] {
            let measured = mean_period_of_sine(period, WARM_UP + 1, WARM_UP);
            assert!(
                (measured - period).abs() < 0.15 * period,
                "period {period}: measured {measured}"
            );
        }
    }

    #[test]
    fn homodyne_dominant_cycle_tracks_period_change() {
        let mut dc = HomodyneDominantCycle::new(Echo::new());
        let mut phase = 0.0;
        for i in 0..400 {
            let period = if i < 200 { 15.0 } else { 30.0 };
            phase += 2.0 * PI / period;
            dc.update(100.0 + phase.sin());
            if i == 199 {
                assert!((dc.last().unwrap() - 15.0).abs() < 1.5);
            }
        }
        assert!((dc.last().unwrap() - 30.0).abs() < 3.0);
        assert!((dc.period().unwrap() - 30.0).abs() < 3.0);
    }

    #[test]
    fn homodyne_dominant_cycle_range() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let mut dc = HomodyneDominantCycle::new(Echo::new());
        let mut price = 100.0;
        for _ in 0..10_000 {
            price += rng.f64() - 0.5;
            dc.update(price);
            if let Some(p) = dc.last() {
                assert!((6.0..=50.0).contains(&p), "{p}");
                let raw = dc.period().unwrap();
                assert!((6.0..=50.0).contains(&raw), "{raw}");
            }
        }
    }

    #[test]
    fn homodyne_dominant_cycle_range_on_constant_input() {
        for value in [0.0, 5.0] {
            let mut dc = HomodyneDominantCycle::new(Echo::new());
            for _ in 0..300 {
                dc.update(value);
                if let Some(p) = dc.last() {
                    assert!((6.0..=50.0).contains(&p), "{p}");
                    let raw = dc.period().unwrap();
                    assert!((6.0..=50.0).contains(&raw), "{raw}");
                }
            }
        }
    }

    #[test]
    fn homodyne_dominant_cycle_f32() {
        let mut dc = HomodyneDominantCycle::<f32, _>::new(Echo::new());
        for i in 0..400 {
            dc.update(100.0 + (2.0 * std::f32::consts::PI * i as f32 / 20.0).sin());
        }
        let p = dc.last().unwrap();
        assert!((p - 20.0).abs() < 2.0, "{p}");
    }

    #[test]
    fn homodyne_dominant_cycle_plot() {
        let mut dc = HomodyneDominantCycle::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            dc.update(*v);
            if let Some(val) = dc.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/homodyne_dominant_cycle.png").unwrap();
    }
}

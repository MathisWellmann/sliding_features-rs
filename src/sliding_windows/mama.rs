// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use num::Float;

use super::ehlers_hilbert_core::{
    HilbertCore,
    c,
};
use crate::View;

/// Number of inputs before MAMA is reported, equal to TA-Lib's `MAMA` lookback.
/// MAMA starts at the first input, so only the phase measurement needs to settle.
const WARM_UP: usize = 32;

/// MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA)
/// by John Ehlers.
///
/// An exponential moving average whose smoothing factor adapts to the rate of change of the
/// Hilbert transform phase `atan(Q1 / I1)`: `alpha = fast_limit / DeltaPhase` (in degrees,
/// `DeltaPhase >= 1`), limited to `[slow_limit, fast_limit]`. In trends the phase barely moves,
/// so alpha goes to `fast_limit` and MAMA follows the price closely; in cycles the phase
/// advances, so alpha drops towards `slow_limit` and the cycle is filtered out.
/// FAMA applies `alpha / 2` to MAMA, so it lags MAMA; their crossings are trade signals.
///
/// Reference: John Ehlers, "MESA Adaptive Moving Averages" (2001),
/// <https://www.mesasoftware.com/papers/MAMA.pdf>, and TA-Lib's `MAMA`.
///
/// Deviation: MAMA and FAMA start at the first input instead of zero, so they need no
/// long ramp-up from zero. Like Ehlers' code (unlike TA-Lib, which resets it to zero), the
/// phase is held when the InPhase component is zero.
///
/// Output range: within the range of the inputs. Ready after 32 inputs.
/// Complexity: O(1) per update.
#[derive(Debug, Clone)]
pub struct Mama<T, V> {
    view: V,
    core: HilbertCore<T>,
    fast_limit: T,
    slow_limit: T,
    /// Hilbert phase in degrees.
    phase: T,
    mama: T,
    fama: T,
}

/// Ehlers' adaptive smoothing factor from the phase change per bar in degrees.
#[inline(always)]
fn adaptive_alpha<T: Float>(delta_phase: T, fast_limit: T, slow_limit: T) -> T {
    debug_assert!(slow_limit <= fast_limit, "slow limit above fast limit");
    let alpha = (fast_limit / delta_phase.max(T::one())).max(slow_limit);
    debug_assert!(alpha <= fast_limit, "alpha above fast limit");
    debug_assert!(alpha >= slow_limit, "alpha below slow limit");
    alpha
}

impl<T, V> Mama<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View and Ehlers' default limits
    /// `fast_limit = 0.5` and `slow_limit = 0.05`.
    pub fn new(view: V) -> Self {
        Self::with_limits(view, c(0.5), c(0.05))
    }

    /// Create a new instance with a chained View and custom smoothing factor limits.
    ///
    /// # Panics
    /// Unless `0 < slow_limit <= fast_limit < 1`.
    pub fn with_limits(view: V, fast_limit: T, slow_limit: T) -> Self {
        assert!(slow_limit > T::zero(), "slow_limit must be positive");
        assert!(
            slow_limit <= fast_limit,
            "slow_limit must not exceed fast_limit"
        );
        assert!(fast_limit < T::one(), "fast_limit must be below 1");
        Self {
            view,
            core: HilbertCore::new(),
            fast_limit,
            slow_limit,
            phase: T::zero(),
            mama: T::zero(),
            fama: T::zero(),
        }
    }

    /// The upper limit of the smoothing factor, used when the phase does not advance.
    pub fn fast_limit(&self) -> T {
        self.fast_limit
    }

    /// The lower limit of the smoothing factor, used when the phase advances quickly.
    pub fn slow_limit(&self) -> T {
        self.slow_limit
    }

    /// The Following Adaptive Moving Average, `None` while warming up.
    pub fn fama(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        debug_assert!(self.fama.is_finite(), "value must be finite");
        Some(self.fama)
    }
}

impl<T, V> View<T> for Mama<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        if self.core.n_inputs() == 0 {
            self.mama = val;
            self.fama = val;
        }
        self.core.update(val);

        let prev_phase = self.phase;
        let i1 = self.core.i1();
        if i1 != T::zero() {
            self.phase = (self.core.q1() / i1).atan().to_degrees();
        }
        // Ehlers' phase decreases as a cycle advances, hence previous minus current.
        let alpha = adaptive_alpha(prev_phase - self.phase, self.fast_limit, self.slow_limit);
        self.mama = alpha * val + (T::one() - alpha) * self.mama;
        let half = c::<T>(0.5) * alpha;
        self.fama = half * self.mama + (T::one() - half) * self.fama;
        debug_assert!(self.mama.is_finite(), "value must be finite");
        debug_assert!(self.fama.is_finite(), "value must be finite");
    }

    fn last(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        debug_assert!(self.mama.is_finite(), "value must be finite");
        Some(self.mama)
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
    fn mama_warm_up() {
        let mut m = Mama::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(m.last().is_none());
            assert!(m.fama().is_none());
            m.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(m.last().is_some());
        assert!(m.fama().is_some());
    }

    #[test]
    fn mama_defaults() {
        let m = Mama::<f64, _>::new(Echo::new());
        assert_eq!(m.fast_limit(), 0.5);
        assert_eq!(m.slow_limit(), 0.05);
    }

    #[test]
    fn mama_adaptive_alpha() {
        assert_eq!(adaptive_alpha(0.0, 0.5, 0.05), 0.5);
        assert_eq!(adaptive_alpha(-170.0, 0.5, 0.05), 0.5);
        assert_eq!(adaptive_alpha(5.0, 0.5, 0.05), 0.1);
        assert_eq!(adaptive_alpha(18.0, 0.5, 0.05), 0.05);
    }

    #[test]
    fn mama_follows_step_and_fama_lags() {
        let mut m = Mama::new(Echo::new());
        for i in 0..400 {
            let price = if i < 100 { 100.0 } else { 110.0 };
            m.update(price);
            let (Some(mama), Some(fama)) = (m.last(), m.fama()) else {
                continue;
            };
            assert!((100.0..=110.0).contains(&mama), "{mama}");
            assert!((100.0..=110.0).contains(&fama), "{fama}");
            if i >= 100 {
                assert!(fama <= mama, "FAMA must lag MAMA after a step up");
            }
            if i == 101 {
                assert!(mama > 104.0, "MAMA reacts quickly: {mama}");
                assert!(fama < mama - 1.0, "FAMA reacts slower: {fama}");
            }
        }
        assert!((m.last().unwrap() - 110.0).abs() < 1e-3);
        assert!((m.fama().unwrap() - 110.0).abs() < 1e-2);
    }

    #[test]
    fn mama_ramp_fast_tracking() {
        // On a ramp the phase barely moves, so alpha stays near the fast limit and MAMA lags
        // by about slope * (1 - alpha) / alpha = 1 bar on average, far less than the 19 bars at the slow limit.
        // The lag wobbles because the measured period, and with it the Hilbert FIR gain,
        // wanders on a cycle-free input.
        let mut m = Mama::new(Echo::new());
        let (mut sum_lag, mut max_lag, mut n) = (0.0, 0.0_f64, 0);
        for i in 0..400 {
            let price = 100.0 + i as f64;
            m.update(price);
            let (Some(mama), Some(fama)) = (m.last(), m.fama()) else {
                continue;
            };
            assert!(mama < price);
            assert!(fama < mama);
            if i >= 100 {
                sum_lag += price - mama;
                max_lag = max_lag.max(price - mama);
                n += 1;
            }
        }
        let mean_lag = sum_lag / n as f64;
        assert!(mean_lag < 3.0, "mean lag {mean_lag}");
        assert!(max_lag < 6.0, "max lag {max_lag}");
    }

    #[test]
    fn mama_cycle_slow_smoothing() {
        // On a cycle the phase advances 360 / period degrees per bar, alpha drops to the slow
        // limit and MAMA strongly attenuates the cycle.
        let amplitude = 5.0;
        let mut m = Mama::new(Echo::new());
        for i in 0..400 {
            m.update(100.0 + amplitude * (2.0 * PI * i as f64 / 20.0).sin());
            if i >= 150 {
                let dev = (m.last().unwrap() - 100.0).abs();
                assert!(dev < 0.4 * amplitude, "{dev}");
            }
        }
    }

    #[test]
    fn mama_range_f32() {
        let rng = romu::Rng::from_seed_with_64bit(5);
        let mut m = Mama::<f32, _>::new(Echo::new());
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for _ in 0..10_000 {
            let v = 100.0 + 10.0 * rng.f32();
            lo = lo.min(v);
            hi = hi.max(v);
            m.update(v);
            if let (Some(mama), Some(fama)) = (m.last(), m.fama()) {
                assert!((lo..=hi).contains(&mama));
                assert!((lo..=hi).contains(&fama));
            }
        }
    }

    #[test]
    #[should_panic(expected = "slow_limit must be positive")]
    fn mama_zero_slow_limit() {
        let _ = Mama::with_limits(Echo::<f64>::new(), 0.5, 0.0);
    }

    #[test]
    #[should_panic(expected = "slow_limit must not exceed fast_limit")]
    fn mama_slow_above_fast() {
        let _ = Mama::with_limits(Echo::<f64>::new(), 0.1, 0.2);
    }

    #[test]
    #[should_panic(expected = "fast_limit must be below 1")]
    fn mama_fast_limit_one() {
        let _ = Mama::with_limits(Echo::<f64>::new(), 1.0, 0.05);
    }

    #[test]
    #[should_panic(expected = "slow_limit must be positive")]
    fn mama_nan_limit() {
        let _ = Mama::with_limits(Echo::<f64>::new(), 0.5, f64::NAN);
    }

    #[test]
    fn mama_equal_limits_is_ema() {
        // With equal limits alpha is constant, so MAMA is a plain EMA.
        let mut m = Mama::with_limits(Echo::new(), 0.2, 0.2);
        let mut ema = TEST_DATA[0];
        for v in &TEST_DATA {
            m.update(*v);
            ema = 0.2 * v + 0.8 * ema;
            if let Some(mama) = m.last() {
                assert!((mama - ema).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn mama_plot() {
        let mut m = Mama::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            m.update(*v);
            if let Some(val) = m.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/mama.png").unwrap();
    }
}

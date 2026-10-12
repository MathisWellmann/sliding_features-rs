use num::Float;

use super::{
    ehlers_hilbert_core::{
        HilbertCore,
        c,
    },
    ehlers_hilbert_dc_phase::DcPhase,
    ehlers_hilbert_itrend::ITrend,
};
use crate::View;

/// Number of inputs before the mode is reported, equal to TA-Lib's `HT_TRENDMODE` lookback.
/// It covers the dominant cycle period settling (50 bars) and full DFT and price histories.
const WARM_UP: usize = 63;

/// Ehlers' trend / cycle decision for one bar, as in TA-Lib's `HT_TRENDMODE`.
/// Returns `true` for trend mode.
#[inline(always)]
fn is_trend<T: Float>(
    crossed: bool,
    days_in_trend: T,
    smooth_period: T,
    delta_phase: T,
    smooth_price: T,
    trendline: T,
) -> bool {
    debug_assert!(days_in_trend >= T::one(), "days in trend start at one");
    debug_assert!(smooth_period > T::zero(), "smooth period must be positive");
    // A sine / lead sine crossing means the market is cycling, and so does a fresh crossing
    // that happened less than half a cycle ago.
    let mut trend = !crossed && days_in_trend >= c::<T>(0.5) * smooth_period;
    // A phase advance within [0.67, 1.5] of the dominant cycle's rate is a cycle.
    let rate = c::<T>(360.0) / smooth_period;
    if delta_phase > c::<T>(0.67) * rate && delta_phase < c::<T>(1.5) * rate {
        trend = false;
    }
    // A smoothed price more than 1.5% away from the trendline overrides it all.
    if trendline != T::zero() && ((smooth_price - trendline) / trendline).abs() >= c(0.015) {
        trend = true;
    }
    trend
}

/// John Ehlers' Hilbert transform trend vs. cycle mode, as implemented by TA-Lib's
/// `HT_TRENDMODE`.
///
/// Defaults to trend mode and switches to cycle mode when
/// - the Sine Wave and Lead Sine (see `SineWave`) crossed less than half a dominant cycle ago, or
/// - the dominant cycle phase advances at 0.67 to 1.5 times the rate `360 / period` of the
///   dominant cycle.
///
/// It is forced back to trend mode when the 4-bar WMA smoothed price deviates from the
/// instantaneous trendline ([`Self::trendline`], see `InstantaneousTrendline`) by at least 1.5%.
/// That threshold is relative, so the input should be a positive price.
///
/// Reference: John Ehlers, "Rocket Science for Traders" (2001), and TA-Lib's `HT_TRENDMODE`.
///
/// Output: 1 for trend mode, 0 for cycle mode. Ready after 63 inputs.
/// Complexity: O(P) per update, where P <= 50 is the rounded dominant cycle period.
#[derive(Debug, Clone)]
pub struct HilbertTrendMode<T, V> {
    view: V,
    core: HilbertCore<T>,
    dc_phase: DcPhase<T>,
    itrend: ITrend<T>,
    sine: T,
    lead_sine: T,
    days_in_trend: usize,
    trend: bool,
}

impl<T, V> HilbertTrendMode<T, V>
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
            itrend: ITrend::new(),
            sine: T::zero(),
            lead_sine: T::zero(),
            days_in_trend: 0,
            trend: true,
        }
    }

    /// The instantaneous trendline used for the decision, `None` while warming up.
    pub fn trendline(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let out = self.itrend.trendline();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

impl<T, V> View<T> for HilbertTrendMode<T, V>
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
        let smooth_period = self.core.smooth_period();
        self.itrend.update(val, smooth_period);
        if !self.core.has_output() {
            return;
        }

        let prev_phase = self.dc_phase.phase();
        self.dc_phase.update(self.core.smooth(), smooth_period);
        let phase = self.dc_phase.phase();
        let (prev_sine, prev_lead_sine) = (self.sine, self.lead_sine);
        self.sine = phase.to_radians().sin();
        self.lead_sine = (phase + c(45.0)).to_radians().sin();

        let crossed = (self.sine > self.lead_sine && prev_sine <= prev_lead_sine)
            || (self.sine < self.lead_sine && prev_sine >= prev_lead_sine);
        if crossed {
            self.days_in_trend = 0;
        }
        self.days_in_trend = self.days_in_trend.saturating_add(1);
        self.trend = is_trend(
            crossed,
            T::from(self.days_in_trend).expect("can convert"),
            smooth_period,
            phase - prev_phase,
            self.core.smooth(),
            self.itrend.trendline(),
        );
    }

    fn last(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        Some(if self.trend { T::one() } else { T::zero() })
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::InstantaneousTrendline,
        test_data::TEST_DATA,
    };

    /// Fraction of bars in trend mode over `[skip, prices.len())`.
    fn trend_fraction(prices: impl Iterator<Item = f64>, skip: usize) -> f64 {
        let mut tm = HilbertTrendMode::new(Echo::new());
        let (mut trend, mut n) = (0.0, 0);
        for (i, p) in prices.enumerate() {
            tm.update(p);
            if i >= skip {
                let mode = tm.last().unwrap();
                assert!(mode == 0.0 || mode == 1.0);
                trend += mode;
                n += 1;
            }
        }
        trend / n as f64
    }

    #[test]
    fn hilbert_trend_mode_warm_up() {
        let mut tm = HilbertTrendMode::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(tm.last().is_none());
            assert!(tm.trendline().is_none());
            tm.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(tm.last().is_some());
        assert!(tm.trendline().is_some());
    }

    #[test]
    fn hilbert_trend_mode_ramp_is_trend() {
        let ramp = (0..400).map(|i| 100.0 + 2.0 * i as f64);
        let f = trend_fraction(ramp, 100);
        assert!(f > 0.95, "trend fraction {f}");
    }

    #[test]
    fn hilbert_trend_mode_sine_is_cycle() {
        for period in [15.0, 20.0, 30.0] {
            let sine = (0..400).map(|i| 100.0 + (2.0 * PI * i as f64 / period).sin());
            let f = trend_fraction(sine, 100);
            assert!(f < 0.05, "period {period}: trend fraction {f}");
        }
    }

    #[test]
    fn hilbert_trend_mode_switches() {
        // A cycle followed by a strong ramp: cycle mode first, trend mode later.
        let prices = (0..400).map(|i| {
            let cycle = 100.0 + (2.0 * PI * i as f64 / 20.0).sin();
            if i < 200 {
                cycle
            } else {
                cycle + 2.0 * (i - 200) as f64
            }
        });
        let mut tm = HilbertTrendMode::new(Echo::new());
        let mut modes = Vec::new();
        for p in prices {
            tm.update(p);
            modes.push(tm.last());
        }
        assert_eq!(modes[199], Some(0.0));
        assert!(modes[250..].iter().all(|m| *m == Some(1.0)));
    }

    #[test]
    fn hilbert_trend_mode_trendline_matches_instantaneous_trendline() {
        let mut tm = HilbertTrendMode::new(Echo::new());
        let mut it = InstantaneousTrendline::new(Echo::new());
        for v in &TEST_DATA {
            tm.update(*v);
            it.update(*v);
            assert_eq!(tm.trendline(), it.last());
        }
    }

    #[test]
    fn hilbert_trend_mode_decision() {
        // Default is trend, after enough days without crossing and no cycle-like phase rate.
        assert!(is_trend(false, 20.0, 20.0, 0.0, 100.0, 100.0));
        // A crossing or a recent crossing means cycle.
        assert!(!is_trend(true, 1.0, 20.0, 0.0, 100.0, 100.0));
        assert!(!is_trend(false, 9.0, 20.0, 0.0, 100.0, 100.0));
        // Phase rate band edges: 0.67 * 18 = 12.06 and 1.5 * 18 = 27 are exclusive.
        assert!(!is_trend(false, 20.0, 20.0, 18.0, 100.0, 100.0));
        assert!(is_trend(false, 20.0, 20.0, 12.0, 100.0, 100.0));
        assert!(is_trend(false, 20.0, 20.0, 27.0, 100.0, 100.0));
        // 1.5% away from the trendline forces trend, just below it does not.
        assert!(is_trend(true, 1.0, 20.0, 18.0, 101.5, 100.0));
        assert!(is_trend(true, 1.0, 20.0, 18.0, 98.5, 100.0));
        assert!(!is_trend(true, 1.0, 20.0, 18.0, 101.49, 100.0));
    }

    #[test]
    fn hilbert_trend_mode_plot() {
        let mut tm = HilbertTrendMode::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            tm.update(*v);
            if let Some(val) = tm.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/hilbert_trend_mode.png").unwrap();
    }
}

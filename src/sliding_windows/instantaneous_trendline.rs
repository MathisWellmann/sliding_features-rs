use num::Float;

use super::{
    ehlers_hilbert_core::HilbertCore,
    ehlers_hilbert_itrend::ITrend,
};
use crate::View;

/// Number of inputs before the trendline is reported, equal to TA-Lib's `HT_TRENDLINE` lookback.
/// It covers the dominant cycle period settling (50 bars) and a full price history.
const WARM_UP: usize = 63;

/// John Ehlers' Instantaneous Trendline (ITrend).
///
/// The dominant cycle period is measured with the homodyne discriminator
/// (see `HomodyneDominantCycle`). Averaging the price over exactly one dominant cycle removes
/// that cycle, leaving the trend; the average is then smoothed with a 4-bar WMA.
/// Because the averaging window follows the market's cycle, the trendline has less lag than a
/// fixed-length moving average that is long enough to suppress the cycle.
///
/// Reference: John Ehlers, "Rocket Science for Traders" (2001),
/// and TA-Lib's `HT_TRENDLINE`.
///
/// Output range: within the range of the inputs. Ready after 63 inputs.
/// Complexity: O(P) per update, where P <= 50 is the rounded dominant cycle period.
#[derive(Debug, Clone)]
pub struct InstantaneousTrendline<T, V> {
    view: V,
    core: HilbertCore<T>,
    itrend: ITrend<T>,
}

impl<T, V> InstantaneousTrendline<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View.
    pub fn new(view: V) -> Self {
        Self {
            view,
            core: HilbertCore::new(),
            itrend: ITrend::new(),
        }
    }
}

impl<T, V> View<T> for InstantaneousTrendline<T, V>
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
        self.itrend.update(val, self.core.smooth_period());
    }

    fn last(&self) -> Option<T> {
        if self.core.n_inputs() < WARM_UP {
            return None;
        }
        let out = self.itrend.trendline();
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
    fn instantaneous_trendline_warm_up() {
        let mut it = InstantaneousTrendline::new(Echo::new());
        for i in 0..WARM_UP {
            assert!(it.last().is_none());
            it.update(100.0 + (i as f64 * 0.3).sin());
        }
        assert!(it.last().is_some());
    }

    #[test]
    fn instantaneous_trendline_constant() {
        let mut it = InstantaneousTrendline::new(Echo::new());
        for _ in 0..200 {
            it.update(42.0);
        }
        assert!((it.last().unwrap() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn instantaneous_trendline_removes_cycle() {
        for period in [15.0, 20.0, 30.0] {
            let amplitude = 5.0;
            let mut it = InstantaneousTrendline::new(Echo::new());
            for i in 0..400 {
                it.update(100.0 + amplitude * (2.0 * PI * i as f64 / period).sin());
                if i >= 150 {
                    let dev = (it.last().unwrap() - 100.0).abs();
                    assert!(dev < 0.15 * amplitude, "period {period}: deviation {dev}");
                }
            }
        }
    }

    #[test]
    fn instantaneous_trendline_follows_trend() {
        // A ramp with a superimposed cycle: the trendline rises with the ramp,
        // lagging it by less than one maximal dominant cycle.
        let slope = 0.5;
        let mut it = InstantaneousTrendline::new(Echo::new());
        let mut prev: Option<f64> = None;
        for i in 0..400 {
            let trend = 100.0 + slope * i as f64;
            it.update(trend + (2.0 * PI * i as f64 / 20.0).sin());
            let Some(tl) = it.last() else { continue };
            if i < 100 {
                continue;
            }
            let lag = trend - tl;
            assert!(lag > 0.0, "{lag}");
            assert!(lag < slope * 25.0, "{lag}");
            if let Some(p) = prev {
                assert!(tl > p);
            }
            prev = Some(tl);
        }
    }

    #[test]
    fn instantaneous_trendline_plot() {
        let mut it = InstantaneousTrendline::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            it.update(*v);
            if let Some(val) = it.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/instantaneous_trendline.png").unwrap();
    }
}

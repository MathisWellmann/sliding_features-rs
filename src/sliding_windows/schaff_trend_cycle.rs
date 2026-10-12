//! Schaff Trend Cycle

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Ema,
    Max,
    Min,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// Schaff Trend Cycle (Doug Schaff, 1999).
///
/// The classic "double stochastic of MACD" variant, as commonly implemented
/// on charting platforms (e.g. the MT4/MT5 "Schaff Trend Cycle" indicator):
/// 1. `macd = EMA(fast_period) - EMA(slow_period)`.
/// 2. First stochastic of `macd` over `cycle_period`:
///    `f1 = 100 * (macd - min(macd, cycle)) / (max(macd, cycle) - min(macd, cycle))`,
///    recursively smoothed with factor `0.5`: `pf = pf_prev + 0.5 * (f1 - pf_prev)`.
/// 3. Second stochastic of `pf` over `cycle_period`, smoothed the same way,
///    giving the final output `pff`.
///
/// Flat-window behavior: if a stochastic's range (`max - min`) is `0`, its
/// raw `f1`/`f2` holds its previous value (or `0` on the very first such
/// window), rather than being `NaN`.
///
/// `last()` is `None` until both stochastic stages have seen `cycle_period`
/// values.
///
/// Output range: `[0, 100]` (both stochastic ratios are in `[0, 100]`, and
/// the `0.5` recursive smoothing is a convex combination, so it cannot leave
/// that range).
///
/// Complexity: O(1) amortized per update (two chained EMAs and four
/// monotonic min/max deques), O(cycle_period) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct SchaffTrendCycle<T, V> {
    view: V,
    /// The fast EMA period.
    #[getset(get_copy = "pub")]
    fast_period: NonZeroUsize,
    /// The slow EMA period.
    #[getset(get_copy = "pub")]
    slow_period: NonZeroUsize,
    /// The stochastic cycle length, used for both stochastic stages.
    #[getset(get_copy = "pub")]
    cycle_period: NonZeroUsize,
    ema_fast: Ema<T, Echo<T>>,
    ema_slow: Ema<T, Echo<T>>,
    max1: Max<T, Echo<T>>,
    min1: Min<T, Echo<T>>,
    max2: Max<T, Echo<T>>,
    min2: Min<T, Echo<T>>,
    n_macd: usize,
    n_pf: usize,
    prev_f1: Option<T>,
    prev_f2: Option<T>,
    pf: Option<T>,
    out: Option<T>,
}

impl<T, V> SchaffTrendCycle<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Schaff Trend Cycle with a chained `View` and custom
    /// fast, slow and cycle periods.
    ///
    /// # Panics
    /// If `fast_period >= slow_period`.
    pub fn with_periods(
        view: V,
        fast_period: NonZeroUsize,
        slow_period: NonZeroUsize,
        cycle_period: NonZeroUsize,
    ) -> Self {
        assert!(
            fast_period.get() < slow_period.get(),
            "the fast period must be shorter than the slow period"
        );
        Self {
            view,
            fast_period,
            slow_period,
            cycle_period,
            ema_fast: Ema::new(Echo::new(), fast_period),
            ema_slow: Ema::new(Echo::new(), slow_period),
            max1: Max::new(Echo::new(), cycle_period),
            min1: Min::new(Echo::new(), cycle_period),
            max2: Max::new(Echo::new(), cycle_period),
            min2: Min::new(Echo::new(), cycle_period),
            n_macd: 0,
            n_pf: 0,
            prev_f1: None,
            prev_f2: None,
            pf: None,
            out: None,
        }
    }

    /// Create a new Schaff Trend Cycle with a chained `View` and the classic
    /// default periods: 23 (fast), 50 (slow) and a 10-bar cycle.
    pub fn new(view: V) -> Self {
        Self::with_periods(
            view,
            NonZeroUsize::new(23).expect("23 is non-zero"),
            NonZeroUsize::new(50).expect("50 is non-zero"),
            NonZeroUsize::new(10).expect("10 is non-zero"),
        )
    }

    /// One `%K`-like stochastic stage with `0.5` recursive smoothing.
    /// Returns the smoothed value, or `None` if the window is not yet full.
    fn stochastic_stage(
        x: T,
        hi: T,
        lo: T,
        prev_raw: &mut Option<T>,
        prev_smoothed: Option<T>,
    ) -> T {
        let hundred = T::from(100.0).expect("can convert");
        let raw = if hi == lo {
            prev_raw.unwrap_or(T::zero())
        } else {
            hundred * (x - lo) / (hi - lo)
        };
        *prev_raw = Some(raw);
        let half = T::from(0.5).expect("can convert");
        match prev_smoothed {
            None => raw,
            Some(prev) => prev + half * (raw - prev),
        }
    }
}

impl<T, V> View<T> for SchaffTrendCycle<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.ema_fast.update(val);
        self.ema_slow.update(val);
        let (Some(fast), Some(slow)) = (self.ema_fast.last(), self.ema_slow.last()) else {
            return;
        };
        let macd = fast - slow;
        debug_assert!(macd.is_finite(), "`macd` must be finite");

        self.max1.update(macd);
        self.min1.update(macd);
        self.n_macd += 1;
        if self.n_macd < self.cycle_period.get() {
            return;
        }
        let hi1 = self.max1.last().expect("at least one value was observed");
        let lo1 = self.min1.last().expect("at least one value was observed");
        debug_assert!(hi1 >= lo1, "the max is never below the min");
        let pf = Self::stochastic_stage(macd, hi1, lo1, &mut self.prev_f1, self.pf);
        debug_assert!(pf.is_finite(), "`pf` must be finite");
        self.pf = Some(pf);

        self.max2.update(pf);
        self.min2.update(pf);
        self.n_pf += 1;
        if self.n_pf < self.cycle_period.get() {
            return;
        }
        let hi2 = self.max2.last().expect("at least one value was observed");
        let lo2 = self.min2.last().expect("at least one value was observed");
        debug_assert!(hi2 >= lo2, "the max is never below the min");
        let pff = Self::stochastic_stage(pf, hi2, lo2, &mut self.prev_f2, self.out);
        debug_assert!(pff.is_finite(), "`pff` must be finite");
        self.out = Some(pff);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    #[test]
    fn schaff_trend_cycle_plot() {
        let mut stc = SchaffTrendCycle::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            stc.update(*v);
            if let Some(val) = stc.last() {
                out.push(val);
            }
        }
        let filename = "img/schaff_trend_cycle.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn schaff_trend_cycle_warms_up() {
        let mut stc = SchaffTrendCycle::with_periods(
            Echo::new(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        let mut seen = false;
        for v in (0..60).map(|i| (i as f64 * 0.37).sin() * 10.0 + 50.0) {
            stc.update(v);
            if stc.last().is_some() {
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn schaff_trend_cycle_range() {
        let mut stc = SchaffTrendCycle::new(Echo::new());
        for v in &TEST_DATA {
            stc.update(*v);
            if let Some(val) = stc.last() {
                assert!((0.0..=100.0).contains(&val), "val = {val}");
            }
        }
    }

    #[test]
    fn schaff_trend_cycle_constant_input_holds_neutral() {
        let mut stc = SchaffTrendCycle::with_periods(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(5).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        for _ in 0..200 {
            stc.update(42.0);
        }
        // MACD is flat at 0, so both stochastic ranges are 0 and the raw
        // ratios fall back to their previous value, settling at 0.
        assert_eq!(stc.last(), Some(0.0));
    }

    #[test]
    #[should_panic(expected = "fast period must be shorter")]
    fn schaff_trend_cycle_fast_not_shorter_than_slow_panics() {
        let _ = SchaffTrendCycle::with_periods(
            Echo::<f64>::new(),
            NonZeroUsize::new(50).unwrap(),
            NonZeroUsize::new(50).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );
    }
}

//! RSX - Jurik "noise free" Relative Strength Index

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// RSX - Jurik's "noise free" Relative Strength Index.
///
/// A direct translation of the widely published open-source RSX recursion
/// (the `f8..f90` variable names below match the original; see e.g. the
/// ProRealTime "Jurik RSX" port at
/// <https://www.prorealcode.com/prorealtime-indicators/jurik-rsx/> and the
/// Pine Script port by `everget`/TradingView, both of which reproduce Mark
/// Jurik's published algorithm identically). It triple-smooths the 1-bar
/// momentum (and its absolute value) with a cascade of paired EMAs blended
/// as `1.5 * fast - 0.5 * slow` (a lag-reducing trick), then forms a ratio
/// of the two triple-smoothed series, scaled to `[0, 100]` like the RSI.
///
/// Deviation from this crate's usual warm-up convention: the published
/// recursion defines its own short, fixed (not `period`-dependent) internal
/// warm-up by outputting the neutral value `50` for the first few updates
/// (gated by the `f88`/`f90` bar-counting state below), so `last()` is
/// `Some` from the very first update once the chained `View` is ready,
/// unlike other indicators in this crate where `last()` is `None` during
/// warm-up.
///
/// Output range: `[0, 100]`.
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Rsx<T, V> {
    view: V,
    /// The smoothing period.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    alpha: T,
    ialpha: T,
    prev_f8: T,
    f28: T,
    f30: T,
    f38: T,
    f40: T,
    f48: T,
    f50: T,
    f58: T,
    f60: T,
    f68: T,
    f70: T,
    f78: T,
    f80: T,
    /// Bar-counting warm-up state (`f90_` in the original formula).
    f90_underscore: T,
    /// Bar-counting warm-up state (`f88` in the original formula).
    f88: T,
    out: Option<T>,
}

impl<T, V> Rsx<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new RSX with a chained `View` and a given smoothing period.
    pub fn new(view: V, period: NonZeroUsize) -> Self {
        let len = T::from(period.get()).expect("can convert");
        let alpha = T::from(3.0).expect("can convert") / (len + T::from(2.0).expect("can convert"));
        Self {
            view,
            period,
            alpha,
            ialpha: T::one() - alpha,
            prev_f8: T::zero(),
            f28: T::zero(),
            f30: T::zero(),
            f38: T::zero(),
            f40: T::zero(),
            f48: T::zero(),
            f50: T::zero(),
            f58: T::zero(),
            f60: T::zero(),
            f68: T::zero(),
            f70: T::zero(),
            f78: T::zero(),
            f80: T::zero(),
            f90_underscore: T::zero(),
            f88: T::zero(),
            out: None,
        }
    }

    /// One EMA-pair smoothing stage: `ema_fast = ialpha*prev_fast + alpha*x`,
    /// `ema_slow = alpha*ema_fast + ialpha*prev_slow`, output
    /// `1.5*ema_fast - 0.5*ema_slow` (Jurik's lag-reducing blend).
    fn smooth_stage(&self, prev_fast: T, prev_slow: T, x: T) -> (T, T, T) {
        let fast = self.ialpha * prev_fast + self.alpha * x;
        let slow = self.alpha * fast + self.ialpha * prev_slow;
        let half = T::from(0.5).expect("can convert");
        let blended = fast * (T::one() + half) - slow * half;
        (fast, slow, blended)
    }
}

impl<T, V> View<T> for Rsx<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let f8 = T::from(100.0).expect("can convert") * val;
        let f10 = self.prev_f8;
        let v8 = f8 - f10;
        self.prev_f8 = f8;

        let (f28, f30, v_c) = self.smooth_stage(self.f28, self.f30, v8);
        self.f28 = f28;
        self.f30 = f30;
        let (f38, f40, v10) = self.smooth_stage(self.f38, self.f40, v_c);
        self.f38 = f38;
        self.f40 = f40;
        let (f48, f50, v14) = self.smooth_stage(self.f48, self.f50, v10);
        self.f48 = f48;
        self.f50 = f50;

        let (f58, f60, v18) = self.smooth_stage(self.f58, self.f60, v8.abs());
        self.f58 = f58;
        self.f60 = f60;
        let (f68, f70, v1c) = self.smooth_stage(self.f68, self.f70, v18);
        self.f68 = f68;
        self.f70 = f70;
        let (f78, f80, v20) = self.smooth_stage(self.f78, self.f80, v1c);
        self.f78 = f78;
        self.f80 = f80;

        // Bar-counting warm-up state machine, verbatim from the published
        // recursion: `f88` settles at `max(period - 1, 5)` only transiently
        // and then at a fixed `5`; `f90_` counts up until it first exceeds
        // `f88`, which gates when the real ratio (rather than the neutral
        // `50`) is output below.
        let five = T::from(5.0).expect("can convert");
        let len_minus_one = T::from(self.period.get().saturating_sub(1)).expect("can convert");
        let prev_f90_underscore = self.f90_underscore;
        let prev_f88 = self.f88;
        self.f90_underscore = if prev_f90_underscore == T::zero() {
            T::one()
        } else if prev_f88 <= prev_f90_underscore {
            prev_f88 + T::one()
        } else {
            prev_f90_underscore + T::one()
        };
        self.f88 = if prev_f90_underscore == T::zero() && len_minus_one >= five {
            len_minus_one
        } else {
            five
        };
        let f0_is_set = self.f88 >= self.f90_underscore && f8 != f10;
        let f90 = if self.f88 == self.f90_underscore && !f0_is_set {
            T::zero()
        } else {
            self.f90_underscore
        };

        let fifty = T::from(50.0).expect("can convert");
        let rsx = if self.f88 < f90 && v20 > T::zero() {
            (v14 / v20 + T::one()) * fifty
        } else {
            fifty
        };
        let rsx = rsx.min(T::from(100.0).expect("can convert")).max(T::zero());
        debug_assert!(rsx.is_finite(), "`rsx` must be finite");
        self.out = Some(rsx);
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
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    #[test]
    fn rsx_plot() {
        let mut rsx = Rsx::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rsx.update(*v);
            if let Some(val) = rsx.last() {
                out.push(val);
            }
        }
        let filename = "img/rsx.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn rsx_is_some_from_the_first_update() {
        let mut rsx = Rsx::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        assert_eq!(rsx.last(), None);
        rsx.update(100.0);
        assert!(rsx.last().is_some());
    }

    #[test]
    fn rsx_range() {
        let mut rsx = Rsx::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        for v in &TEST_DATA {
            rsx.update(*v);
            if let Some(val) = rsx.last() {
                assert!((0.0..=100.0).contains(&val), "val = {val}");
            }
        }
    }

    #[test]
    fn rsx_constant_input_converges_to_neutral() {
        let mut rsx = Rsx::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        for _ in 0..200 {
            rsx.update(42.0);
        }
        // Zero momentum forever: both smoothed series stay at 0, so the
        // ratio is undefined and the formula falls back to the neutral 50.
        assert_eq!(rsx.last(), Some(50.0));
    }

    #[test]
    fn rsx_first_bars_are_neutral() {
        // The first few bars are always exactly 50, regardless of period,
        // because of the fixed (not period-scaled) `f88`/`f90` warm-up gate.
        let mut rsx = Rsx::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        for v in [10.0, 20.0, 15.0, 25.0] {
            rsx.update(v);
            assert_eq!(rsx.last(), Some(50.0));
        }
    }
}

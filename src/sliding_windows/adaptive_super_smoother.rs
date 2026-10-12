use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Largest rate of change in standard deviations, 2 in the paper.
const MAX_ROC: f64 = 2.0;

/// Shortest tuning period in bars, 2 in the paper.
const MIN_PERIOD: f64 = 2.0;

/// John Ehlers Adaptive SuperSmoother: a SuperSmoother whose critical period is
/// tuned in a closed loop by its own rate of change.
///
/// Reference: John Ehlers, "Adaptive SuperSmoother" (2025),
/// <https://www.mesasoftware.com/papers/Adaptive%20SuperSmoother.pdf>, Code Listings 1 to 3:
///
/// ```text
/// SS     = $SuperSmoother(Close, Period)      (Period of the previous bar)
/// ROC1   = SS - SS[1]
/// ROCRMS = $RMS(ROC1, 81)
/// If ROCRMS <> 0 Then ROC = |ROC1 / ROCRMS|
/// If ROC > 2 Then ROC = 2
/// Period = Period0 (1 - 0.5 ROC)^2,  at least 2
/// ```
///
/// with the 2025 SuperSmoother `Q = exp(-1.414 pi / P)`, `c1 = 2 Q cos(1.414 pi / P)`,
/// `c2 = Q^2`, `a0 = (1 - c1 + c2) / 2`, `SS = a0 (x + x[1]) + c1 SS[1] - c2 SS[2]`
/// (`SS = x` for the first 3 bars), whose coefficients are recomputed every bar.
/// A fast move (large ROC in standard deviations) shortens the period, so the filter
/// catches up with the price; in quiet markets it smooths with the full `Period0`.
///
/// Deviations: `ROC1` is 0 on the first bar (the paper would difference against 0),
/// and the period is only adapted once `rms_len` values of `ROC1` exist, so the RMS
/// is not computed over missing bars; until then `last()` is `None`.
/// The paper also plots the fixed `SuperSmoother(Close, Period0)` for comparison;
/// that is the plain [`SuperSmoother`](super::SuperSmoother) and not part of this view.
///
/// Output range: follows the input. Constant input is passed through up to rounding;
/// as the ROC is measured in standard deviations, the period then follows the rounding noise.
/// Complexity: O(rms_len) per update for the RMS (as in the paper), O(rms_len) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct AdaptiveSuperSmoother<T, V> {
    view: V,
    /// The longest tuning period `Period0`, used in quiet markets.
    #[getset(get_copy = "pub")]
    base_period: NonZeroUsize,
    /// The number of rates of change in the RMS.
    #[getset(get_copy = "pub")]
    rms_len: NonZeroUsize,
    /// Current tuning period.
    period: T,
    /// Input one bar ago.
    val_1: T,
    /// Filter output now and one bar ago.
    ss: [T; 2],
    /// Ring buffer of the last rates of change.
    rocs: Vec<T>,
    rms: T,
    roc: T,
    /// Number of values received from the inner view.
    n: usize,
}

impl<T, V> AdaptiveSuperSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with the longest tuning period `base_period` (`Period0`,
    /// 20 in the paper) and the RMS length `rms_len` of the rate of change (81 in the paper).
    ///
    /// # Panics
    /// If `base_period < 2`, as 2 bars is the shortest tuning period.
    pub fn new(view: V, base_period: NonZeroUsize, rms_len: NonZeroUsize) -> Self {
        assert!(base_period.get() >= 2, "base_period must be >= 2");
        Self {
            view,
            base_period,
            rms_len,
            period: T::from(base_period.get()).expect("can convert"),
            val_1: T::zero(),
            ss: [T::zero(); 2],
            rocs: vec![T::zero(); rms_len.get()],
            rms: T::zero(),
            roc: T::zero(),
            n: 0,
        }
    }

    /// The tuning period in bars for the next update, in `[2, base_period]`.
    /// `None` while warming up.
    pub fn period(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.period >= T::from(MIN_PERIOD).expect("can convert"));
        debug_assert!(self.period <= T::from(self.base_period.get()).expect("can convert"));
        Some(self.period)
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.n >= self.rms_len.get()
    }
}

impl<T, V> View<T> for AdaptiveSuperSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let ss = if self.n < 3 {
            val
        } else {
            super_smoother_step(self.period, val, self.val_1, self.ss)
        };
        debug_assert!(ss.is_finite(), "filter output must be finite");
        let roc1 = if self.n == 0 {
            T::zero()
        } else {
            ss - self.ss[0]
        };
        let len = self.rms_len.get();
        self.rocs[self.n % len] = roc1;
        self.n = self.n.saturating_add(1);
        self.val_1 = val;
        self.ss = [ss, self.ss[0]];
        if self.n < len {
            return;
        }

        let sum_sq = self.rocs.iter().fold(T::zero(), |s, &r| s + r * r);
        // As in the paper, the RMS and ROC keep their previous value if undefined.
        if sum_sq != T::zero() {
            self.rms = (sum_sq / T::from(len).expect("can convert")).sqrt();
        }
        if self.rms != T::zero() {
            self.roc = (roc1 / self.rms).abs();
        }
        self.roc = self.roc.min(T::from(MAX_ROC).expect("can convert"));
        let k = T::one() - T::from(0.5).expect("can convert") * self.roc;
        self.period = (T::from(self.base_period.get()).expect("can convert") * k * k)
            .max(T::from(MIN_PERIOD).expect("can convert"));
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.ss[0].is_finite(), "value must be finite");
        Some(self.ss[0])
    }
}

/// One step of the paper's 2025 SuperSmoother with critical `period`.
fn super_smoother_step<T: Float>(period: T, val: T, val_1: T, ss: [T; 2]) -> T {
    debug_assert!(period >= T::from(MIN_PERIOD).expect("can convert"));
    let angle = T::from(1.414 * PI).expect("can convert") / period;
    let q = (-angle).exp();
    let c1 = T::from(2.0).expect("can convert") * q * angle.cos();
    let c2 = q * q;
    let a0 = (T::one() - c1 + c2) / T::from(2.0).expect("can convert");
    a0 * (val + val_1) + c1 * ss[0] - c2 * ss[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::SuperSmoother,
        test_data::TEST_DATA,
    };

    fn ass(period: usize, rms_len: usize) -> AdaptiveSuperSmoother<f64, Echo<f64>> {
        AdaptiveSuperSmoother::new(
            Echo::new(),
            NonZeroUsize::new(period).unwrap(),
            NonZeroUsize::new(rms_len).unwrap(),
        )
    }

    #[test]
    fn adaptive_super_smoother_plot() {
        let mut f = ass(20, 81);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/adaptive_super_smoother.png").unwrap();
    }

    #[test]
    fn adaptive_super_smoother_warmup() {
        let mut f = ass(20, 81);
        assert_eq!(f.base_period().get(), 20);
        assert_eq!(f.rms_len().get(), 81);
        for i in 0..80 {
            f.update(i as f64);
            assert_eq!(f.last(), None);
            assert_eq!(f.period(), None);
        }
        f.update(80.0);
        assert!(f.last().is_some());
        assert!(f.period().is_some());
    }

    #[test]
    fn adaptive_super_smoother_constant_input() {
        let mut f = ass(20, 81);
        for _ in 0..200 {
            f.update(42.0);
        }
        assert!((f.last().unwrap() - 42.0).abs() < 1e-9);
        // The ROC is scale free, so the period follows the rounding noise of the filter.
        assert!((2.0..=20.0).contains(&f.period().unwrap()));
    }

    #[test]
    fn adaptive_super_smoother_period_in_range() {
        let mut f = AdaptiveSuperSmoother::<f32, _>::new(
            Echo::new(),
            NonZeroUsize::new(20).unwrap(),
            NonZeroUsize::new(81).unwrap(),
        );
        let rng = romu::Rng::from_seed_with_64bit(41);
        let mut price = 100.0;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for _ in 0..5_000 {
            price += rng.f32() - 0.5;
            f.update(price);
            if let Some(p) = f.period() {
                assert!((2.0..=20.0).contains(&p), "period {p}");
                assert!(f.last().unwrap().is_finite());
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        // The period does adapt over the whole range.
        assert!(lo < 5.0, "min period {lo}");
        assert!(hi > 19.0, "max period {hi}");
    }

    /// A sudden jump is a large rate of change, so the period shrinks
    /// and the adaptive filter catches up faster than the fixed one.
    #[test]
    fn adaptive_super_smoother_catches_up_after_step() {
        let mut adaptive = ass(20, 81);
        let mut fixed = SuperSmoother::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let rng = romu::Rng::from_seed_with_64bit(43);
        for i in 0..300 {
            let v = if i < 200 { 100.0 } else { 110.0 } + 0.1 * rng.f64();
            adaptive.update(v);
            fixed.update(v);
            if i == 202 {
                assert!(adaptive.period().unwrap() < 10.0);
            }
            if i == 205 {
                let err_adaptive = (adaptive.last().unwrap() - 110.0).abs();
                let err_fixed = (fixed.last().unwrap() - 110.0).abs();
                assert!(
                    err_adaptive < 0.5 * err_fixed,
                    "{err_adaptive} vs {err_fixed}"
                );
            }
        }
        assert!((adaptive.last().unwrap() - 110.05).abs() < 0.2);
    }

    /// Direct transcription of the paper's code with the same start-up.
    #[test]
    fn adaptive_super_smoother_matches_paper_transcription() {
        let (period0, rms_len) = (20.0, 81);
        let mut f = ass(20, rms_len);
        let (mut close, mut ss, mut roc1s) = (Vec::new(), Vec::new(), Vec::new());
        let (mut period, mut rms, mut roc) = (period0, 0.0, 0.0);
        let rng = romu::Rng::from_seed_with_64bit(47);
        let mut price = 100.0;
        for bar in 1..=1_000 {
            price += rng.f64() - 0.5;
            f.update(price);
            close.insert(0, price);
            let s = if bar >= 4 {
                let q = f64::exp(-1.414 * PI / period);
                let c1 = 2.0 * q * (1.414 * PI / period).cos();
                let c2 = q * q;
                let a0 = (1.0 - c1 + c2) / 2.0;
                a0 * (close[0] + close[1]) + c1 * ss[0] - c2 * ss[1]
            } else {
                close[0]
            };
            ss.insert(0, s);
            roc1s.insert(0, if bar == 1 { 0.0 } else { ss[0] - ss[1] });
            if roc1s.len() < rms_len {
                assert_eq!(f.last(), None);
                continue;
            }
            let sum_sq: f64 = roc1s[..rms_len].iter().map(|r| r * r).sum();
            if sum_sq != 0.0 {
                rms = (sum_sq / rms_len as f64).sqrt();
            }
            if rms != 0.0 {
                roc = (roc1s[0] / rms).abs();
            }
            roc = roc.min(2.0);
            period = f64::max(period0 * (1.0 - 0.5 * roc) * (1.0 - 0.5 * roc), 2.0);
            assert!((f.last().unwrap() - s).abs() < 1e-9, "bar {bar}");
            assert!((f.period().unwrap() - period).abs() < 1e-9, "bar {bar}");
        }
    }

    #[test]
    #[should_panic(expected = "base_period must be >= 2")]
    fn adaptive_super_smoother_period_too_small() {
        let _ = ass(1, 81);
    }
}

use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use num::{
    Float,
    ToPrimitive,
};

use super::{
    AutocorrelationAveraging,
    AutocorrelationPeriodogram,
};
use crate::View;

/// John Ehlers Adaptive RSI, tuned to the measured dominant cycle.
///
/// Reference: John Ehlers, "Cycle Analytics for Traders" (2013), chapter 11,
/// "Adaptive RSI". The dominant cycle `DC` is measured by an
/// [`AutocorrelationPeriodogram`], whose roofing filtered data `Filt` is also the RSI input:
///
/// ```text
/// ClosesUp = sum of  (Filt[i] - Filt[i+1]) where positive, i = 0 .. DC/2 - 1
/// ClosesDn = sum of -(Filt[i] - Filt[i+1]) where negative, i = 0 .. DC/2 - 1
/// Denom    = ClosesUp + ClosesDn
/// If Denom <> 0 and Denom[1] <> 0:
///     MyRSI = c1 (ClosesUp / Denom + ClosesUp[1] / Denom[1]) / 2 + c2 MyRSI[1] + c3 MyRSI[2]
/// ```
///
/// i.e. the RSI length is half the dominant cycle (`floor(DC / 2)` differences,
/// as the EasyLanguage loop runs to the fractional bound `DC / 2 - 1`),
/// smoothed by a SuperSmoother with length `min_period` (10 in the book).
/// Tuning the length to half a cycle makes the RSI swing fully over each cycle
/// while the roofing filter removes the trend and the noise.
///
/// Like the book, the output is a fraction in `[0, 1]` (not `[0, 100]`), with the
/// book's reference lines at 0.3 and 0.7. The SuperSmoother may overshoot the range
/// by a few percent after sharp reversals; the output is not clamped to stay faithful.
/// If `Denom` is zero (e.g. constant input) the RSI is held, so constant input yields
/// the book's initial value 0.
///
/// Deviations: those of the [`AutocorrelationPeriodogram`]; the RSI starts once the
/// periodogram is ready and `last()` is `None` for its first `min_period - 1` bars,
/// while the smoother settles.
///
/// Output range: `[0, 1]` (approximately, see above).
/// Complexity per update: O(L^2) for the periodogram (see [`AutocorrelationPeriodogram`])
/// plus O(L) for the RSI, with `L = max_period`.
#[derive(Debug, Clone)]
pub struct AdaptiveRsi<T, V> {
    periodogram: AutocorrelationPeriodogram<T, V>,
    /// SuperSmoother coefficients.
    c1: T,
    c2: T,
    c3: T,
    /// `ClosesUp / Denom` of the previous bar, `None` if `Denom` was zero.
    ratio_1: Option<T>,
    /// `MyRSI` of the latest bar.
    rsi: T,
    /// `MyRSI` one bar ago.
    rsi_1: T,
    /// Number of bars since the periodogram is ready.
    n: usize,
}

impl<T, V> AdaptiveRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with the book's parameters, see [`AutocorrelationPeriodogram::new`].
    pub fn new(view: V) -> Self {
        Self::from_periodogram(AutocorrelationPeriodogram::new(view))
    }

    /// Create a new instance whose dominant cycle is measured in `min_period..=max_period`,
    /// see [`AutocorrelationPeriodogram::with_params`].
    ///
    /// # Panics
    /// If the parameters are invalid for [`AutocorrelationPeriodogram::with_params`].
    pub fn with_params(
        view: V,
        min_period: NonZeroUsize,
        max_period: NonZeroUsize,
        averaging: AutocorrelationAveraging,
    ) -> Self {
        Self::from_periodogram(AutocorrelationPeriodogram::with_params(
            view, min_period, max_period, averaging,
        ))
    }

    fn from_periodogram(periodogram: AutocorrelationPeriodogram<T, V>) -> Self {
        let len = periodogram
            .min_period()
            .get()
            .to_f64()
            .expect("can convert");
        let a1 = (-1.414 * PI / len).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / len).cos();
        let (c2, c3) = (b1, -a1 * a1);
        let c1 = 1.0 - c2 - c3;
        Self {
            periodogram,
            c1: T::from(c1).expect("can convert"),
            c2: T::from(c2).expect("can convert"),
            c3: T::from(c3).expect("can convert"),
            ratio_1: None,
            rsi: T::zero(),
            rsi_1: T::zero(),
            n: 0,
        }
    }

    /// The dominant cycle period in bars that tunes the RSI length.
    /// `None` while warming up.
    #[inline]
    pub fn dominant_cycle(&self) -> Option<T> {
        self.periodogram.last()
    }
}

impl<T, V> View<T> for AdaptiveRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.periodogram.update(val);
        let Some(dc) = self.periodogram.last() else {
            return;
        };
        let hist = self
            .periodogram
            .filtered_history()
            .expect("history is ready when the periodogram is");
        let len = (dc / T::from(2.0).expect("can convert"))
            .floor()
            .to_usize()
            .expect("dominant cycle is finite and positive");
        assert!(len >= 1, "RSI length must be >= 1");
        assert!(len < hist.len(), "history must cover the RSI length");

        let (up, dn) = closes_up_down(&hist[..=len]);
        let denom = up + dn;
        let ratio = (denom != T::zero()).then(|| up / denom);
        let rsi = match (ratio, self.ratio_1) {
            (Some(r), Some(r_1)) => {
                self.c1 * (r + r_1) / T::from(2.0).expect("can convert")
                    + self.c2 * self.rsi
                    + self.c3 * self.rsi_1
            }
            _ => self.rsi,
        };
        debug_assert!(rsi.is_finite(), "value must be finite");
        self.ratio_1 = ratio;
        self.rsi_1 = self.rsi;
        self.rsi = rsi;
        self.n = self.n.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < self.periodogram.min_period().get() {
            return None;
        }
        debug_assert!(self.rsi.is_finite(), "value must be finite");
        Some(self.rsi)
    }
}

/// Sums of the up and down moves in `filt` (newest first).
fn closes_up_down<T: Float>(filt: &[T]) -> (T, T) {
    let (mut up, mut dn) = (T::zero(), T::zero());
    for w in filt.windows(2) {
        let diff = w[0] - w[1];
        if diff > T::zero() {
            up = up + diff;
        } else {
            dn = dn - diff;
        }
    }
    debug_assert!(up >= T::zero(), "up moves must be >= 0");
    debug_assert!(dn >= T::zero(), "down moves must be >= 0");
    (up, dn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn sine(i: usize, period: f64) -> f64 {
        100.0 + 10.0 * (2.0 * PI * i as f64 / period).sin()
    }

    #[test]
    fn adaptive_rsi_plot() {
        let mut rsi = AdaptiveRsi::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            rsi.update(*v);
            if let Some(val) = rsi.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/adaptive_rsi.png").unwrap();
    }

    #[test]
    fn adaptive_rsi_warmup() {
        let mut rsi = AdaptiveRsi::new(Echo::new());
        // The periodogram is ready after 107 values, then the smoother needs 10.
        let warmup = 107 + 10 - 1;
        for i in 0..warmup - 1 {
            rsi.update(sine(i, 20.0));
            assert_eq!(rsi.last(), None);
            assert_eq!(rsi.dominant_cycle().is_some(), i >= 106);
        }
        rsi.update(sine(warmup - 1, 20.0));
        assert!(rsi.last().is_some());
    }

    #[test]
    fn adaptive_rsi_tracks_cycle_and_swings() {
        for period in [15.0, 20.0, 30.0] {
            let mut rsi = AdaptiveRsi::new(Echo::new());
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for i in 0..800 {
                rsi.update(sine(i, period));
                if i > 400 {
                    let v = rsi.last().unwrap();
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let dc = rsi.dominant_cycle().unwrap();
            assert!(
                (dc - period).abs() < 0.1 * period,
                "period {period}: dc {dc}"
            );
            // Half a cycle of rising data gives an RSI near 1, falling data near 0.
            assert!(hi > 0.8, "period {period}: max {hi}");
            assert!(lo < 0.2, "period {period}: min {lo}");
        }
    }

    /// The output is bounded by the range of the SuperSmoother's impulse response,
    /// as its input is in `[0, 1]`.
    #[test]
    fn adaptive_rsi_range() {
        let rsi = AdaptiveRsi::<f64, _>::new(Echo::new());
        let (mut pos, mut neg) = (0.0, 0.0);
        let (mut y1, mut y2) = (0.0, 0.0);
        for i in 0..1_000 {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let y = rsi.c1 * x + rsi.c2 * y1 + rsi.c3 * y2;
            (y2, y1) = (y1, y);
            if y > 0.0 { pos += y } else { neg += y }
        }
        let mut rsi = rsi;
        let rng = romu::Rng::from_seed_with_64bit(17);
        let mut price = 100.0;
        let mut inside = 0;
        for _ in 0..5_000 {
            price += rng.f64() - 0.5;
            rsi.update(price);
            if let Some(v) = rsi.last() {
                assert!(v >= neg - 1e-12, "rsi {v} below {neg}");
                assert!(v <= pos + 1e-12, "rsi {v} above {pos}");
                inside += usize::from((0.0..=1.0).contains(&v));
            }
        }
        assert!(inside > 4_500, "only {inside} values in [0, 1]");
    }

    #[test]
    fn adaptive_rsi_constant_input() {
        let mut rsi = AdaptiveRsi::new(Echo::new());
        for _ in 0..300 {
            rsi.update(42.0);
        }
        assert_eq!(rsi.last(), Some(0.0));
        assert_eq!(rsi.dominant_cycle(), Some(10.0));
    }

    /// Direct transcription of the book's RSI loop, fed by a second periodogram.
    #[test]
    fn adaptive_rsi_matches_book_transcription() {
        let mut rsi = AdaptiveRsi::new(Echo::new());
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        let a1 = (-1.414 * PI / 10.0).exp();
        let c2 = 2.0 * a1 * (1.414 * PI / 10.0).cos();
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;
        let (mut up_1, mut denom_1, mut my_rsi, mut my_rsi_1) = (0.0, 0.0, 0.0, 0.0);
        let rng = romu::Rng::from_seed_with_64bit(23);
        let mut price = 100.0;
        let mut checked = 0;
        for _ in 0..1_000 {
            price += rng.f64() - 0.5;
            rsi.update(price);
            ap.update(price);
            let Some(dc) = ap.last() else { continue };
            let filt = ap.filtered_history().unwrap();
            let (mut up, mut dn) = (0.0, 0.0);
            let mut count = 0.0;
            while count <= dc / 2.0 - 1.0 {
                let i = count as usize;
                if filt[i] > filt[i + 1] {
                    up += filt[i] - filt[i + 1];
                }
                if filt[i] < filt[i + 1] {
                    dn += filt[i + 1] - filt[i];
                }
                count += 1.0;
            }
            let denom = up + dn;
            let mut new_rsi = my_rsi;
            if denom != 0.0 && denom_1 != 0.0 {
                new_rsi = c1 * (up / denom + up_1 / denom_1) / 2.0 + c2 * my_rsi + c3 * my_rsi_1;
            }
            (up_1, denom_1, my_rsi_1, my_rsi) = (up, denom, my_rsi, new_rsi);
            if let Some(v) = rsi.last() {
                assert!((v - my_rsi).abs() < 1e-12, "{v} != {my_rsi}");
                assert_eq!(rsi.dominant_cycle(), Some(dc));
                checked += 1;
            }
        }
        assert!(checked > 800);
    }

    #[test]
    #[should_panic(expected = "max_period must be greater than min_period")]
    fn adaptive_rsi_invalid_periods() {
        let _ = AdaptiveRsi::<f64, _>::with_params(
            Echo::new(),
            NonZeroUsize::new(20).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            AutocorrelationAveraging::Fixed(NonZeroUsize::new(3).unwrap()),
        );
    }
}

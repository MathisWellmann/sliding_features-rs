use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::{
    Float,
    ToPrimitive,
};

use super::{
    HighPass,
    HighPassPoles,
    SuperSmoother,
};
use crate::View;

/// Smallest lag used in the discrete Fourier transform of the correlations.
/// Ehlers starts at 3 because the shortest lags are dominated by the smoothing
/// of the roofing filter itself rather than by a market cycle.
const FIRST_DFT_LAG: usize = 3;

/// The number of bars over which the Pearson correlation of each lag is computed
/// in an [`AutocorrelationPeriodogram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocorrelationAveraging {
    /// A fixed number of bars, which must be at least 2.
    /// The book uses 3 (`AvgLength = 3`), which gives the most responsive measurement.
    Fixed(NonZeroUsize),
    /// Use the lag itself as the averaging length (`AvgLength = 0` in the book code).
    /// Smoother but slower, and the history grows to twice the maximum period.
    Lag,
}

/// John Ehlers Autocorrelation Periodogram, measuring the dominant cycle period in bars.
///
/// Reference: John Ehlers, "Cycle Analytics for Traders" (2013), chapter 8,
/// "Autocorrelation Periodogram" (EasyLanguage code of figure 8.3):
///
/// 1. Roofing filter: two-pole [`HighPass`] with cutoff `max_period` (48 in the book),
///    then a [`SuperSmoother`] with length `min_period` (10 in the book).
/// 2. Pearson correlation `Corr[lag]` of the filtered series with its lagged copy,
///    for every lag `0..=max_period`, averaged over [`AutocorrelationAveraging`] bars.
///    If a correlation is undefined (zero variance), the previous value is kept, as in the book.
/// 3. For every period `P` in `min_period..=max_period`:
///    `SqSum[P] = (sum_n Corr[n] cos(2 pi n / P))^2 + (sum_n Corr[n] sin(2 pi n / P))^2`
///    over the lags `n = 3..=max_period`.
/// 4. Smoothing `R[P] = 0.2 SqSum[P]^2 + 0.8 R[P]`.
/// 5. Automatic gain control: `MaxPwr = max(0.995 MaxPwr, max_P R[P])`,
///    normalized power `Pwr[P] = R[P] / MaxPwr`.
/// 6. Dominant cycle = center of gravity of the periods whose `Pwr >= 0.5`,
///    clamped to `[min_period, max_period]`.
///
/// The book hard-codes the period range 10..=48 and ties the filter lengths and
/// the maximum lag to it; here these follow `min_period` and `max_period`.
///
/// Deviations: the high-pass filter starts at rest (see [`HighPass`]), and the
/// correlation stage only starts once the filtered history covers all lags,
/// instead of reading not yet existing bars as zero.
/// If no period has power (e.g. constant input), the previous dominant cycle is kept;
/// before any power was measured it is `min_period` (the book's initial 0, clamped).
///
/// Output range: `[min_period, max_period]` bars.
/// Complexity per update: O(L * M) for the correlations plus O(L * (max_period - min_period))
/// for the spectrum, i.e. O(L^2) with the maximum lag `L = max_period` and averaging length `M`.
/// Memory: O(L * (max_period - min_period)) for the precomputed cosine and sine tables.
#[derive(Debug, Clone, CopyGetters)]
pub struct AutocorrelationPeriodogram<T, V> {
    /// Roofing filter, with the inner view chained into the high-pass filter.
    filter: SuperSmoother<T, HighPass<T, V>>,
    /// The shortest measurable cycle period in bars.
    #[getset(get_copy = "pub")]
    min_period: NonZeroUsize,
    /// The longest measurable cycle period in bars, also the maximum lag.
    #[getset(get_copy = "pub")]
    max_period: NonZeroUsize,
    /// The averaging length of the correlations.
    #[getset(get_copy = "pub")]
    averaging: AutocorrelationAveraging,
    /// Filtered values, written twice (at `pos` and `pos + cap`),
    /// so that `history[pos..pos + cap]` is always a contiguous, newest first view.
    history: Vec<T>,
    cap: usize,
    pos: usize,
    /// Number of filtered values received, saturating at `cap`.
    n_filt: usize,
    /// Correlation per lag `0..=max_period`.
    corr: Vec<T>,
    /// `cos(2 pi n / P)` for each period (rows) and DFT lag (columns).
    cos_table: Vec<T>,
    /// `sin(2 pi n / P)` for each period (rows) and DFT lag (columns).
    sin_table: Vec<T>,
    /// Smoothed power per period.
    r: Vec<T>,
    /// Normalized power per period, in `[0, 1]`.
    pwr: Vec<T>,
    max_pwr: T,
    dominant_cycle: T,
}

impl<T, V> AutocorrelationPeriodogram<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with the book's parameters:
    /// periods `10..=48` and an averaging length of 3 bars.
    pub fn new(view: V) -> Self {
        Self::with_params(
            view,
            NonZeroUsize::new(10).expect("is non zero"),
            NonZeroUsize::new(48).expect("is non zero"),
            AutocorrelationAveraging::Fixed(NonZeroUsize::new(3).expect("is non zero")),
        )
    }

    /// Create a new instance measuring cycles in `min_period..=max_period`.
    ///
    /// # Panics
    /// - If `min_period < 3`, as the smoother and the DFT need at least 3 bars.
    /// - If `max_period <= min_period`.
    /// - If `averaging` is `Fixed(m)` with `m < 2`, as a correlation needs two points.
    pub fn with_params(
        view: V,
        min_period: NonZeroUsize,
        max_period: NonZeroUsize,
        averaging: AutocorrelationAveraging,
    ) -> Self {
        assert!(min_period.get() >= 3, "min_period must be >= 3");
        assert!(
            max_period > min_period,
            "max_period must be greater than min_period"
        );
        let max_lag = max_period.get();
        let max_avg = match averaging {
            AutocorrelationAveraging::Fixed(m) => {
                assert!(m.get() >= 2, "averaging length must be >= 2");
                m.get()
            }
            AutocorrelationAveraging::Lag => max_lag,
        };
        let cap = max_lag + max_avg;
        let n_periods = max_period.get() - min_period.get() + 1;
        let n_lags = max_lag + 1 - FIRST_DFT_LAG;
        let mut cos_table = Vec::with_capacity(n_periods * n_lags);
        let mut sin_table = Vec::with_capacity(n_periods * n_lags);
        for period in min_period.get()..=max_period.get() {
            for lag in FIRST_DFT_LAG..=max_lag {
                let angle = 2.0 * PI * lag.to_f64().expect("can convert")
                    / period.to_f64().expect("can convert");
                cos_table.push(T::from(angle.cos()).expect("can convert"));
                sin_table.push(T::from(angle.sin()).expect("can convert"));
            }
        }
        let filter = SuperSmoother::new(
            HighPass::new(view, max_period, HighPassPoles::Two),
            min_period,
        );

        Self {
            filter,
            min_period,
            max_period,
            averaging,
            history: vec![T::zero(); 2 * cap],
            cap,
            pos: 0,
            n_filt: 0,
            corr: vec![T::zero(); max_lag + 1],
            cos_table,
            sin_table,
            r: vec![T::zero(); n_periods],
            pwr: vec![T::zero(); n_periods],
            max_pwr: T::zero(),
            dominant_cycle: T::from(min_period.get()).expect("can convert"),
        }
    }

    /// The normalized power of the cycle with `period` bars, in `[0, 1]`.
    /// `None` while warming up or if `period` is outside `min_period..=max_period`.
    pub fn power(&self, period: usize) -> Option<T> {
        if !self.is_ready() || period < self.min_period.get() || period > self.max_period.get() {
            return None;
        }
        let p = self.pwr[period - self.min_period.get()];
        debug_assert!(p >= T::zero(), "power must be >= 0");
        debug_assert!(p <= T::one(), "power must be <= 1");
        Some(p)
    }

    /// The roofing filtered history, newest first, once all lags are covered.
    /// Shared with the indicators that are tuned by this periodogram,
    /// so they do not need a second roofing filter.
    pub(super) fn filtered_history(&self) -> Option<&[T]> {
        if !self.is_ready() {
            return None;
        }
        Some(&self.history[self.pos..self.pos + self.cap])
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.n_filt >= self.cap
    }
}

impl<T, V> View<T> for AutocorrelationPeriodogram<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.filter.update(val);
        let Some(filt) = self.filter.last() else {
            return;
        };
        debug_assert!(filt.is_finite(), "value must be finite");

        // Move backwards, so the newest value is always first in the slice.
        self.pos = if self.pos == 0 {
            self.cap - 1
        } else {
            self.pos - 1
        };
        self.history[self.pos] = filt;
        self.history[self.pos + self.cap] = filt;
        self.n_filt = (self.n_filt + 1).min(self.cap);
        if !self.is_ready() {
            return;
        }

        let hist = &self.history[self.pos..self.pos + self.cap];
        correlations(hist, self.averaging, &mut self.corr);
        smoothed_spectrum(&self.corr, &self.cos_table, &self.sin_table, &mut self.r);
        // Automatic gain control, so the power is comparable across market regimes.
        self.max_pwr = self.r.iter().fold(
            T::from(0.995).expect("can convert") * self.max_pwr,
            |m, &r| m.max(r),
        );
        normalize(&self.r, self.max_pwr, &mut self.pwr);
        if let Some(dc) = center_of_gravity(&self.pwr, self.min_period.get()) {
            self.dominant_cycle = dc;
        }
        let min = T::from(self.min_period.get()).expect("can convert");
        let max = T::from(self.max_period.get()).expect("can convert");
        self.dominant_cycle = self.dominant_cycle.max(min).min(max);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        let dc = self.dominant_cycle;
        debug_assert!(dc.is_finite(), "value must be finite");
        debug_assert!(dc >= T::from(self.min_period.get()).expect("can convert"));
        debug_assert!(dc <= T::from(self.max_period.get()).expect("can convert"));
        Some(dc)
    }
}

/// Pearson correlation of `x` and `y`, `None` if either has zero variance.
fn pearson<T: Float>(x: &[T], y: &[T]) -> Option<T> {
    debug_assert_eq!(x.len(), y.len(), "slices must have the same length");
    let m = T::from(x.len()).expect("can convert");
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) =
        (T::zero(), T::zero(), T::zero(), T::zero(), T::zero());
    for (&x, &y) in x.iter().zip(y) {
        sx = sx + x;
        sy = sy + y;
        sxx = sxx + x * x;
        syy = syy + y * y;
        sxy = sxy + x * y;
    }
    let denom = (m * sxx - sx * sx) * (m * syy - sy * sy);
    (denom > T::zero()).then(|| (m * sxy - sx * sy) / denom.sqrt())
}

/// Correlation of `hist` (newest first) with its lagged copy for every lag in `0..corr.len()`.
/// Undefined correlations keep their previous value, as in the book.
fn correlations<T: Float>(hist: &[T], averaging: AutocorrelationAveraging, corr: &mut [T]) {
    for (lag, c) in corr.iter_mut().enumerate() {
        let m = match averaging {
            AutocorrelationAveraging::Fixed(m) => m.get(),
            AutocorrelationAveraging::Lag => lag,
        };
        debug_assert!(lag + m <= hist.len(), "history must cover all lags");
        if let Some(r) = pearson(&hist[..m], &hist[lag..lag + m]) {
            *c = r;
        }
    }
}

/// DFT of the correlations for every period (row of the tables),
/// smoothed into `r` with `R = 0.2 SqSum^2 + 0.8 R`.
fn smoothed_spectrum<T: Float>(corr: &[T], cos_table: &[T], sin_table: &[T], r: &mut [T]) {
    let lags = &corr[FIRST_DFT_LAG..];
    let n = lags.len();
    debug_assert_eq!(cos_table.len(), n * r.len(), "cos table must match");
    debug_assert_eq!(sin_table.len(), n * r.len(), "sin table must match");
    let (a, b) = (
        T::from(0.2).expect("can convert"),
        T::from(0.8).expect("can convert"),
    );
    for ((r, cos_row), sin_row) in r
        .iter_mut()
        .zip(cos_table.chunks_exact(n))
        .zip(sin_table.chunks_exact(n))
    {
        let (mut cos_part, mut sin_part) = (T::zero(), T::zero());
        for ((&c, &cs), &sn) in lags.iter().zip(cos_row).zip(sin_row) {
            cos_part = cos_part + c * cs;
            sin_part = sin_part + c * sn;
        }
        let sq_sum = cos_part * cos_part + sin_part * sin_part;
        *r = a * sq_sum * sq_sum + b * *r;
    }
}

/// `pwr = r / max_pwr`, or zero if there is no power at all.
fn normalize<T: Float>(r: &[T], max_pwr: T, pwr: &mut [T]) {
    debug_assert!(max_pwr >= T::zero(), "max power must be >= 0");
    for (p, &r) in pwr.iter_mut().zip(r) {
        *p = if max_pwr > T::zero() {
            r / max_pwr
        } else {
            T::zero()
        };
    }
}

/// Center of gravity of the periods with at least half of the maximum power.
fn center_of_gravity<T: Float>(pwr: &[T], min_period: usize) -> Option<T> {
    let half = T::from(0.5).expect("can convert");
    let (mut spx, mut sp) = (T::zero(), T::zero());
    for (i, &p) in pwr.iter().enumerate() {
        if p >= half {
            spx = spx + T::from(min_period + i).expect("can convert") * p;
            sp = sp + p;
        }
    }
    (sp != T::zero()).then(|| spx / sp)
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

    /// Dominant cycle after `n` bars of a sine with `period`.
    fn measure(ap: &mut AutocorrelationPeriodogram<f64, Echo<f64>>, period: f64, n: usize) -> f64 {
        for i in 0..n {
            ap.update(sine(i, period));
        }
        ap.last().unwrap()
    }

    #[test]
    fn autocorrelation_periodogram_plot() {
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            ap.update(*v);
            if let Some(val) = ap.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/autocorrelation_periodogram.png").unwrap();
    }

    #[test]
    fn autocorrelation_periodogram_warmup() {
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        assert_eq!(ap.min_period().get(), 10);
        assert_eq!(ap.max_period().get(), 48);
        // High-pass needs 48 values, the SuperSmoother 10, the history 48 + 3.
        let warmup = 48 + 10 + 51 - 2;
        for i in 0..warmup - 1 {
            ap.update(sine(i, 20.0));
            assert_eq!(ap.last(), None);
            assert_eq!(ap.power(20), None);
        }
        ap.update(sine(warmup - 1, 20.0));
        assert!(ap.last().is_some());
    }

    #[test]
    fn autocorrelation_periodogram_measures_sine_period() {
        for period in [15.0, 20.0, 30.0, 40.0] {
            let mut ap = AutocorrelationPeriodogram::new(Echo::new());
            let dc = measure(&mut ap, period, 600);
            assert!(
                (dc - period).abs() < 0.1 * period,
                "period {period}: measured {dc}"
            );
            // The spectrum peaks at the input period and is low far away from it.
            assert!(ap.power(period as usize).unwrap() > 0.5);
            assert!(ap.power(10).unwrap() < 0.5);
        }
    }

    #[test]
    fn autocorrelation_periodogram_lag_averaging_measures_sine_period() {
        for period in [15.0, 25.0] {
            let mut ap = AutocorrelationPeriodogram::with_params(
                Echo::new(),
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(48).unwrap(),
                AutocorrelationAveraging::Lag,
            );
            let dc = measure(&mut ap, period, 800);
            assert!(
                (dc - period).abs() < 0.1 * period,
                "period {period}: measured {dc}"
            );
        }
    }

    #[test]
    fn autocorrelation_periodogram_follows_period_change() {
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        let _ = measure(&mut ap, 15.0, 400);
        for i in 400..800 {
            ap.update(sine(i, 35.0));
        }
        let dc = ap.last().unwrap();
        assert!((dc - 35.0).abs() < 3.5, "measured {dc}");
    }

    #[test]
    fn autocorrelation_periodogram_constant_input() {
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        for _ in 0..300 {
            ap.update(42.0);
        }
        assert_eq!(ap.last(), Some(10.0));
        assert_eq!(ap.power(20), Some(0.0));
    }

    #[test]
    fn autocorrelation_periodogram_range_and_power_bounds() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let mut ap = AutocorrelationPeriodogram::<f64, _>::new(Echo::new());
        let mut price = 100.0;
        for _ in 0..3_000 {
            price += rng.f64() - 0.5;
            ap.update(price);
            if let Some(dc) = ap.last() {
                assert!((10.0..=48.0).contains(&dc), "dc {dc}");
                for p in 10..=48 {
                    let pwr = ap.power(p).unwrap();
                    assert!((0.0..=1.0).contains(&pwr), "power {pwr}");
                }
            }
        }
        assert_eq!(ap.power(9), None);
        assert_eq!(ap.power(49), None);
    }

    /// Direct transcription of the book's loops on a growing `Vec`, fed with the same filter.
    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "indexing mirrors the book's EasyLanguage code"
    )]
    fn autocorrelation_periodogram_matches_book_transcription() {
        let mut ap = AutocorrelationPeriodogram::new(Echo::new());
        let mut filter = SuperSmoother::new(
            HighPass::new(
                Echo::new(),
                NonZeroUsize::new(48).unwrap(),
                HighPassPoles::Two,
            ),
            NonZeroUsize::new(10).unwrap(),
        );
        let mut filt: Vec<f64> = Vec::new();
        let (mut corr, mut r, mut pwr) = ([0.0; 49], [0.0; 49], [0.0; 49]);
        let (mut max_pwr, mut dc) = (0.0, 0.0);
        let rng = romu::Rng::from_seed_with_64bit(11);
        let mut price = 100.0;
        for bar in 0..1_000 {
            price += rng.f64() - 0.5;
            ap.update(price);
            filter.update(price);
            let Some(f) = filter.last() else { continue };
            filt.insert(0, f);
            if filt.len() < 51 {
                assert_eq!(ap.last(), None);
                continue;
            }
            for lag in 0..=48 {
                let m = 3.0;
                let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for count in 0..3 {
                    let (x, y) = (filt[count], filt[lag + count]);
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }
                if (m * sxx - sx * sx) * (m * syy - sy * sy) > 0.0 {
                    corr[lag] =
                        (m * sxy - sx * sy) / ((m * sxx - sx * sx) * (m * syy - sy * sy)).sqrt();
                }
            }
            for period in 10..=48 {
                let (mut cp, mut sp) = (0.0, 0.0);
                for n in 3..=48 {
                    let a = 2.0 * PI * n as f64 / period as f64;
                    cp += corr[n] * a.cos();
                    sp += corr[n] * a.sin();
                }
                let sq = cp * cp + sp * sp;
                r[period] = 0.2 * sq * sq + 0.8 * r[period];
            }
            max_pwr *= 0.995;
            for period in 10..=48 {
                if r[period] > max_pwr {
                    max_pwr = r[period];
                }
            }
            for period in 10..=48 {
                pwr[period] = r[period] / max_pwr;
            }
            let (mut spx, mut sp) = (0.0, 0.0);
            for period in 10..=48 {
                if pwr[period] >= 0.5 {
                    spx += period as f64 * pwr[period];
                    sp += pwr[period];
                }
            }
            if sp != 0.0 {
                dc = spx / sp;
            }
            dc = f64::clamp(dc, 10.0, 48.0);
            let got = ap.last().unwrap();
            assert!((got - dc).abs() < 1e-9, "bar {bar}: {got} != {dc}");
            assert!((ap.power(20).unwrap() - pwr[20]).abs() < 1e-9);
        }
    }

    #[test]
    fn autocorrelation_periodogram_f32() {
        let mut ap = AutocorrelationPeriodogram::<f32, _>::new(Echo::new());
        for i in 0..600 {
            ap.update((100.0 + 10.0 * (2.0 * PI * i as f64 / 20.0).sin()) as f32);
        }
        let dc = ap.last().unwrap();
        assert!((dc - 20.0).abs() < 2.0, "measured {dc}");
    }

    #[test]
    #[should_panic(expected = "min_period must be >= 3")]
    fn autocorrelation_periodogram_min_period_too_small() {
        let _ = AutocorrelationPeriodogram::<f64, _>::with_params(
            Echo::new(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(48).unwrap(),
            AutocorrelationAveraging::Lag,
        );
    }

    #[test]
    #[should_panic(expected = "max_period must be greater than min_period")]
    fn autocorrelation_periodogram_max_not_above_min() {
        let _ = AutocorrelationPeriodogram::<f64, _>::with_params(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            AutocorrelationAveraging::Lag,
        );
    }

    #[test]
    #[should_panic(expected = "averaging length must be >= 2")]
    fn autocorrelation_periodogram_averaging_too_short() {
        let _ = AutocorrelationPeriodogram::<f64, _>::with_params(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(48).unwrap(),
            AutocorrelationAveraging::Fixed(NonZeroUsize::new(1).unwrap()),
        );
    }
}

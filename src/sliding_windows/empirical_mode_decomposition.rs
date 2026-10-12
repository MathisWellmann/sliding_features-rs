use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use super::Sma;
use crate::{
    View,
    pure_functions::Echo,
};

/// Number of bars over which peaks and valleys are averaged, fixed in the paper.
const PEAK_AVG_LEN: usize = 50;

/// John Ehlers and Ric Way Empirical Mode Decomposition (EMD).
///
/// Splits the input into a cycle component (a band-pass around `period`) and
/// a trend component (the average of the band-pass over two periods), and
/// compares the trend with a fraction of the average cycle peaks and valleys
/// to decide whether the market is trending or cycling.
///
/// Reference: John Ehlers and Ric Way, "Empirical Mode Decomposition",
/// Technical Analysis of Stocks & Commodities, March 2010,
/// <https://mesasoftware.com/papers/EmpiricalModeDecomposition.pdf> (Figure 9):
///
/// ```text
/// beta  = cos(2pi / P), gamma = 1 / cos(4pi delta / P), alpha = gamma - sqrt(gamma^2 - 1)
/// BP    = 0.5 (1 - alpha) (x - x[2]) + beta (1 + alpha) BP[1] - alpha BP[2]
/// Mean  = SMA(BP, 2P)
/// Peak  = BP[1] if BP[1] > BP and BP[1] > BP[2], else Peak[1]
/// Valley = BP[1] if BP[1] < BP and BP[1] < BP[2], else Valley[1]
/// upper = Fraction SMA(Peak, 50), lower = Fraction SMA(Valley, 50)
/// ```
///
/// `delta` is the half-bandwidth of the paper (Ehlers uses 0.5 with
/// `Fraction` 0.1 to 0.25). The band-pass is the same filter as
/// [`BandPass`](super::BandPass) with `bandwidth = 2 delta`, but computed here,
/// because `BandPass` requires `bandwidth < 1` and so cannot use the default
/// `delta = 0.5`; a test checks both agree for `delta < 0.5`.
///
/// The band-pass starts at rest (input history seeded with the first value),
/// replacing EasyLanguage's zero initialization; `Peak` and `Valley` start at 0.
/// `last()` returns `Mean`, the trend component, and is `None` for the first
/// `max(2 period, 50) - 1` values. See [`upper_threshold`](Self::upper_threshold),
/// [`lower_threshold`](Self::lower_threshold), [`band_pass`](Self::band_pass)
/// and [`trend_mode`](Self::trend_mode) for the other outputs.
///
/// Output range: unbounded, oscillates around 0 (in input units).
/// Constant input yields 0.
/// Complexity: O(1) per update, O(period) memory (moving average windows).
#[derive(Debug, Clone, CopyGetters)]
pub struct EmpiricalModeDecomposition<T, V> {
    view: V,
    /// The center period of the band-pass.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    delta: T,
    fraction: T,
    /// Gain applied to `x - x[2]`, `0.5 (1 - alpha)`.
    c_in: T,
    /// Feedback coefficient of `BP[1]`, `beta (1 + alpha)`.
    c_bp_1: T,
    /// Feedback coefficient of `BP[2]`, `-alpha`.
    c_bp_2: T,
    /// Whether the input history has been seeded.
    started: bool,
    x_1: T,
    x_2: T,
    bp: T,
    bp_1: T,
    bp_2: T,
    peak: T,
    valley: T,
    mean: Sma<T, Echo<T>>,
    avg_peak: Sma<T, Echo<T>>,
    avg_valley: Sma<T, Echo<T>>,
}

impl<T, V> EmpiricalModeDecomposition<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Empirical Mode Decomposition with a chained `View`.
    ///
    /// Ehlers uses `period = 20`, `delta = 0.5`, `fraction = 0.1`.
    ///
    /// # Panics
    /// - If `period < 2`, as shorter periods are above the Nyquist frequency.
    /// - If `delta` is not positive or not below `period / 8`
    ///   (`cos(4pi delta / P)` must be positive for a stable filter).
    /// - If `fraction` is not positive and finite.
    pub fn new(view: V, period: NonZeroUsize, delta: T, fraction: T) -> Self {
        assert!(period.get() >= 2, "period must be >= 2");
        assert!(delta > T::zero(), "delta must be > 0");
        let p = T::from(period.get()).expect("can convert");
        assert!(
            T::from(8.0).expect("can convert") * delta < p,
            "delta must be < period / 8"
        );
        assert!(fraction.is_finite(), "fraction must be finite");
        assert!(fraction > T::zero(), "fraction must be > 0");
        let angle = T::from(2.0 * PI).expect("can convert") / p;
        let theta = T::from(2.0).expect("can convert") * delta * angle;
        // Algebraically equal to `gamma - sqrt(gamma^2 - 1)` with `gamma = 1 / cos(theta)`,
        // without the cancellation of two large terms when cos(theta) is close to 0.
        let alpha = theta.cos() / (T::one() + theta.sin());
        assert!(alpha > T::zero(), "alpha must be positive");
        assert!(
            alpha < T::one(),
            "alpha must be below 1 for a stable filter"
        );
        let two_p = NonZeroUsize::new(2 * period.get()).expect("period is non-zero");
        let avg_len = NonZeroUsize::new(PEAK_AVG_LEN).expect("constant is non-zero");
        Self {
            view,
            period,
            delta,
            fraction,
            c_in: T::from(0.5).expect("can convert") * (T::one() - alpha),
            c_bp_1: angle.cos() * (T::one() + alpha),
            c_bp_2: -alpha,
            started: false,
            x_1: T::zero(),
            x_2: T::zero(),
            bp: T::zero(),
            bp_1: T::zero(),
            bp_2: T::zero(),
            peak: T::zero(),
            valley: T::zero(),
            mean: Sma::new(Echo::new(), two_p),
            avg_peak: Sma::new(Echo::new(), avg_len),
            avg_valley: Sma::new(Echo::new(), avg_len),
        }
    }

    /// The half-bandwidth of the band-pass.
    // Not generated by `getset`, as its impl block lacks the `T: Copy` bound.
    #[inline]
    pub fn delta(&self) -> T {
        self.delta
    }

    /// The fraction of the average peaks and valleys used as thresholds.
    #[inline]
    pub fn fraction(&self) -> T {
        self.fraction
    }

    /// The cycle component: the band-pass filtered input.
    pub fn band_pass(&self) -> Option<T> {
        self.last()?;
        Some(self.bp)
    }

    /// `Fraction` times the average of the last 50 band-pass peak values.
    pub fn upper_threshold(&self) -> Option<T> {
        self.last()?;
        Some(self.fraction * self.avg_peak.last()?)
    }

    /// `Fraction` times the average of the last 50 band-pass valley values.
    pub fn lower_threshold(&self) -> Option<T> {
        self.last()?;
        Some(self.fraction * self.avg_valley.last()?)
    }

    /// The market mode: `1` in an uptrend (`Mean` above the upper threshold),
    /// `-1` in a downtrend (`Mean` below the lower threshold), `0` in a cycle mode.
    pub fn trend_mode(&self) -> Option<T> {
        let mean = self.last()?;
        let mode = if mean > self.upper_threshold()? {
            T::one()
        } else if mean < self.lower_threshold()? {
            -T::one()
        } else {
            T::zero()
        };
        Some(mode)
    }
}

impl<T, V> View<T> for EmpiricalModeDecomposition<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if !self.started {
            // Start at rest, see struct docs.
            self.x_1 = val;
            self.x_2 = val;
            self.started = true;
        }
        self.bp_2 = self.bp_1;
        self.bp_1 = self.bp;
        self.bp = self.c_in * (val - self.x_2) + self.c_bp_1 * self.bp_1 + self.c_bp_2 * self.bp_2;
        debug_assert!(self.bp.is_finite(), "filter output must be finite");
        self.x_2 = self.x_1;
        self.x_1 = val;

        if self.bp_1 > self.bp && self.bp_1 > self.bp_2 {
            self.peak = self.bp_1;
        }
        if self.bp_1 < self.bp && self.bp_1 < self.bp_2 {
            self.valley = self.bp_1;
        }
        self.mean.update(self.bp);
        self.avg_peak.update(self.peak);
        self.avg_valley.update(self.valley);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        // All outputs become ready together.
        self.avg_peak.last()?;
        let mean = self.mean.last()?;
        debug_assert!(mean.is_finite(), "value must be finite");
        Some(mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        sliding_windows::BandPass,
        test_data::TEST_DATA,
    };

    fn emd(period: usize, delta: f64, fraction: f64) -> EmpiricalModeDecomposition<f64, Echo<f64>> {
        EmpiricalModeDecomposition::new(
            Echo::new(),
            NonZeroUsize::new(period).unwrap(),
            delta,
            fraction,
        )
    }

    #[test]
    fn empirical_mode_decomposition_plot() {
        let mut f = emd(20, 0.5, 0.1);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/empirical_mode_decomposition.png").unwrap();
    }

    #[test]
    fn empirical_mode_decomposition_warmup() {
        for (period, ready) in [(20, 50), (30, 60)] {
            let mut f = emd(period, 0.5, 0.1);
            assert_eq!(f.period().get(), period);
            assert_eq!(f.delta(), 0.5);
            assert_eq!(f.fraction(), 0.1);
            for i in 0..ready - 1 {
                f.update(i as f64);
                assert_eq!(f.last(), None);
                assert_eq!(f.band_pass(), None);
                assert_eq!(f.upper_threshold(), None);
                assert_eq!(f.lower_threshold(), None);
                assert_eq!(f.trend_mode(), None);
            }
            f.update(0.0);
            assert!(f.last().is_some());
            assert!(f.band_pass().is_some());
            assert!(f.upper_threshold().is_some());
            assert!(f.lower_threshold().is_some());
            assert!(f.trend_mode().is_some());
        }
    }

    #[test]
    fn empirical_mode_decomposition_matches_paper_code() {
        // Transcription of Figure 9, with the band-pass starting at rest.
        let (period, delta, fraction) = (20_usize, 0.5, 0.25);
        let p = period as f64;
        let beta = (360.0 / p).to_radians().cos();
        let gamma = 1.0 / (720.0 * delta / p).to_radians().cos();
        let alpha = gamma - (gamma * gamma - 1.0).sqrt();
        let x = |i: usize| TEST_DATA[i];
        let mut bp: Vec<f64> = Vec::new();
        let (mut peaks, mut valleys) = (Vec::new(), Vec::new());
        let (mut peak, mut valley) = (0.0, 0.0);
        let avg = |v: &[f64], n: usize| v[v.len() - n..].iter().sum::<f64>() / n as f64;
        let mut f = emd(period, delta, fraction);
        for i in 0..TEST_DATA.len() {
            let b1 = if i > 0 { bp[i - 1] } else { 0.0 };
            let b2 = if i > 1 { bp[i - 2] } else { 0.0 };
            bp.push(
                0.5 * (1.0 - alpha) * (x(i) - x(i.saturating_sub(2))) + beta * (1.0 + alpha) * b1
                    - alpha * b2,
            );
            if b1 > bp[i] && b1 > b2 {
                peak = b1;
            }
            if b1 < bp[i] && b1 < b2 {
                valley = b1;
            }
            peaks.push(peak);
            valleys.push(valley);
            f.update(x(i));
            let Some(mean) = f.last() else { continue };
            assert!((mean - avg(&bp, 2 * period)).abs() < 1e-9, "bar {i}");
            assert!((f.band_pass().unwrap() - bp[i]).abs() < 1e-9);
            let upper = fraction * avg(&peaks, PEAK_AVG_LEN);
            let lower = fraction * avg(&valleys, PEAK_AVG_LEN);
            assert!(
                (f.upper_threshold().unwrap() - upper).abs() < 1e-9,
                "bar {i}"
            );
            assert!(
                (f.lower_threshold().unwrap() - lower).abs() < 1e-9,
                "bar {i}"
            );
        }
    }

    #[test]
    fn empirical_mode_decomposition_band_pass_matches_band_pass() {
        let mut f = emd(20, 0.3, 0.1);
        let mut b = BandPass::new(Echo::new(), NonZeroUsize::new(20).unwrap(), 0.6);
        for v in TEST_DATA {
            f.update(v);
            b.update(v);
            if let Some(bp) = f.band_pass() {
                assert!((bp - b.last().unwrap()).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn empirical_mode_decomposition_constant_input() {
        let mut f = emd(20, 0.5, 0.1);
        for _ in 0..200 {
            f.update(42.0);
        }
        assert_eq!(f.last(), Some(0.0));
        assert_eq!(f.band_pass(), Some(0.0));
        assert_eq!(f.upper_threshold(), Some(0.0));
        assert_eq!(f.lower_threshold(), Some(0.0));
        assert_eq!(f.trend_mode(), Some(0.0));
    }

    #[test]
    fn empirical_mode_decomposition_trend_modes() {
        // A 20 bar cycle on top of a rising, flat or falling trend.
        for (slope, want) in [(0.3, 1.0), (0.0, 0.0), (-0.3, -1.0)] {
            let mut f = emd(20, 0.5, 0.1);
            for i in 0..1_000 {
                let t = i as f64;
                f.update(100.0 + slope * t + (2.0 * PI * t / 20.0).sin());
            }
            assert_eq!(f.trend_mode(), Some(want), "slope {slope}");
            let (u, l) = (f.upper_threshold().unwrap(), f.lower_threshold().unwrap());
            assert!(l < u, "slope {slope}: {l} {u}");
        }
    }

    #[test]
    #[should_panic(expected = "period must be >= 2")]
    fn empirical_mode_decomposition_period_too_small() {
        let _ = emd(1, 0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "delta must be > 0")]
    fn empirical_mode_decomposition_delta_zero() {
        let _ = emd(20, 0.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "delta must be < period / 8")]
    fn empirical_mode_decomposition_delta_too_wide() {
        let _ = emd(20, 2.5, 0.1);
    }

    #[test]
    #[should_panic(expected = "fraction must be > 0")]
    fn empirical_mode_decomposition_fraction_zero() {
        let _ = emd(20, 0.5, 0.0);
    }

    #[test]
    #[should_panic(expected = "fraction must be finite")]
    fn empirical_mode_decomposition_fraction_nan() {
        let _ = emd(20, 0.5, f64::NAN);
    }
}

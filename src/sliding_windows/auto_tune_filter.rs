// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Relative bandwidth of the tuned band-pass filter, 0.25 in the paper.
const BANDWIDTH: f64 = 0.25;

/// Maximum change of the dominant cycle per bar, 2 in the paper.
const MAX_DC_STEP: usize = 2;

/// John Ehlers AutoTune Filter: a band-pass filter tuned to the dominant cycle,
/// which is measured as twice the lag of the minimum autocorrelation.
///
/// Reference: John Ehlers, "The AutoTune Filter" (2025),
/// <https://www.mesasoftware.com/papers/The%20AutoTune%20Filter.pdf>, Code Listings 1, 3 and 4:
///
/// ```text
/// Filt = $Highpass(Close, Window)       (0 for the first 3 bars)
/// Corr[Lag] = Pearson correlation of Filt[J] and Filt[Lag + J], J = 0..Window-1, Lag = 1..Window
///             (kept from the previous bar if a variance is zero)
/// DC = 2 * (lag of the minimum Corr below 1), limited to DC[1] +- 2
/// BP = $Bandpass(Close, DC, 0.25)
/// ```
///
/// with the 2025 two-pole high-pass `Q = exp(-1.414 pi / W)`, `c1 = 2 Q cos(1.414 pi / W)`,
/// `c2 = Q^2`, `a0 = (1 + c1 + c2) / 4`, `HP = a0 (x - 2x[1] + x[2]) + c1 HP[1] - c2 HP[2]`
/// and the band-pass `L1 = cos(2 pi / P)`, `G1 = cos(0.25 * 2 pi / P)`,
/// `S1 = 1 / G1 - sqrt(1 / G1^2 - 1)`,
/// `BP = 0.5 (1 - S1) (x - x[2]) + L1 (1 + S1) BP[1] - S1 BP[2]`, whose coefficients
/// are recomputed every bar from the current `DC` while its state carries over.
/// A cycle of period `P` is most anti-correlated with itself half a period ago,
/// so the measured period is in `[2, 2 * window]` bars.
///
/// Deviations: the correlation, the dominant cycle and the band-pass start once
/// `2 * window` filtered values cover all lags (TradeStation's `MaxBarsBack`),
/// the band-pass then starts at rest on the current value. As in the paper, `DC`
/// starts at 0 and ramps up by at most 2 bars per bar, so `last()` is `None`
/// for the first `3 * window - 2` values. The dominant cycle is kept at least 2,
/// where the paper's would stay 0 if no correlation were below 1.
/// Constant input yields 0 with a dominant cycle of 2.
///
/// Output range: unbounded, oscillates around 0 with roughly the amplitude of the input cycle.
/// Complexity: O(window^2) per update for the correlations, O(window) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct AutoTuneFilter<T, V> {
    view: V,
    /// The correlation window and maximum lag.
    #[getset(get_copy = "pub")]
    window: NonZeroUsize,
    /// High-pass coefficients.
    hp_a0: T,
    hp_c1: T,
    hp_c2: T,
    /// Input one and two bars ago, for the high-pass filter.
    hp_x: [T; 2],
    /// High-pass output one and two bars ago.
    hp_y: [T; 2],
    /// High-pass values, written twice (at `pos` and `pos + 2 * window`),
    /// so that `history[pos..pos + 2 * window]` is a contiguous, newest first view.
    history: Vec<T>,
    pos: usize,
    /// Number of high-pass values, saturating at `2 * window`.
    n_filt: usize,
    /// Correlation per lag `0..=window`, index 0 is unused.
    corr: Vec<T>,
    /// Dominant cycle period in bars, 0 before the first measurement.
    dc: usize,
    /// Input one and two bars ago, for the band-pass filter.
    bp_x: [T; 2],
    /// Band-pass output now and one bar ago.
    bp_y: [T; 2],
    /// Number of band-pass updates.
    n_bp: usize,
}

impl<T, V> AutoTuneFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with the correlation `window` (20 in the paper).
    ///
    /// # Panics
    /// If `window < 2`, as a correlation needs at least two points.
    pub fn new(view: V, window: NonZeroUsize) -> Self {
        assert!(window.get() >= 2, "window must be >= 2");
        let w = T::from(window.get()).expect("can convert");
        let q = (-T::from(1.414 * PI).expect("can convert") / w).exp();
        let c1 = T::from(2.0).expect("can convert")
            * q
            * (T::from(1.414 * PI).expect("can convert") / w).cos();
        let c2 = q * q;
        let len = 2 * window.get();
        Self {
            view,
            window,
            hp_a0: (T::one() + c1 + c2) / T::from(4.0).expect("can convert"),
            hp_c1: c1,
            hp_c2: c2,
            hp_x: [T::zero(); 2],
            hp_y: [T::zero(); 2],
            history: vec![T::zero(); 2 * len],
            pos: 0,
            n_filt: 0,
            corr: vec![T::zero(); window.get() + 1],
            dc: 0,
            bp_x: [T::zero(); 2],
            bp_y: [T::zero(); 2],
            n_bp: 0,
        }
    }

    /// The dominant cycle period in bars that tunes the band-pass, in `[2, 2 * window]`.
    /// `None` while warming up.
    pub fn dominant_cycle(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.dc >= 2, "dominant cycle must be >= 2");
        debug_assert!(
            self.dc <= 2 * self.window.get(),
            "dominant cycle must be <= 2 * window"
        );
        Some(T::from(self.dc).expect("can convert"))
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.n_bp >= self.window.get()
    }

    /// The 2025 high-pass filter, 0 for the first 3 bars as in the paper.
    fn high_pass(&self, val: T, n: usize) -> T {
        if n < 3 {
            return T::zero();
        }
        let two = T::from(2.0).expect("can convert");
        self.hp_a0 * (val - two * self.hp_x[0] + self.hp_x[1]) + self.hp_c1 * self.hp_y[0]
            - self.hp_c2 * self.hp_y[1]
    }
}

impl<T, V> View<T> for AutoTuneFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let len = 2 * self.window.get();
        let hp = self.high_pass(val, self.n_filt);
        debug_assert!(hp.is_finite(), "filter output must be finite");
        self.hp_x = [val, self.hp_x[0]];
        self.hp_y = [hp, self.hp_y[0]];
        // Move backwards, so the newest value is always first in the slice.
        self.pos = if self.pos == 0 { len - 1 } else { self.pos - 1 };
        self.history[self.pos] = hp;
        self.history[self.pos + len] = hp;
        self.n_filt = (self.n_filt + 1).min(len);
        if self.n_filt < len {
            return;
        }

        let hist = &self.history[self.pos..self.pos + len];
        let dc_prev = self.dc;
        if let Some(lag) = min_correlation_lag(hist, &mut self.corr) {
            self.dc = 2 * lag;
        }
        self.dc = self
            .dc
            .clamp(dc_prev.saturating_sub(MAX_DC_STEP), dc_prev + MAX_DC_STEP);
        // If no correlation is below 1 on the first bar, the paper's DC would stay 0
        // and its band-pass would divide by zero; 2 is the shortest measurable cycle.
        self.dc = self.dc.max(2);
        debug_assert!(self.dc <= len, "dominant cycle must be <= 2 * window");

        if self.n_bp == 0 {
            // Start at rest, the high-pass warm-up took the place of the paper's first bars.
            self.bp_x = [val; 2];
        }
        let (c_in, c_bp_1, c_bp_2) = band_pass_coefficients::<T>(self.dc);
        let bp = c_in * (val - self.bp_x[1]) + c_bp_1 * self.bp_y[0] + c_bp_2 * self.bp_y[1];
        debug_assert!(bp.is_finite(), "filter output must be finite");
        self.bp_x = [val, self.bp_x[0]];
        self.bp_y = [bp, self.bp_y[0]];
        self.n_bp = self.n_bp.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.bp_y[0].is_finite(), "value must be finite");
        Some(self.bp_y[0])
    }
}

/// Updates the correlation of `hist` (newest first, `2 * window` values) with its lagged
/// copy for each lag `1..=window` and returns the lag of the minimum correlation below 1.
/// Correlations with a zero variance keep their previous value, as in the paper.
fn min_correlation_lag<T: Float>(hist: &[T], corr: &mut [T]) -> Option<usize> {
    let window = corr.len() - 1;
    debug_assert_eq!(hist.len(), 2 * window, "history must cover all lags");
    let w = T::from(window).expect("can convert");
    let x = &hist[..window];
    let (mut min_corr, mut min_lag) = (T::one(), None);
    for (lag, c) in corr.iter_mut().enumerate().skip(1) {
        let y = &hist[lag..lag + window];
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) =
            (T::zero(), T::zero(), T::zero(), T::zero(), T::zero());
        for (&x, &y) in x.iter().zip(y) {
            sx = sx + x;
            sy = sy + y;
            sxx = sxx + x * x;
            syy = syy + y * y;
            sxy = sxy + x * y;
        }
        let (vx, vy) = (w * sxx - sx * sx, w * syy - sy * sy);
        if vx > T::zero() && vy > T::zero() {
            *c = (w * sxy - sx * sy) / (vx * vy).sqrt();
        }
        if *c < min_corr {
            min_corr = *c;
            min_lag = Some(lag);
        }
    }
    min_lag
}

/// Coefficients of `x - x[2]`, `BP[1]` and `BP[2]` of the paper's band-pass with `period`.
fn band_pass_coefficients<T: Float>(period: usize) -> (T, T, T) {
    debug_assert!(period >= 2, "period must be >= 2");
    let angle = T::from(2.0 * PI).expect("can convert") / T::from(period).expect("can convert");
    let l1 = angle.cos();
    let theta = T::from(BANDWIDTH).expect("can convert") * angle;
    // Algebraically equal to `1 / G1 - sqrt(1 / G1^2 - 1)` with `G1 = cos(theta)`,
    // see `BandPass`; theta <= pi / 4 as the period is at least 2.
    let s1 = theta.cos() / (T::one() + theta.sin());
    debug_assert!(s1 > T::zero(), "S1 must be positive");
    debug_assert!(s1 < T::one(), "S1 must be below 1");
    let half = T::from(0.5).expect("can convert");
    (half * (T::one() - s1), l1 * (T::one() + s1), -s1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn atf(window: usize) -> AutoTuneFilter<f64, Echo<f64>> {
        AutoTuneFilter::new(Echo::new(), NonZeroUsize::new(window).unwrap())
    }

    fn sine(i: usize, period: f64) -> f64 {
        100.0 + 10.0 * (2.0 * PI * i as f64 / period).sin()
    }

    #[test]
    fn auto_tune_filter_plot() {
        let mut f = atf(20);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/auto_tune_filter.png").unwrap();
    }

    #[test]
    fn auto_tune_filter_warmup() {
        let mut f = atf(20);
        assert_eq!(f.window().get(), 20);
        for i in 0..58 {
            f.update(sine(i, 20.0));
            assert_eq!(f.last(), None);
            assert_eq!(f.dominant_cycle(), None);
        }
        f.update(sine(58, 20.0));
        assert!(f.last().is_some());
        assert!(f.dominant_cycle().is_some());
    }

    #[test]
    fn auto_tune_filter_measures_sine_period() {
        for period in [15.0, 20.0, 30.0] {
            let mut f = atf(20);
            let mut max_abs: f64 = 0.0;
            for i in 0..600 {
                f.update(sine(i, period));
                if i > 400 {
                    max_abs = max_abs.max(f.last().unwrap().abs());
                }
            }
            let dc = f.dominant_cycle().unwrap();
            // The measurement is quantized to even periods.
            assert!((dc - period).abs() <= 2.0, "period {period}: dc {dc}");
            // The tuned band-pass passes the cycle with about unit gain.
            assert!(
                (7.0..13.0).contains(&max_abs),
                "period {period}: amplitude {max_abs}"
            );
        }
    }

    #[test]
    fn auto_tune_filter_dominant_cycle_rate_limited() {
        let mut f = atf(20);
        let mut prev: Option<f64> = None;
        for i in 0..1_000 {
            f.update(sine(i, if i < 500 { 12.0 } else { 36.0 }));
            let dc = f.dominant_cycle();
            if let (Some(dc), Some(prev)) = (dc, prev) {
                assert!((dc - prev).abs() <= 2.0, "dc jumped from {prev} to {dc}");
                assert!((2.0..=40.0).contains(&dc));
            }
            prev = dc;
        }
        let dc = f.dominant_cycle().unwrap();
        assert!((dc - 36.0).abs() <= 2.0, "dc {dc}");
    }

    #[test]
    fn auto_tune_filter_constant_input() {
        let mut f = atf(20);
        for _ in 0..200 {
            f.update(42.0);
        }
        assert_eq!(f.last(), Some(0.0));
        assert_eq!(f.dominant_cycle(), Some(2.0));
    }

    /// Direct transcription of the paper's code with the same start-up.
    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "indexing mirrors the paper's EasyLanguage code"
    )]
    fn auto_tune_filter_matches_paper_transcription() {
        let window = 20;
        let w = window as f64;
        let mut f = atf(window);
        let q = (-1.414 * PI / w).exp();
        let c1 = 2.0 * q * (1.414 * PI / w).cos();
        let c2 = q * q;
        let a0 = (1.0 + c1 + c2) / 4.0;
        let (mut close, mut hp, mut bp, mut bp_x) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut corr = vec![0.0; window + 1];
        let (mut dc, mut checked) = (0.0, 0);
        let rng = romu::Rng::from_seed_with_64bit(31);
        let mut price = 100.0;
        for bar in 1..=1_000 {
            price += rng.f64() - 0.5;
            f.update(price);
            close.insert(0, price);
            let h = if bar >= 4 {
                a0 * (close[0] - 2.0 * close[1] + close[2]) + c1 * hp[0] - c2 * hp[1]
            } else {
                0.0
            };
            hp.insert(0, h);
            if hp.len() < 2 * window {
                continue;
            }
            for lag in 1..=window {
                let (mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for j in 0..window {
                    let (x, y) = (hp[j], hp[lag + j]);
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    sxy += x * y;
                    syy += y * y;
                }
                if w * sxx - sx * sx > 0.0 && w * syy - sy * sy > 0.0 {
                    corr[lag] =
                        (w * sxy - sx * sy) / ((w * sxx - sx * sx) * (w * syy - sy * sy)).sqrt();
                }
            }
            let dc_1 = dc;
            let mut min_corr = 1.0;
            for lag in 1..=window {
                if corr[lag] < min_corr {
                    min_corr = corr[lag];
                    dc = 2.0 * lag as f64;
                }
            }
            dc = f64::clamp(dc, dc_1 - 2.0, dc_1 + 2.0);
            // The band-pass starts at rest on the first bar after the warm-up.
            if bp.is_empty() {
                bp_x = vec![price, price];
            }
            let x2 = bp_x[1];
            bp_x.insert(0, price);
            let (b1, b2) = (
                bp.first().copied().unwrap_or(0.0),
                bp.get(1).copied().unwrap_or(0.0),
            );
            let l1 = (2.0 * PI / dc).cos();
            let g1 = (0.25 * 2.0 * PI / dc).cos();
            let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();
            bp.insert(
                0,
                0.5 * (1.0 - s1) * (price - x2) + l1 * (1.0 + s1) * b1 - s1 * b2,
            );
            if let Some(v) = f.last() {
                assert!((v - bp[0]).abs() < 1e-9, "bar {bar}: {v} != {}", bp[0]);
                assert_eq!(f.dominant_cycle(), Some(dc));
                checked += 1;
            }
        }
        assert!(checked > 900);
    }

    #[test]
    #[should_panic(expected = "window must be >= 2")]
    fn auto_tune_filter_window_too_small() {
        let _ = atf(1);
    }
}

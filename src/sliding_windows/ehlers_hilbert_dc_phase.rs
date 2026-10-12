//! Dominant cycle phase of Ehlers' Sine Wave indicator, shared by `SineWave` and
//! `HilbertTrendMode`. Reference: "Rocket Science for Traders" (2001), chapter 9,
//! and TA-Lib's `HT_SINE`.

use num::Float;

use super::ehlers_hilbert_core::c;

/// Capacity of the smoothed price history, the largest possible dominant cycle in bars.
/// Equals `MAX_PERIOD`, checked in the tests.
pub(super) const SIZE: usize = 50;

/// Integer dominant cycle length: `SmoothPeriod` rounded to the nearest bar.
#[inline(always)]
pub(super) fn dominant_cycle_bars<T: Float>(smooth_period: T) -> usize {
    debug_assert!(
        smooth_period >= T::zero(),
        "smooth period must not be negative"
    );
    let bars = (smooth_period + c(0.5))
        .floor()
        .to_usize()
        .expect("smooth period is finite and non-negative");
    // The period is a convex combination of values clamped to MAX_PERIOD, so it cannot exceed it.
    assert!(bars <= SIZE, "dominant cycle exceeds the history size");
    bars
}

/// Correlate the last `bars` values (newest first, ending at `head` in the ring buffer `buf`)
/// with one cycle of a sine and a cosine. Returns `(real, imag)`.
/// The sine and cosine are advanced by complex rotation, so only one `sin_cos` call is needed.
#[inline(always)]
fn dft<T: Float>(buf: &[T; SIZE], head: usize, bars: usize) -> (T, T) {
    debug_assert!(head < SIZE, "head out of bounds");
    debug_assert!(bars <= SIZE, "bars out of bounds");
    if bars == 0 {
        return (T::zero(), T::zero());
    }
    let step = c::<T>(std::f64::consts::TAU) / T::from(bars).expect("can convert");
    let (sin_step, cos_step) = step.sin_cos();
    let (mut sin, mut cos) = (T::zero(), T::one());
    let (mut real, mut imag) = (T::zero(), T::zero());
    for i in 0..bars {
        let val = buf[(head + SIZE - i) % SIZE];
        real = real + sin * val;
        imag = imag + cos * val;
        (sin, cos) = (
            sin * cos_step + cos * sin_step,
            cos * cos_step - sin * sin_step,
        );
    }
    (real, imag)
}

/// Ehlers' phase resolution of the DFT, in degrees.
/// Adds 90 degrees to undo the sine reference, `360 / SmoothPeriod` to compensate the
/// one bar lag of the 4-bar WMA, resolves the arctangent quadrant from the sign of `imag`
/// and wraps into `(-45, 315]` for a non-degenerate input.
#[inline(always)]
fn resolve_phase<T: Float>(real: T, imag: T, prev_phase: T, smooth_period: T) -> T {
    debug_assert!(smooth_period > T::zero(), "smooth period must be positive");
    let quarter = c::<T>(90.0);
    // TA-Lib keeps the previous phase, rotated by 90 degrees toward the sign of `real`,
    // when the cosine correlation vanishes.
    let mut phase = if imag != T::zero() {
        (real / imag).atan().to_degrees()
    } else if real < T::zero() {
        prev_phase - quarter
    } else if real > T::zero() {
        prev_phase + quarter
    } else {
        prev_phase
    };
    phase = phase + quarter + c::<T>(360.0) / smooth_period;
    if imag < T::zero() {
        phase = phase + c(180.0);
    }
    if phase > c(315.0) {
        phase = phase - c(360.0);
    }
    phase
}

/// Streaming dominant cycle phase. O(dominant cycle) <= O(50) per update.
#[derive(Debug, Clone)]
pub(super) struct DcPhase<T> {
    /// Ring buffer of smoothed prices.
    smooth_prices: [T; SIZE],
    /// Index of the newest value in `smooth_prices`.
    head: usize,
    n_updates: usize,
    /// Dominant cycle phase in degrees.
    phase: T,
}

impl<T: Float> DcPhase<T> {
    pub(super) fn new() -> Self {
        Self {
            smooth_prices: [T::zero(); SIZE],
            head: 0,
            n_updates: 0,
            phase: T::zero(),
        }
    }

    /// Feed the 4-bar WMA smoothed price and the current smoothed dominant cycle period.
    pub(super) fn update(&mut self, smooth: T, smooth_period: T) {
        debug_assert!(smooth.is_finite(), "value must be finite");
        debug_assert!(smooth_period > T::zero(), "smooth period must be positive");
        if self.n_updates == 0 {
            // Pre-fill instead of zeros, so the DFT never sees an artificial jump to the price level.
            self.smooth_prices = [smooth; SIZE];
        }
        self.n_updates = self.n_updates.saturating_add(1);
        self.head = (self.head + 1) % SIZE;
        self.smooth_prices[self.head] = smooth;

        let bars = dominant_cycle_bars(smooth_period);
        let (real, imag) = dft(&self.smooth_prices, self.head, bars);
        self.phase = resolve_phase(real, imag, self.phase, smooth_period);
        debug_assert!(self.phase.is_finite(), "phase must be finite");
    }

    /// Dominant cycle phase in degrees.
    #[inline(always)]
    pub(super) fn phase(&self) -> T {
        self.phase
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::{
        super::ehlers_hilbert_core::MAX_PERIOD,
        *,
    };

    #[test]
    fn ehlers_hilbert_dc_phase_size_is_max_period() {
        assert_eq!(SIZE as f64, MAX_PERIOD);
    }

    #[test]
    fn ehlers_hilbert_dc_phase_dft_matches_direct() {
        let mut buf = [0.0; SIZE];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i as f64 * 0.37).sin() * 3.0 + i as f64;
        }
        let head = 17;
        for bars in [1, 6, 20, 33, 50] {
            let (real, imag) = dft(&buf, head, bars);
            let (mut r, mut im) = (0.0, 0.0);
            for i in 0..bars {
                let a = 2.0 * PI * i as f64 / bars as f64;
                let v = buf[(head + SIZE - i) % SIZE];
                r += a.sin() * v;
                im += a.cos() * v;
            }
            assert!((real - r).abs() < 1e-9, "{bars}");
            assert!((imag - im).abs() < 1e-9, "{bars}");
        }
        assert_eq!(dft(&buf, head, 0), (0.0, 0.0));
    }

    #[test]
    fn ehlers_hilbert_dc_phase_bars() {
        assert_eq!(dominant_cycle_bars(0.0), 0);
        assert_eq!(dominant_cycle_bars(19.49), 19);
        assert_eq!(dominant_cycle_bars(19.5), 20);
        assert_eq!(dominant_cycle_bars(50.0), 50);
    }

    #[test]
    #[should_panic]
    fn ehlers_hilbert_dc_phase_bars_too_large() {
        let _ = dominant_cycle_bars(51.0);
    }

    #[test]
    fn ehlers_hilbert_dc_phase_quadrants() {
        // imag > 0, real = 0 -> 90 degrees plus lag compensation.
        assert!((resolve_phase(0.0, 1.0, 0.0, 20.0) - 108.0).abs() < 1e-9);
        // imag < 0 adds 180 degrees.
        assert!((resolve_phase(0.0, -1.0, 0.0, 20.0) - 288.0).abs() < 1e-9);
        // Wraps above 315 degrees.
        assert!((resolve_phase(-1.0, -1e-3, 0.0, 20.0) - (18.0 - 0.0573)).abs() < 1e-3);
        // Degenerate imag keeps the previous phase, rotated by 90 degrees.
        assert!((resolve_phase(1.0, 0.0, 10.0, 20.0) - 208.0).abs() < 1e-9);
        assert!((resolve_phase(-1.0, 0.0, 10.0, 20.0) - 28.0).abs() < 1e-9);
    }
}

//! Shared building blocks of John Ehlers' Hilbert-transform indicators,
//! as published in "Rocket Science for Traders" (2001), chapter 7, and
//! implemented by TA-Lib's `HT_*` functions.
//!
//! The pipeline per bar is:
//! 4-bar WMA smoother -> Hilbert FIR detrender -> InPhase / Quadrature ->
//! 90 degree phase advance (jI, jQ) -> homodyne discriminator -> period.

use num::Float;

/// Smallest dominant cycle period in bars that the homodyne discriminator may report.
pub(super) const MIN_PERIOD: f64 = 6.0;
/// Largest dominant cycle period in bars that the homodyne discriminator may report.
pub(super) const MAX_PERIOD: f64 = 50.0;

/// Convert an `f64` constant into `T`.
#[inline(always)]
pub(super) fn c<T: Float>(x: f64) -> T {
    T::from(x).expect("can convert")
}

/// Ehlers' Hilbert transformer FIR, `x` is newest first.
/// The coefficients (0.0962, 0.5769) approximate a Hilbert transform,
/// the gain `0.075 * Period[1] + 0.54` compensates its amplitude roll-off for long periods.
#[inline(always)]
fn hilbert_fir<T: Float>(x: &[T], gain: T) -> T {
    debug_assert!(x.len() >= 7, "FIR needs 7 taps");
    let a = c::<T>(0.0962);
    let b = c::<T>(0.5769);
    // Grouped by symmetric taps, so a constant input cancels exactly.
    (a * (x[0] - x[6]) + b * (x[2] - x[4])) * gain
}

/// Push `val` to the front of a newest-first history, dropping the oldest value.
#[inline(always)]
fn push_front<T: Copy, const N: usize>(hist: &mut [T; N], val: T) {
    hist.copy_within(0..N - 1, 1);
    hist[0] = val;
}

/// Restrict the freshly measured period to [0.67, 1.5] times the previous period
/// and to [`MIN_PERIOD`, `MAX_PERIOD`], then smooth it 0.2 / 0.8 with the previous period.
#[inline(always)]
fn constrain_period<T: Float>(measured: T, prev: T) -> T {
    debug_assert!(prev >= T::zero(), "previous period must not be negative");
    let period = measured
        .min(c::<T>(1.5) * prev)
        .max(c::<T>(0.67) * prev)
        .max(c(MIN_PERIOD))
        .min(c(MAX_PERIOD));
    let out = c::<T>(0.2) * period + c::<T>(0.8) * prev;
    debug_assert!(out <= c(MAX_PERIOD), "period above maximum");
    out
}

/// The shared Hilbert transform and homodyne discriminator state.
/// Every update is O(1).
#[derive(Debug, Clone)]
pub(super) struct HilbertCore<T> {
    /// Raw prices, newest first, for the 4-bar WMA.
    prices: [T; 4],
    /// Number of observed inputs, saturating.
    n_inputs: usize,
    /// WMA-smoothed prices, newest first.
    smooth: [T; 7],
    /// Detrender history, newest first. 10 values because `I1 = Detrender[3]`
    /// and the jI FIR reaches 6 bars back from there.
    detrender: [T; 10],
    /// Quadrature history, newest first.
    q1: [T; 7],
    i2: T,
    q2: T,
    re: T,
    im: T,
    period: T,
    smooth_period: T,
}

impl<T: Float> HilbertCore<T> {
    pub(super) fn new() -> Self {
        Self {
            prices: [T::zero(); 4],
            n_inputs: 0,
            smooth: [T::zero(); 7],
            detrender: [T::zero(); 10],
            q1: [T::zero(); 7],
            i2: T::zero(),
            q2: T::zero(),
            re: T::zero(),
            im: T::zero(),
            period: T::zero(),
            smooth_period: T::zero(),
        }
    }

    /// Feed a new price. The Hilbert stage only starts once the 4-bar WMA is fully populated,
    /// so its history never contains a partially filled smoother output.
    pub(super) fn update(&mut self, price: T) {
        debug_assert!(price.is_finite(), "value must be finite");
        push_front(&mut self.prices, price);
        self.n_inputs = self.n_inputs.saturating_add(1);
        if self.n_inputs < self.prices.len() {
            return;
        }

        let p = &self.prices;
        let smooth =
            (c::<T>(4.0) * p[0] + c::<T>(3.0) * p[1] + c::<T>(2.0) * p[2] + p[3]) / c::<T>(10.0);
        if self.n_inputs == self.prices.len() {
            // Pre-fill instead of starting from zeros, as a jump from zero to the price level
            // would ring through the FIR stages and distort the period for dozens of bars.
            self.smooth = [smooth; 7];
        }
        push_front(&mut self.smooth, smooth);

        let gain = c::<T>(0.075) * self.period + c::<T>(0.54);
        let detrender = hilbert_fir(&self.smooth, gain);
        push_front(&mut self.detrender, detrender);
        let q1 = hilbert_fir(&self.detrender, gain);
        push_front(&mut self.q1, q1);
        let i1 = self.detrender[3];
        // Advance the phase of I1 and Q1 by 90 degrees.
        let ji = hilbert_fir(&self.detrender[3..], gain);
        let jq = hilbert_fir(&self.q1, gain);

        // Phasor addition for 3 bar averaging, smoothed before the discriminator.
        let alpha = c::<T>(0.2);
        let beta = c::<T>(0.8);
        let i2 = alpha * (i1 - jq) + beta * self.i2;
        let q2 = alpha * (q1 + ji) + beta * self.q2;

        // Homodyne discriminator: multiply the phasor with the conjugate of its previous value.
        self.re = alpha * (i2 * self.i2 + q2 * self.q2) + beta * self.re;
        self.im = alpha * (i2 * self.q2 - q2 * self.i2) + beta * self.im;
        self.i2 = i2;
        self.q2 = q2;

        let prev = self.period;
        let measured = if self.im != T::zero() && self.re != T::zero() {
            c::<T>(360.0) / (self.im / self.re).atan().to_degrees()
        } else {
            prev
        };
        self.period = constrain_period(measured, prev);
        self.smooth_period = c::<T>(0.33) * self.period + c::<T>(0.67) * self.smooth_period;
        debug_assert!(self.period.is_finite(), "period must be finite");
        debug_assert!(
            self.smooth_period.is_finite(),
            "smooth period must be finite"
        );
    }

    /// Number of observed inputs, saturating at `usize::MAX`.
    #[inline(always)]
    pub(super) fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Whether the 4-bar WMA is full, so the Hilbert stage has run at least once
    /// and [`Self::smooth_period`] is positive.
    #[inline(always)]
    pub(super) fn has_output(&self) -> bool {
        self.n_inputs >= self.prices.len()
    }

    /// The latest 4-bar WMA smoothed price.
    #[inline(always)]
    pub(super) fn smooth(&self) -> T {
        self.smooth[0]
    }

    /// The latest InPhase component (`Detrender[3]`).
    #[inline(always)]
    pub(super) fn i1(&self) -> T {
        self.detrender[3]
    }

    /// The latest Quadrature component.
    #[inline(always)]
    pub(super) fn q1(&self) -> T {
        self.q1[0]
    }

    /// The constrained, 0.2 / 0.8 smoothed period in bars.
    #[inline(always)]
    pub(super) fn period(&self) -> T {
        self.period
    }

    /// The period additionally smoothed 0.33 / 0.67, in bars.
    #[inline(always)]
    pub(super) fn smooth_period(&self) -> T {
        self.smooth_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ehlers_hilbert_core_constrain_period() {
        // Growth limited to 1.5x then smoothed.
        assert_eq!(constrain_period(100.0, 10.0), 0.2 * 15.0 + 0.8 * 10.0);
        // Shrink limited to 0.67x, but never below MIN_PERIOD.
        assert_eq!(constrain_period(1.0, 10.0), 0.2 * 6.7 + 0.8 * 10.0);
        assert_eq!(constrain_period(1.0, 0.0), 0.2 * MIN_PERIOD);
        // Never above MAX_PERIOD.
        assert_eq!(constrain_period(1000.0, 50.0), 50.0);
        // Negative measurements (atan < 0) fall back to the lower bounds.
        assert_eq!(constrain_period(-30.0, 20.0), 0.2 * 13.4 + 0.8 * 20.0);
    }

    #[test]
    fn ehlers_hilbert_core_hilbert_fir_kills_dc() {
        // The FIR coefficients are antisymmetric, so a constant input yields zero.
        let x = [5.0_f64; 7];
        assert_eq!(hilbert_fir(&x, 1.0), 0.0);
    }

    #[test]
    fn ehlers_hilbert_core_wma() {
        let mut core = HilbertCore::<f64>::new();
        for v in [1.0, 2.0, 3.0] {
            core.update(v);
            assert_eq!(core.smooth[0], 0.0);
        }
        core.update(4.0);
        assert_eq!(
            core.smooth[0],
            (4.0 * 4.0 + 3.0 * 3.0 + 2.0 * 2.0 + 1.0) / 10.0
        );
        assert_eq!(core.n_inputs(), 4);
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    collections::VecDeque,
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// John Ehlers Deviation Scaled Moving Average (DSMA).
///
/// An EMA whose `alpha` follows the amplitude of a smoothed oscillator in
/// units of its own RMS: large moves (relative to recent volatility) make the
/// average follow quickly, quiet markets get heavy smoothing.
///
/// Reference: John Ehlers, "The Deviation-Scaled Moving Average",
/// Technical Analysis of Stocks & Commodities, July 2018,
/// <https://www.mesasoftware.com/papers/DEVIATION%20SCALED%20MOVING%20AVERAGE.pdf>:
///
/// ```text
/// Zeros      = x - x[2]
/// Filt       = SuperSmoother(Zeros, 0.5 P)   // c1 (Zeros + Zeros[1]) / 2 + c2 Filt[1] + c3 Filt[2]
/// RMS        = sqrt(sum_{k=0}^{P-1} Filt[k]^2 / P)
/// ScaledFilt = Filt / RMS                     // kept when RMS = 0
/// alpha      = |ScaledFilt| 5 / P
/// DSMA       = alpha x + (1 - alpha) DSMA[1]
/// ```
///
/// Deviations:
/// - `alpha` is clamped to at most 1. Since `Filt^2 <= P RMS^2`,
///   `alpha <= 5 / sqrt(P)`, so the clamp is never active for `P >= 25`
///   (Ehlers uses 40); for shorter periods it prevents the EMA from
///   overshooting the input or becoming unstable.
/// - The input history is seeded with the first value (so `Zeros` starts at 0)
///   and `DSMA` starts at the first input instead of 0.
/// - Exact pi instead of `3.14159`.
///
/// `last()` is `None` for the first `period - 1` values.
///
/// Output range: within the range of the inputs seen so far (`alpha` is in `[0, 1]`).
/// Constant input yields the constant.
/// Complexity: amortized O(1) per update (running sum of `Filt^2`, recomputed
/// every `period` updates to stop floating point drift), O(period) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Dsma<T, V> {
    view: V,
    /// The critical period; the SuperSmoother uses half of it.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    /// SuperSmoother coefficients.
    c1: T,
    c2: T,
    c3: T,
    /// Whether the input history has been seeded.
    started: bool,
    x_1: T,
    x_2: T,
    zeros_1: T,
    filt_1: T,
    filt_2: T,
    /// `Filt^2` of the last `period` values.
    sq_filts: VecDeque<T>,
    sum_sq: T,
    /// Updates since `sum_sq` was last recomputed from scratch.
    since_recompute: usize,
    scaled_filt: T,
    dsma: T,
}

impl<T, V> Dsma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new DSMA with a chained `View`. Ehlers uses `period = 40`.
    ///
    /// # Panics
    /// If `period < 4`, as the SuperSmoother period `period / 2` must be at least 2.
    pub fn new(view: V, period: NonZeroUsize) -> Self {
        assert!(period.get() >= 4, "period must be >= 4");
        let ss_period =
            T::from(0.5).expect("can convert") * T::from(period.get()).expect("can convert");
        let sqrt2_pi = T::from(1.414 * PI).expect("can convert");
        let a1 = (-sqrt2_pi / ss_period).exp();
        let b1 = T::from(2.0).expect("can convert") * a1 * (sqrt2_pi / ss_period).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        Self {
            view,
            period,
            c1: T::one() - c2 - c3,
            c2,
            c3,
            started: false,
            x_1: T::zero(),
            x_2: T::zero(),
            zeros_1: T::zero(),
            filt_1: T::zero(),
            filt_2: T::zero(),
            sq_filts: VecDeque::with_capacity(period.get()),
            sum_sq: T::zero(),
            since_recompute: 0,
            scaled_filt: T::zero(),
            dsma: T::zero(),
        }
    }

    /// Push `Filt^2` into the RMS window and return the new RMS.
    fn push_rms(&mut self, filt: T) -> T {
        let sq = filt * filt;
        if self.sq_filts.len() == self.period.get() {
            let old = self.sq_filts.pop_front().expect("the window is full");
            self.sum_sq = self.sum_sq - old;
        }
        self.sq_filts.push_back(sq);
        self.sum_sq = self.sum_sq + sq;
        self.since_recompute += 1;
        if self.since_recompute >= self.period.get() {
            // Exact sum, so rounding errors of the running sum cannot accumulate.
            self.sum_sq = self.sq_filts.iter().fold(T::zero(), |acc, v| acc + *v);
            self.since_recompute = 0;
        }
        debug_assert!(self.sq_filts.len() <= self.period.get());
        // Always divided by `period`, as EasyLanguage reads 0 before the first bar.
        (self.sum_sq.max(T::zero()) / T::from(self.period.get()).expect("can convert")).sqrt()
    }
}

impl<T, V> View<T> for Dsma<T, V>
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
            self.dsma = val;
            self.started = true;
        }
        let zeros = val - self.x_2;
        let half = T::from(0.5).expect("can convert");
        let filt =
            self.c1 * (zeros + self.zeros_1) * half + self.c2 * self.filt_1 + self.c3 * self.filt_2;
        let rms = self.push_rms(filt);
        if rms > T::zero() {
            self.scaled_filt = filt / rms;
        }
        let five = T::from(5.0).expect("can convert");
        let p = T::from(self.period.get()).expect("can convert");
        let alpha = (self.scaled_filt.abs() * five / p).min(T::one());
        debug_assert!(alpha >= T::zero(), "alpha must not be negative");
        debug_assert!(alpha <= T::one(), "alpha must not exceed 1");
        self.dsma = alpha * val + (T::one() - alpha) * self.dsma;
        debug_assert!(self.dsma.is_finite(), "filter output must be finite");

        self.x_2 = self.x_1;
        self.x_1 = val;
        self.zeros_1 = zeros;
        self.filt_2 = self.filt_1;
        self.filt_1 = filt;
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.sq_filts.len() < self.period.get() {
            return None;
        }
        debug_assert!(self.dsma.is_finite(), "value must be finite");
        Some(self.dsma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    fn dsma(period: usize) -> Dsma<f64, Echo<f64>> {
        Dsma::new(Echo::new(), NonZeroUsize::new(period).unwrap())
    }

    /// Transcription of the published code (with this crate's seeding), without the clamp.
    fn reference(prices: &[f64], period: usize) -> Vec<f64> {
        let p = period as f64;
        // The published truncated pi, written as a ratio to keep clippy quiet.
        let ehlers_pi = 314_159.0 / 100_000.0;
        let a1 = (-1.414 * ehlers_pi / (0.5 * p)).exp();
        let b1 = 2.0 * a1 * (1.414 * 180.0 / (0.5 * p)).to_radians().cos();
        let (c2, c3) = (b1, -a1 * a1);
        let c1 = 1.0 - c2 - c3;
        let x = |i: usize| prices[i];
        let mut zeros: Vec<f64> = Vec::new();
        let mut filt: Vec<f64> = Vec::new();
        let mut out: Vec<f64> = Vec::new();
        let mut scaled = 0.0;
        for i in 0..prices.len() {
            zeros.push(x(i) - x(i.saturating_sub(2)));
            let z1 = if i > 0 { zeros[i - 1] } else { 0.0 };
            let f1 = if i > 0 { filt[i - 1] } else { 0.0 };
            let f2 = if i > 1 { filt[i - 2] } else { 0.0 };
            filt.push(c1 * (zeros[i] + z1) / 2.0 + c2 * f1 + c3 * f2);
            let mut rms = 0.0;
            for count in 0..period.min(i + 1) {
                rms += filt[i - count] * filt[i - count];
            }
            rms = (rms / p).sqrt();
            if rms != 0.0 {
                scaled = filt[i] / rms;
            }
            let alpha = scaled.abs() * 5.0 / p;
            let prev = if i > 0 { out[i - 1] } else { x(0) };
            out.push(alpha * x(i) + (1.0 - alpha) * prev);
        }
        out
    }

    #[test]
    fn dsma_plot() {
        let mut f = dsma(40);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/dsma.png").unwrap();
    }

    #[test]
    fn dsma_warmup() {
        let mut f = dsma(10);
        assert_eq!(f.period().get(), 10);
        for i in 0..9 {
            f.update(i as f64);
            assert_eq!(f.last(), None);
        }
        f.update(9.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn dsma_matches_published_code() {
        for period in [25, 40] {
            let want = reference(&TEST_DATA, period);
            let mut f = dsma(period);
            for (i, v) in TEST_DATA.iter().enumerate() {
                f.update(*v);
                if let Some(got) = f.last() {
                    // 3.14159 vs exact pi.
                    assert!(
                        (got - want[i]).abs() < 1e-3,
                        "P {period} bar {i}: {got} vs {}",
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn dsma_constant_input() {
        let mut f = dsma(20);
        for _ in 0..100 {
            f.update(42.0);
        }
        assert_eq!(f.last(), Some(42.0));
    }

    #[test]
    fn dsma_short_period_stays_within_input_range() {
        let rng = romu::Rng::from_seed_with_64bit(17);
        let mut f = dsma(4);
        for i in 0..10_000 {
            // Spikes push |ScaledFilt| to its maximum, where the clamp matters.
            let v = if i % 50 == 0 {
                200.0
            } else {
                100.0 + rng.f64()
            };
            f.update(v);
            if let Some(out) = f.last() {
                assert!(out >= 100.0, "{out}");
                assert!(out <= 200.0, "{out}");
            }
        }
    }

    #[test]
    fn dsma_adapts_to_large_moves() {
        // After a quiet phase a jump is many deviations large, so alpha grows and the
        // DSMA follows faster than an EMA with the base alpha 2 / (P + 1).
        let rng = romu::Rng::from_seed_with_64bit(19);
        let mut f = dsma(40);
        let mut ema = Ema::new(Echo::new(), NonZeroUsize::new(40).unwrap());
        for i in 0..210 {
            let v = if i < 200 { 100.0 } else { 110.0 } + 0.01 * rng.f64();
            f.update(v);
            ema.update(v);
        }
        let (d, e) = (f.last().unwrap(), ema.last().unwrap());
        assert!(d > e + 1.0, "dsma {d}, ema {e}");
        assert!(d < 110.1);
    }

    #[test]
    #[should_panic(expected = "period must be >= 4")]
    fn dsma_period_too_small() {
        let _ = dsma(3);
    }
}

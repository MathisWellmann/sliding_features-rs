use std::{
    f64::consts::PI,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Recursion state of the UltimateSmoother.
///
/// Kept separate from the [`UltimateSmoother`] view, because other indicators
/// need a fractional period (e.g. `Length / 2`) or the value during warm-up.
#[derive(Debug, Clone)]
pub(super) struct UltimateCore<T> {
    /// Coefficient of `x`, `1 - c1`.
    c_x0: T,
    /// Coefficient of `x[1]`, `2 c1 - c2`.
    c_x1: T,
    /// Coefficient of `x[2]`, `-(c1 + c3)`.
    c_x2: T,
    /// Coefficient of `US[1]`, `c2`.
    c_us1: T,
    /// Coefficient of `US[2]`, `c3`.
    c_us2: T,
    /// Number of values seen so far.
    n: usize,
    x_1: T,
    x_2: T,
    us_1: T,
    us_2: T,
}

impl<T: Float> UltimateCore<T> {
    /// # Panics
    /// If `period` is below 2 or not finite.
    pub(super) fn new(period: T) -> Self {
        assert!(period.is_finite(), "period must be finite");
        assert!(
            period >= T::from(2.0).expect("can convert"),
            "period must be >= 2"
        );
        let sqrt2_pi = T::from(1.414 * PI).expect("can convert");
        let a1 = (-sqrt2_pi / period).exp();
        // Ehlers writes `Cosine(1.414 * 180 / Period)` in degrees.
        let b1 = T::from(2.0).expect("can convert") * a1 * (sqrt2_pi / period).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = (T::one() + c2 - c3) / T::from(4.0).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        Self {
            c_x0: T::one() - c1,
            c_x1: two * c1 - c2,
            c_x2: -(c1 + c3),
            c_us1: c2,
            c_us2: c3,
            n: 0,
            x_1: T::zero(),
            x_2: T::zero(),
            us_1: T::zero(),
            us_2: T::zero(),
        }
    }

    /// Feed the next input and return the new filter value.
    pub(super) fn update(&mut self, x: T) -> T {
        debug_assert!(x.is_finite(), "value must be finite");
        // `If CurrentBar < 4 Then US = Price`: the recursion needs two bars of history.
        let us = if self.n < 3 {
            x
        } else {
            self.c_x0 * x
                + self.c_x1 * self.x_1
                + self.c_x2 * self.x_2
                + self.c_us1 * self.us_1
                + self.c_us2 * self.us_2
        };
        debug_assert!(us.is_finite(), "filter output must be finite");
        self.x_2 = self.x_1;
        self.x_1 = x;
        self.us_2 = self.us_1;
        self.us_1 = us;
        self.n = self.n.saturating_add(1);
        us
    }

    /// The most recent filter value, `0` before the first update.
    #[inline]
    pub(super) fn value(&self) -> T {
        self.us_1
    }
}

/// John Ehlers UltimateSmoother.
///
/// A low-pass filter built by subtracting a two-pole high-pass response from
/// the input (an all-pass). It has unit gain in the pass band, zero gain at
/// the Nyquist frequency and almost no lag for periods longer than `period`,
/// in particular zero steady state lag on a linear ramp.
///
/// Reference: John Ehlers, "The Ultimate Smoother",
/// Technical Analysis of Stocks & Commodities, April 2024:
///
/// ```text
/// a1 = exp(-1.414 pi / P)
/// b1 = 2 a1 cos(1.414 pi / P)
/// c2 = b1, c3 = -a1^2, c1 = (1 + c2 - c3) / 4
/// US = (1 - c1) x + (2 c1 - c2) x[1] - (c1 + c3) x[2] + c2 US[1] + c3 US[2]
/// ```
///
/// As in the published code (`If CurrentBar < 4 Then US = Price`), the first
/// three outputs equal the input, which starts the filter at rest.
/// `pi` is exact instead of `3.14159`.
/// `last()` is `None` for the first `period - 1` values.
///
/// Output range: unbounded, follows the input. Constant input yields the constant.
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct UltimateSmoother<T, V> {
    view: V,
    /// The critical period: shorter periods are attenuated.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    core: UltimateCore<T>,
    /// Number of values received from the inner view.
    n: usize,
}

impl<T, V> UltimateSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new UltimateSmoother with a chained `View`.
    ///
    /// # Panics
    /// If `period < 2`, as shorter periods are above the Nyquist frequency.
    pub fn new(view: V, period: NonZeroUsize) -> Self {
        assert!(period.get() >= 2, "period must be >= 2");
        Self {
            view,
            period,
            core: UltimateCore::new(T::from(period.get()).expect("can convert")),
            n: 0,
        }
    }
}

impl<T, V> View<T> for UltimateSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let _ = self.core.update(val);
        self.n = self.n.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < self.period.get() {
            return None;
        }
        let out = self.core.value();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
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

    fn us(period: usize) -> UltimateSmoother<f64, Echo<f64>> {
        UltimateSmoother::new(Echo::new(), NonZeroUsize::new(period).unwrap())
    }

    /// Steady state amplitude of the filter output for a unit sine input.
    fn sine_gain(period: usize, sine_period: f64) -> f64 {
        let mut f = us(period);
        let n = (40.0 * sine_period.max(period as f64)) as usize;
        let mut max_abs: f64 = 0.0;
        for i in 0..n {
            f.update((2.0 * PI * i as f64 / sine_period).sin());
            if i > n - 4 * sine_period as usize {
                max_abs = max_abs.max(f.last().unwrap().abs());
            }
        }
        max_abs
    }

    #[test]
    fn ultimate_smoother_plot() {
        let mut f = us(20);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/ultimate_smoother.png").unwrap();
    }

    #[test]
    fn ultimate_smoother_warmup() {
        let mut f = us(10);
        assert_eq!(f.period().get(), 10);
        for i in 0..9 {
            f.update(i as f64);
            assert_eq!(f.last(), None);
        }
        f.update(9.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn ultimate_smoother_hand_computed() {
        // Direct transcription of the published EasyLanguage code.
        let p = 10.0;
        // The published truncated pi, written as a ratio to keep clippy quiet.
        let ehlers_pi = 314_159.0 / 100_000.0;
        let a1 = (-1.414 * ehlers_pi / p).exp();
        let b1 = 2.0 * a1 * (1.414 * 180.0 / p).to_radians().cos();
        let (c2, c3) = (b1, -a1 * a1);
        let c1 = (1.0 + c2 - c3) / 4.0;
        let mut f = us(10);
        let rng = romu::Rng::from_seed_with_64bit(7);
        let vals: Vec<f64> = (0..40).map(|_| 100.0 + rng.f64() * 10.0).collect();
        let mut out: Vec<f64> = Vec::new();
        for (i, v) in vals.iter().enumerate() {
            let u = if i < 3 {
                *v
            } else {
                (1.0 - c1) * v + (2.0 * c1 - c2) * vals[i - 1] - (c1 + c3) * vals[i - 2]
                    + c2 * out[i - 1]
                    + c3 * out[i - 2]
            };
            out.push(u);
            f.update(*v);
            if let Some(got) = f.last() {
                // 3.14159 vs exact pi.
                assert!((got - u).abs() < 1e-4, "bar {i}: {got} vs {u}");
            }
        }
        assert!(f.last().is_some());
    }

    #[test]
    fn ultimate_smoother_constant_input() {
        let mut f = us(20);
        for _ in 0..200 {
            f.update(42.0);
        }
        assert!((f.last().unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn ultimate_smoother_ramp_lags_less_than_super_smoother() {
        let mut f = us(20);
        let mut ss = SuperSmoother::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut x = 0.0;
        for i in 0..1_000 {
            x = i as f64;
            f.update(x);
            ss.update(x);
        }
        let us_lag = x - f.last().unwrap();
        let ss_lag = x - ss.last().unwrap();
        assert!(us_lag.abs() < 1e-6, "UltimateSmoother lag {us_lag}");
        assert!(ss_lag > 1.0, "SuperSmoother lag {ss_lag}");
    }

    #[test]
    fn ultimate_smoother_frequency_response() {
        let slow = sine_gain(20, 400.0);
        assert!((0.98..1.02).contains(&slow), "slow gain {slow}");
        // Only two poles in the stop band, so the attenuation is moderate.
        let fast = sine_gain(20, 4.0);
        assert!(fast < 0.25, "fast gain {fast}");
        let mid = sine_gain(20, 8.0);
        assert!(mid > fast, "mid gain {mid}");
        assert!(mid < 0.8, "mid gain {mid}");
        // Zero gain at Nyquist: an alternating input is removed completely.
        let mut f = us(20);
        for i in 0..1_000 {
            f.update(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        assert!(f.last().unwrap().abs() < 1e-9);
    }

    #[test]
    fn ultimate_smoother_core_fractional_period() {
        let mut a = UltimateCore::new(20.0);
        let mut b = UltimateCore::new(20.5);
        let mut f = us(20);
        for v in TEST_DATA {
            f.update(v);
            let (x, y) = (a.update(v), b.update(v));
            assert_eq!(f.last().unwrap_or(x), x);
            assert!((x - y).abs() < 0.05 * v);
        }
        assert_ne!(a.value(), b.value());
    }

    #[test]
    #[should_panic(expected = "period must be >= 2")]
    fn ultimate_smoother_period_too_small() {
        let _ = us(1);
    }

    #[test]
    #[should_panic(expected = "period must be >= 2")]
    fn ultimate_smoother_core_period_too_small() {
        let _ = UltimateCore::new(1.99);
    }
}

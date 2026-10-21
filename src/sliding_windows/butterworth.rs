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

/// Number of poles of a [`Butterworth`] filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButterworthPoles {
    /// Two poles: 12 dB per octave attenuation above the cutoff frequency.
    Two,
    /// Three poles: 18 dB per octave, but more lag.
    Three,
}

/// Length of the input and filter histories.
const HIST: usize = 3;

/// John Ehlers two- or three-pole Butterworth low-pass filter.
///
/// Maximally flat pass band; the input is smoothed by a binomial FIR
/// (`x + 2x[1] + x[2]` or `x + 3x[1] + 3x[2] + x[3]`), which puts all zeros at
/// the Nyquist frequency, so a two-bar cycle is removed completely.
///
/// Reference: John Ehlers, "Gaussian and Other Low Lag Filters" (2002),
/// <https://www.mesasoftware.com/papers/GaussianFilters.pdf>, and
/// "Cybernetic Analysis for Stocks and Futures" (2004):
///
/// ```text
/// Two poles:
///   a = exp(-1.414 pi / P), b = 2 a cos(1.414 pi / P)
///   f = b f[1] - a^2 f[2] + (1 - b + a^2) / 4 (x + 2x[1] + x[2])
/// Three poles:
///   a = exp(-pi / P), b = 2 a cos(1.738 pi / P), c = a^2
///   f = (b + c) f[1] - (c + b c) f[2] + c^2 f[3]
///       + (1 - b + c)(1 - c) / 8 (x + 3x[1] + 3x[2] + x[3])
/// ```
///
/// The cosine arguments are `1.414 * 180 / P` and `1.738 * 180 / P` degrees in
/// the paper. Deviation: the input and filter histories are seeded with the
/// first input, so the filter starts at rest instead of rising from 0.
/// `last()` is `None` for the first `period - 1` values.
///
/// Output range: unbounded, follows the input. Constant input yields the constant.
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Butterworth<T, V> {
    view: V,
    /// The critical period: shorter periods are attenuated.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    /// The number of poles.
    #[getset(get_copy = "pub")]
    poles: ButterworthPoles,
    /// Coefficient of the current input.
    c_x0: T,
    /// Coefficients of `x[1]..=x[3]`.
    c_x: [T; HIST],
    /// Coefficients of `f[1]..=f[3]`.
    c_f: [T; HIST],
    /// Input history `x[1]..=x[3]`.
    xs: [T; HIST],
    /// Filter history `f[1]..=f[3]`.
    fs: [T; HIST],
    /// Number of values received from the inner view.
    n: usize,
}

impl<T, V> Butterworth<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Butterworth filter with a chained `View`.
    ///
    /// # Panics
    /// If `period < 2`, as shorter periods are above the Nyquist frequency.
    pub fn new(view: V, period: NonZeroUsize, poles: ButterworthPoles) -> Self {
        assert!(period.get() >= 2, "period must be >= 2");
        let p = T::from(period.get()).expect("can convert");
        let pi = T::from(PI).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let three = T::from(3.0).expect("can convert");
        let zero = T::zero();
        let (c_x0, c_x, c_f) = match poles {
            ButterworthPoles::Two => {
                let a = (-T::from(1.414).expect("can convert") * pi / p).exp();
                let b = two * a * (T::from(1.414).expect("can convert") * pi / p).cos();
                let g = (T::one() - b + a * a) / T::from(4.0).expect("can convert");
                (g, [two * g, g, zero], [b, -a * a, zero])
            }
            ButterworthPoles::Three => {
                let a = (-pi / p).exp();
                let b = two * a * (T::from(1.738).expect("can convert") * pi / p).cos();
                let c = a * a;
                let g = (T::one() - b + c) * (T::one() - c) / T::from(8.0).expect("can convert");
                (g, [three * g, three * g, g], [b + c, -(c + b * c), c * c])
            }
        };
        assert!(c_x0 > zero, "input gain must be positive");
        Self {
            view,
            period,
            poles,
            c_x0,
            c_x,
            c_f,
            xs: [zero; HIST],
            fs: [zero; HIST],
            n: 0,
        }
    }
}

impl<T, V> View<T> for Butterworth<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.n == 0 {
            // Start at rest, see struct docs.
            self.xs = [val; HIST];
            self.fs = [val; HIST];
        }
        let mut f = self.c_x0 * val;
        for k in 0..HIST {
            f = f + self.c_x[k] * self.xs[k] + self.c_f[k] * self.fs[k];
        }
        debug_assert!(f.is_finite(), "filter output must be finite");
        self.xs.rotate_right(1);
        self.xs[0] = val;
        self.fs.rotate_right(1);
        self.fs[0] = f;
        self.n = self.n.saturating_add(1);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < self.period.get() {
            return None;
        }
        let out = self.fs[0];
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
        test_data::TEST_DATA,
    };

    const POLES: [ButterworthPoles; 2] = [ButterworthPoles::Two, ButterworthPoles::Three];

    fn bw(period: usize, poles: ButterworthPoles) -> Butterworth<f64, Echo<f64>> {
        Butterworth::new(Echo::new(), NonZeroUsize::new(period).unwrap(), poles)
    }

    /// Steady state amplitude for a unit sine with an integer period (RMS over whole cycles).
    fn sine_gain(period: usize, poles: ButterworthPoles, sine_period: usize) -> f64 {
        let mut f = bw(period, poles);
        let n = 100 * sine_period.max(period);
        let mut sum_sq = 0.0;
        for i in 0..n {
            f.update((2.0 * PI * i as f64 / sine_period as f64).sin());
            if i >= n - 10 * sine_period {
                sum_sq += f.last().unwrap().powi(2);
            }
        }
        (2.0 * sum_sq / (10 * sine_period) as f64).sqrt()
    }

    #[test]
    fn butterworth_plot() {
        let mut f = bw(20, ButterworthPoles::Three);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/butterworth.png").unwrap();
    }

    #[test]
    fn butterworth_warmup() {
        for poles in POLES {
            let mut f = bw(10, poles);
            assert_eq!(f.period().get(), 10);
            assert_eq!(f.poles(), poles);
            for i in 0..9 {
                f.update(i as f64);
                assert_eq!(f.last(), None);
            }
            f.update(9.0);
            assert!(f.last().is_some());
        }
    }

    #[test]
    fn butterworth_matches_paper_recursions() {
        let p = 15.0_f64;
        for poles in POLES {
            let mut f = bw(15, poles);
            let x0 = TEST_DATA[0];
            let (mut x1, mut x2, mut x3) = (x0, x0, x0);
            let (mut f1, mut f2, mut f3) = (x0, x0, x0);
            for (i, x) in TEST_DATA.iter().enumerate() {
                // The paper's code, cosine arguments in degrees.
                let g = match poles {
                    ButterworthPoles::Two => {
                        let a = (-1.414 * PI / p).exp();
                        let b = 2.0 * a * (1.414 * 180.0 / p).to_radians().cos();
                        b * f1 - a * a * f2 + (1.0 - b + a * a) / 4.0 * (x + 2.0 * x1 + x2)
                    }
                    ButterworthPoles::Three => {
                        let a = (-PI / p).exp();
                        let b = 2.0 * a * (1.738 * 180.0 / p).to_radians().cos();
                        let c = a * a;
                        (b + c) * f1 - (c + b * c) * f2
                            + c * c * f3
                            + (1.0 - b + c) * (1.0 - c) / 8.0 * (x + 3.0 * x1 + 3.0 * x2 + x3)
                    }
                };
                (x3, x2, x1) = (x2, x1, *x);
                (f3, f2, f1) = (f2, f1, g);
                f.update(*x);
                if let Some(got) = f.last() {
                    assert!((got - g).abs() < 1e-9, "{poles:?} bar {i}: {got} vs {g}");
                }
            }
        }
    }

    #[test]
    fn butterworth_constant_input() {
        for poles in POLES {
            let mut f = bw(20, poles);
            for _ in 0..100 {
                f.update(42.0);
            }
            assert!((f.last().unwrap() - 42.0).abs() < 1e-10, "{poles:?}");
        }
    }

    #[test]
    fn butterworth_frequency_response() {
        for poles in POLES {
            let slow = sine_gain(20, poles, 400);
            assert!((0.99..1.01).contains(&slow), "{poles:?}: slow gain {slow}");
            let at_period = sine_gain(20, poles, 20);
            assert!(
                (0.5..0.9).contains(&at_period),
                "{poles:?}: gain {at_period}"
            );
            // Zeros at Nyquist remove an alternating input completely.
            let mut f = bw(20, poles);
            for i in 0..1_000 {
                f.update(if i % 2 == 0 { 1.0 } else { -1.0 });
            }
            assert!(f.last().unwrap().abs() < 1e-9, "{poles:?}");
        }
        let two = sine_gain(20, ButterworthPoles::Two, 6);
        let three = sine_gain(20, ButterworthPoles::Three, 6);
        assert!(three < two, "two {two}, three {three}");
        assert!(three < 0.05, "three {three}");
    }

    #[test]
    #[should_panic(expected = "period must be >= 2")]
    fn butterworth_period_too_small() {
        let _ = bw(1, ButterworthPoles::Two);
    }
}

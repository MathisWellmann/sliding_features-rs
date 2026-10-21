// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! JMA - Jurik Moving Average (community reconstruction)

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// JMA - Jurik Moving Average (Mark Jurik, "JMA - Jurik Moving Average",
/// Technical Analysis of Stocks & Commodities, 1999): a low-lag adaptive
/// filter.
///
/// Jurik Research has never published the original algorithm; this is the
/// widely-circulated three-stage reconstruction (`beta`/`alpha`/
/// `phase_ratio` filter chain) used by `pandas-ta`'s `jma()` core loop,
/// TradingView's Pine `jma()` libraries and most MQL ports, reverse
/// engineered from <https://c.mql5.com/forextsd/forum/164/jurik_1.pdf>. It
/// is a close approximation, not a bit-exact match of Jurik's proprietary
/// binary; in particular it omits the original's extra adaptive-volatility
/// band stage, which is itself not fully specified in the open literature.
///
/// `beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)`, `alpha =
/// beta^power` (`period = window_len`, `power` in `1..=4`, default `2`),
/// `phase_ratio = clamp(phase / 100 + 1.5, 0.5, 2.5)` (`phase` in `[-100,
/// 100]`, default `0`, clamped rather than rejected outside that range, as
/// in every reference implementation above).
///
/// `e0 = (1 - alpha) * x + alpha * e0[1]`
/// `e1 = (x - e0) * (1 - beta) + beta * e1[1]`
/// `e2 = (e0 + phase_ratio * e1 - JMA[1]) * (1 - alpha)^2 + alpha^2 * e2[1]`
/// `JMA = JMA[1] + e2`
///
/// Seeded with `e0 = JMA = ` the first input and `e1 = e2 = 0`, so
/// `last()` is `Some` from the very first update (no warm-up) and a
/// constant input is reproduced exactly from the first value onward.
///
/// Complexity: O(1) per update; O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Jma<T, V> {
    view: V,
    /// The non-zero period used to derive `beta`.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The kernel exponent, in `1..=4`.
    #[getset(get_copy = "pub")]
    power: i32,
    phase: T,
    beta: T,
    alpha: T,
    phase_ratio: T,
    e0: T,
    e1: T,
    e2: T,
    last_jma: Option<T>,
}

impl<T, V> Jma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new JMA with a chained `View`, a period and the defaults
    /// `phase = 0`, `power = 2`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self::with_params(view, window_len, T::zero(), 2)
    }

    /// Create a new JMA with a chained `View`, a period, a phase (`[-100,
    /// 100]`, clamped outside that range) and a kernel exponent `power`.
    ///
    /// # Panics
    /// If `power` is not in `1..=4`.
    pub fn with_params(view: V, window_len: NonZeroUsize, phase: T, power: i32) -> Self {
        assert!(power >= 1, "power must be >= 1");
        assert!(power <= 4, "power must be <= 4");

        let len = T::from(window_len.get() - 1).expect("can convert");
        let zero_point_forty_five = T::from(0.45).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let beta = zero_point_forty_five * len / (zero_point_forty_five * len + two);
        let alpha = beta.powi(power);

        let hundred = T::from(100.0).expect("can convert");
        let one_half = T::from(0.5).expect("can convert");
        let two_point_five = T::from(2.5).expect("can convert");
        let one_point_five = T::from(1.5).expect("can convert");
        let phase_ratio = (phase / hundred + one_point_five)
            .max(one_half)
            .min(two_point_five);

        Self {
            view,
            window_len,
            power,
            phase,
            beta,
            alpha,
            phase_ratio,
            e0: T::zero(),
            e1: T::zero(),
            e2: T::zero(),
            last_jma: None,
        }
    }

    /// The configured phase, in `[-100, 100]`.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> T {
        self.phase
    }
}

impl<T, V> View<T> for Jma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(prev_jma) = self.last_jma else {
            // Seed so a constant input is reproduced exactly from the first
            // value onward.
            self.e0 = val;
            self.last_jma = Some(val);
            return;
        };

        self.e0 = (T::one() - self.alpha) * val + self.alpha * self.e0;
        self.e1 = (val - self.e0) * (T::one() - self.beta) + self.beta * self.e1;
        let one_minus_alpha = T::one() - self.alpha;
        self.e2 =
            (self.e0 + self.phase_ratio * self.e1 - prev_jma) * one_minus_alpha * one_minus_alpha
                + self.alpha * self.alpha * self.e2;
        self.last_jma = Some(prev_jma + self.e2);
    }

    fn last(&self) -> Option<T> {
        self.last_jma.inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
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

    #[test]
    fn jma_is_ready_from_the_first_value() {
        let mut jma = Jma::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        jma.update(1.0);
        assert_eq!(jma.last(), Some(1.0));
    }

    #[test]
    fn jma_constant_input_is_the_constant() {
        let mut jma = Jma::new(Echo::new(), NonZeroUsize::new(14).unwrap());
        for _ in 0..30 {
            jma.update(4.5);
        }
        assert_eq!(jma.last(), Some(4.5));
    }

    /// Naive reference: the beta/alpha/phase_ratio filter chain, applied
    /// directly, step by step.
    #[test]
    fn jma_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(345);
        let window_len = 10usize;
        let phase = 20.0_f64;
        let power = 2_i32;

        let len = (window_len - 1) as f64;
        let beta = 0.45 * len / (0.45 * len + 2.0);
        let alpha = beta.powi(power);
        let phase_ratio = (phase / 100.0 + 1.5).clamp(0.5, 2.5);

        let mut jma = Jma::with_params(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            phase,
            power,
        );

        let mut price = 100.0_f64;
        let (mut e0, mut e1, mut e2) = (0.0, 0.0, 0.0);
        let mut naive: Option<f64> = None;
        for i in 0..200 {
            price *= 1.0 + 0.01 * (rng.f64() - 0.5);
            jma.update(price);

            naive = Some(match naive {
                None => {
                    e0 = price;
                    price
                }
                Some(prev_jma) => {
                    e0 = (1.0 - alpha) * price + alpha * e0;
                    e1 = (price - e0) * (1.0 - beta) + beta * e1;
                    let one_minus_alpha = 1.0 - alpha;
                    e2 = (e0 + phase_ratio * e1 - prev_jma) * one_minus_alpha * one_minus_alpha
                        + alpha * alpha * e2;
                    prev_jma + e2
                }
            });
            let got = jma.last().expect("ready from the first value");
            assert!((got - naive.expect("set above")).abs() < 1e-9, "step {i}");
        }
    }

    #[test]
    #[should_panic(expected = "power must be <= 4")]
    fn power_above_four_is_rejected() {
        let _ = Jma::<f64, _>::with_params(Echo::new(), NonZeroUsize::new(10).unwrap(), 0.0, 5);
    }

    #[test]
    #[should_panic(expected = "power must be >= 1")]
    fn power_below_one_is_rejected() {
        let _ = Jma::<f64, _>::with_params(Echo::new(), NonZeroUsize::new(10).unwrap(), 0.0, 0);
    }

    #[test]
    fn extreme_phase_is_clamped() {
        // Both an extremely negative and an extremely positive phase must
        // produce a finite, usable series rather than panic or diverge.
        for phase in [-1000.0, 1000.0] {
            let mut jma = Jma::with_params(Echo::new(), NonZeroUsize::new(10).unwrap(), phase, 2);
            for v in [1.0, 2.0, 3.0, 2.0, 1.0] {
                jma.update(v);
                assert!(jma.last().expect("ready").is_finite());
            }
        }
    }

    #[test]
    fn jma_plot() {
        let mut jma = Jma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            jma.update(*v);
            if let Some(val) = jma.last() {
                out.push(val);
            }
        }
        let filename = "img/jma.png";
        plot_values(out, filename).unwrap();
    }
}

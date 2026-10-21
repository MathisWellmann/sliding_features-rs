// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! TEMA - Triple Exponential Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Ema,
};

/// TEMA - Triple Exponential Moving Average (Patrick Mulloy, "Smoothing Data
/// with Faster Moving Averages", Technical Analysis of Stocks & Commodities,
/// 1994).
///
/// `TEMA = 3*EMA(x) - 3*EMA(EMA(x)) + EMA(EMA(EMA(x)))`, all three EMAs
/// sharing `window_len`. Reduces lag further than [`crate::sliding_windows::Dema`]
/// at the cost of more overshoot on sharp reversals.
///
/// `last()` is `None` until the innermost `EMA(EMA(EMA(x)))` is ready, which
/// needs about `3 * window_len - 2` input values; output is unbounded,
/// following the input.
///
/// Complexity: O(1) per update (three chained EMA updates); O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Tema<T, V> {
    view: V,
    /// The non-zero window length shared by all three inner EMAs.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    ema1: Ema<T, Echo<T>>,
    ema2: Ema<T, Echo<T>>,
    ema3: Ema<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Tema<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new TEMA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            ema1: Ema::new(Echo::new(), window_len),
            ema2: Ema::new(Echo::new(), window_len),
            ema3: Ema::new(Echo::new(), window_len),
            out: None,
        }
    }
}

impl<T, V> View<T> for Tema<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.ema1.update(val);
        let Some(e1) = self.ema1.last() else { return };
        self.ema2.update(e1);
        let Some(e2) = self.ema2.last() else { return };
        self.ema3.update(e2);
        let Some(e3) = self.ema3.last() else { return };

        let three = T::from(3.0).expect("can convert");
        self.out = Some(three * e1 - three * e2 + e3);
    }

    fn last(&self) -> Option<T> {
        self.out.inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    #[test]
    fn tema_warms_up() {
        let mut tema = Tema::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            tema.update(v);
            assert_eq!(tema.last(), None);
        }
        for v in TEST_DATA {
            tema.update(v);
        }
        assert!(tema.last().is_some());
    }

    #[test]
    fn tema_constant_input_is_the_constant() {
        let mut tema = Tema::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..60 {
            tema.update(3.5);
        }
        assert!((tema.last().expect("ready") - 3.5).abs() < 1e-9);
    }

    /// Naive reference: three independent EMA recursions, combined as
    /// `3*e1 - 3*e2 + e3`.
    #[test]
    fn tema_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(33);
        let window_len = 9usize;
        let alpha = 2.0 / (window_len as f64 + 1.0);
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 50.0).collect();

        let mut tema = Tema::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let (mut e1, mut e2, mut e3): (Option<f64>, Option<f64>, Option<f64>) = (None, None, None);
        for v in vals.iter() {
            tema.update(*v);
            e1 = Some(match e1 {
                None => *v,
                Some(prev) => v * alpha + prev * (1.0 - alpha),
            });
            e2 = Some(match e2 {
                None => e1.expect("set above"),
                Some(prev) => e1.expect("set above") * alpha + prev * (1.0 - alpha),
            });
            e3 = Some(match e3 {
                None => e2.expect("set above"),
                Some(prev) => e2.expect("set above") * alpha + prev * (1.0 - alpha),
            });
        }
        let expected = 3.0 * e1.expect("set") - 3.0 * e2.expect("set") + e3.expect("set");
        assert!((tema.last().expect("ready") - expected).abs() < 1e-6);
    }

    /// TEMA follows a ramp with less lag than a plain EMA of the same length:
    /// both converge to zero steady-state lag on a ramp in theory, but EMA's
    /// single-pole filter still trails visibly while TEMA has cancelled it.
    #[test]
    fn tema_lags_less_than_ema_on_a_ramp() {
        let window_len = NonZeroUsize::new(10).unwrap();
        let mut tema = Tema::new(Echo::new(), window_len);
        let mut ema = Ema::new(Echo::new(), window_len);
        let ramp: Vec<f64> = (0..300).map(|i| i as f64).collect();
        for v in &ramp {
            tema.update(*v);
            ema.update(*v);
        }
        let i = ramp.len() - 1;
        let tema_lag = (i as f64 - tema.last().expect("ready")).abs();
        let ema_lag = i as f64 - ema.last().expect("ready");
        assert!(tema_lag < ema_lag, "tema {tema_lag} ema {ema_lag}");
    }

    #[test]
    fn tema_plot() {
        let mut tema = Tema::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            tema.update(*v);
            if let Some(val) = tema.last() {
                out.push(val);
            }
        }
        let filename = "img/tema.png";
        plot_values(out, filename).unwrap();
    }
}

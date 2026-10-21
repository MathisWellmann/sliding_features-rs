// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! TRIX - Triple Exponential Average rate of change

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::Ema;
use crate::{
    View,
    pure_functions::Echo,
};

/// TRIX - Triple Exponential Average (Hutson, 1983; TA-Lib `TRIX`).
///
/// Smooths the input with a triple-cascaded EMA of `period`,
/// `ema3 = EMA(EMA(EMA(x, n), n), n)`, and outputs its 1-bar rate of change
/// in percent: `TRIX = 100 * (ema3 - ema3_prev) / ema3_prev`.
///
/// `last()` is `None` until the third EMA is ready and one more value has
/// been observed to form the rate of change (warm-up: `3 * (period - 1) + 2`
/// values).
///
/// Output range: unbounded (expressed in percent); oscillates around 0.
///
/// Complexity: O(1) per update (three chained EMAs), O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Trix<T, V> {
    view: V,
    /// The EMA period used for all three cascaded smoothings.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    ema1: Ema<T, Echo<T>>,
    ema2: Ema<T, Echo<T>>,
    ema3: Ema<T, Echo<T>>,
    prev_ema3: Option<T>,
    out: Option<T>,
}

impl<T, V> Trix<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new TRIX with a chained `View` and a given EMA period.
    pub fn new(view: V, period: NonZeroUsize) -> Self {
        Self {
            view,
            period,
            ema1: Ema::new(Echo::new(), period),
            ema2: Ema::new(Echo::new(), period),
            ema3: Ema::new(Echo::new(), period),
            prev_ema3: None,
            out: None,
        }
    }
}

impl<T, V> View<T> for Trix<T, V>
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

        let Some(prev) = self.prev_ema3 else {
            self.prev_ema3 = Some(e3);
            return;
        };
        self.prev_ema3 = Some(e3);
        // A zero previous value makes the percent change undefined, so TRIX
        // simply stays at its previous output (there is no natural neutral
        // value to fall back to).
        if prev == T::zero() {
            return;
        }
        let trix = T::from(100.0).expect("can convert") * (e3 - prev) / prev;
        debug_assert!(trix.is_finite(), "`trix` must be finite");
        self.out = Some(trix);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    /// Naive reference: plain EMA recurrence, alpha = 2 / (n + 1). Mirrors
    /// `Ema`'s gating: `update` returns `None` until `period` values were fed.
    struct NaiveEma {
        period: usize,
        val: Option<f64>,
        n: usize,
    }
    impl NaiveEma {
        fn new(period: usize) -> Self {
            Self {
                period,
                val: None,
                n: 0,
            }
        }
        fn update(&mut self, x: f64) -> Option<f64> {
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let out = match self.val {
                None => x,
                Some(prev) => alpha * x + (1.0 - alpha) * prev,
            };
            self.val = Some(out);
            self.n += 1;
            (self.n >= self.period).then_some(out)
        }
    }

    #[test]
    fn trix_plot() {
        let mut trix = Trix::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            trix.update(*v);
            if let Some(val) = trix.last() {
                out.push(val);
            }
        }
        let filename = "img/trix.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn trix_warms_up() {
        let mut trix = Trix::new(Echo::new(), NonZeroUsize::new(2).unwrap());
        assert_eq!(trix.last(), None);
        let mut seen = false;
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            trix.update(v);
            if trix.last().is_some() {
                seen = true;
            }
        }
        assert!(seen, "TRIX should warm up within 10 steps for period 2");
    }

    #[test]
    fn trix_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(17);
        let period = 5;
        let mut trix = Trix::new(Echo::new(), NonZeroUsize::new(period).unwrap());
        let mut naive1 = NaiveEma::new(period);
        let mut naive2 = NaiveEma::new(period);
        let mut naive3 = NaiveEma::new(period);
        let mut prev_e3: Option<f64> = None;
        for _ in 0..2_000 {
            let x = rng.f64() * 100.0 + 50.0;
            trix.update(x);
            // Each stage only feeds the next once it is itself ready, just
            // like `Trix::update` only forwards a `Some` value downstream.
            let e3 = naive1
                .update(x)
                .and_then(|e1| naive2.update(e1))
                .and_then(|e2| naive3.update(e2));
            if let (Some(got), Some(e3)) = (trix.last(), e3) {
                let prev = prev_e3.expect("trix is only Some once a previous ema3 exists");
                let naive_trix = 100.0 * (e3 - prev) / prev;
                assert!((got - naive_trix).abs() < 1e-9);
            }
            if let Some(e3) = e3 {
                prev_e3 = Some(e3);
            }
        }
    }

    #[test]
    fn trix_constant_input_converges_to_zero() {
        let mut trix = Trix::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..500 {
            trix.update(42.0);
        }
        assert!(trix.last().unwrap().abs() < 1e-6);
    }
}

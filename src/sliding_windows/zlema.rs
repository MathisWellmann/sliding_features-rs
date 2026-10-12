//! ZLEMA - Zero-Lag Exponential Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::{
        Ema,
        Lag,
    },
};

/// ZLEMA - Zero-Lag Exponential Moving Average (John Ehlers & Ric Way, "Zero
/// Lag (Well, Almost)", Technical Analysis of Stocks & Commodities, 2010).
///
/// An EMA of the de-lagged series `2*x - x[lag]`, with `lag = (n - 1) / 2`
/// (integer division, i.e. rounded down) applied to the window length `n`.
/// Removing an estimate of the EMA's own lag from the input largely cancels
/// it out, at the cost of overshoot on sharp reversals, similar to DEMA/TEMA.
///
/// `last()` is `None` until `lag + window_len` input values were observed;
/// output is unbounded, following the input.
///
/// Complexity: O(1) per update (a fixed-size lag buffer plus one EMA
/// update); O(lag) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Zlema<T, V> {
    view: V,
    /// The non-zero window length of the inner EMA.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    lag: Lag<T, Echo<T>>,
    ema: Ema<T, Echo<T>>,
}

impl<T, V> Zlema<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new ZLEMA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        // `Lag::new(_, k)` reports the value from `k - 1` steps ago, so
        // `k = lag + 1` is needed to reach back exactly `lag` steps.
        let lag = (window_len.get() - 1) / 2;
        let lag_window = NonZeroUsize::new(lag + 1).expect("lag + 1 is always >= 1");
        Self {
            view,
            window_len,
            lag: Lag::new(Echo::new(), lag_window),
            ema: Ema::new(Echo::new(), window_len),
        }
    }
}

impl<T, V> View<T> for Zlema<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.lag.update(val);
        let Some(lagged) = self.lag.last() else {
            return;
        };
        let two = T::from(2.0).expect("can convert");
        let de_lagged = two * val - lagged;
        debug_assert!(de_lagged.is_finite(), "value must be finite");
        self.ema.update(de_lagged);
    }

    fn last(&self) -> Option<T> {
        self.ema.last().inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    #[test]
    fn zlema_warms_up() {
        let mut zlema = Zlema::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        // lag = (5-1)/2 = 2, so ready after lag + window_len = 7 inputs.
        for v in &TEST_DATA[..6] {
            zlema.update(*v);
            assert_eq!(zlema.last(), None);
        }
        zlema.update(TEST_DATA[6]);
        assert!(zlema.last().is_some());
    }

    #[test]
    fn zlema_constant_input_is_the_constant() {
        let mut zlema = Zlema::new(Echo::new(), NonZeroUsize::new(6).unwrap());
        for _ in 0..30 {
            zlema.update(4.0);
        }
        assert!((zlema.last().expect("ready") - 4.0).abs() < 1e-9);
    }

    /// Naive reference: de-lag with a `lag`-step-old value, then recurse the
    /// EMA formula directly.
    #[test]
    fn zlema_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(55);
        let window_len = 8usize;
        let lag = (window_len - 1) / 2;
        let ready_at = lag + window_len - 1;
        let alpha = 2.0 / (window_len as f64 + 1.0);
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 50.0).collect();

        let mut zlema = Zlema::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let mut naive: Option<f64> = None;
        for (i, v) in vals.iter().enumerate() {
            zlema.update(*v);
            if i < lag {
                assert_eq!(zlema.last(), None);
                continue;
            }
            let de_lagged = 2.0 * v - vals[i - lag];
            naive = Some(match naive {
                None => de_lagged,
                Some(prev) => de_lagged * alpha + prev * (1.0 - alpha),
            });
            if i < ready_at {
                assert_eq!(zlema.last(), None);
                continue;
            }
            let got = zlema.last().expect("ready");
            assert!((got - naive.expect("set above")).abs() < 1e-9, "step {i}");
        }
    }

    /// ZLEMA follows a ramp with less lag than a plain EMA of the same length.
    #[test]
    fn zlema_lags_less_than_ema_on_a_ramp() {
        let window_len = NonZeroUsize::new(10).unwrap();
        let mut zlema = Zlema::new(Echo::new(), window_len);
        let mut ema = Ema::new(Echo::new(), window_len);
        let ramp: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for v in &ramp {
            zlema.update(*v);
            ema.update(*v);
        }
        let i = ramp.len() - 1;
        let zlema_lag = (i as f64 - zlema.last().expect("ready")).abs();
        let ema_lag = i as f64 - ema.last().expect("ready");
        assert!(zlema_lag < ema_lag, "zlema {zlema_lag} ema {ema_lag}");
    }

    #[test]
    fn zlema_plot() {
        let mut zlema = Zlema::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            zlema.update(*v);
            if let Some(val) = zlema.last() {
                out.push(val);
            }
        }
        let filename = "img/zlema.png";
        plot_values(out, filename).unwrap();
    }
}

//! DEMA - Double Exponential Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Ema,
};

/// DEMA - Double Exponential Moving Average (Patrick Mulloy, "Smoothing Data
/// with Faster Moving Averages", Technical Analysis of Stocks & Commodities,
/// 1994).
///
/// `DEMA = 2 * EMA(x) - EMA(EMA(x))`, both EMAs sharing `window_len`. Reduces
/// the lag of a plain EMA at the cost of overshoot on sharp reversals.
///
/// `last()` is `None` until the inner `EMA(EMA(x))` is ready, which needs
/// about `2 * window_len - 1` input values; output is unbounded, following
/// the input.
///
/// Complexity: O(1) per update (two chained EMA updates); O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Dema<T, V> {
    view: V,
    /// The non-zero window length shared by both inner EMAs.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    ema1: Ema<T, Echo<T>>,
    ema2: Ema<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Dema<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new DEMA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            ema1: Ema::new(Echo::new(), window_len),
            ema2: Ema::new(Echo::new(), window_len),
            out: None,
        }
    }
}

impl<T, V> View<T> for Dema<T, V>
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

        let two = T::from(2.0).expect("can convert");
        self.out = Some(two * e1 - e2);
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
        test_data::TEST_DATA,
    };

    #[test]
    fn dema_warms_up() {
        let mut dema = Dema::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0] {
            dema.update(v);
            assert_eq!(dema.last(), None);
        }
        for v in TEST_DATA {
            dema.update(v);
        }
        assert!(dema.last().is_some());
    }

    #[test]
    fn dema_constant_input_is_the_constant() {
        let mut dema = Dema::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..50 {
            dema.update(3.5);
        }
        assert!((dema.last().expect("ready") - 3.5).abs() < 1e-9);
    }

    /// Naive reference: two independent EMA recursions, combined as `2*e1 - e2`.
    #[test]
    fn dema_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(22);
        let window_len = 9usize;
        let alpha = 2.0 / (window_len as f64 + 1.0);
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 50.0).collect();

        let mut dema = Dema::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let (mut e1, mut e2): (Option<f64>, Option<f64>) = (None, None);
        for v in vals.iter() {
            dema.update(*v);
            e1 = Some(match e1 {
                None => *v,
                Some(prev) => v * alpha + prev * (1.0 - alpha),
            });
            e2 = Some(match e2 {
                None => e1.expect("set above"),
                Some(prev) => e1.expect("set above") * alpha + prev * (1.0 - alpha),
            });
        }
        let expected = 2.0 * e1.expect("set") - e2.expect("set");
        assert!((dema.last().expect("ready") - expected).abs() < 1e-6);
    }

    /// DEMA follows a ramp with less lag than a plain EMA of the same length.
    #[test]
    fn dema_lags_less_than_ema_on_a_ramp() {
        let window_len = NonZeroUsize::new(10).unwrap();
        let mut dema = Dema::new(Echo::new(), window_len);
        let mut ema = Ema::new(Echo::new(), window_len);
        let ramp: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for v in &ramp {
            dema.update(*v);
            ema.update(*v);
        }
        let i = ramp.len() - 1;
        let dema_lag = i as f64 - dema.last().expect("ready");
        let ema_lag = i as f64 - ema.last().expect("ready");
        assert!(dema_lag < ema_lag, "dema {dema_lag} ema {ema_lag}");
    }

    #[test]
    fn dema_plot() {
        let mut dema = Dema::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            dema.update(*v);
            if let Some(val) = dema.last() {
                out.push(val);
            }
        }
        let filename = "img/dema.png";
        plot_values(out, filename).unwrap();
    }
}

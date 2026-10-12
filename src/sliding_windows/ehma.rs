//! EHMA - Exponential Hull Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Ema,
};

/// EHMA - Exponential Hull Moving Average: the same construction as the Hull
/// Moving Average (Alan Hull, 2005), but with every WMA stage replaced by an
/// EMA: `EMA(2*EMA(n/2) - EMA(n), round(sqrt(n)))`.
///
/// `n/2` and `sqrt(n)` are rounded to the nearest integer (round half away
/// from zero, `num::Float::round`); some implementations instead floor the
/// half length. `n >= 2` is required so the half length is at least 1.
///
/// Like [`crate::sliding_windows::Hma`], it reduces lag relative to a plain
/// moving average of the same length, at the cost of overshoot on reversals;
/// using EMAs instead of WMAs further smooths the result and reacts faster
/// to the most recent values within each stage.
///
/// `last()` is `None` until the final EMA stage is ready, which needs about
/// `max(n, round(n/2)) + round(sqrt(n)) - 1` input values; output is
/// unbounded, following the input.
///
/// Complexity: O(1) per update (three chained EMA updates); O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Ehma<T, V> {
    view: V,
    /// The non-zero window length `n`.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The rounded half window length, `round(n / 2)`.
    #[getset(get_copy = "pub")]
    half_len: NonZeroUsize,
    /// The rounded square-root window length, `round(sqrt(n))`.
    #[getset(get_copy = "pub")]
    sqrt_len: NonZeroUsize,
    ema_half: Ema<T, Echo<T>>,
    ema_full: Ema<T, Echo<T>>,
    ema_final: Ema<T, Echo<T>>,
}

impl<T, V> Ehma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new EHMA with a chained `View` and a window length.
    ///
    /// # Panics
    /// If `window_len` is 1: the half-length would be below 1.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= 2,
            "exponential Hull MA needs a window length >= 2"
        );

        let n = T::from(window_len.get()).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let half_len = round_to_window_len(n / two);
        let sqrt_len = round_to_window_len(n.sqrt());

        Self {
            view,
            window_len,
            half_len,
            sqrt_len,
            ema_half: Ema::new(Echo::new(), half_len),
            ema_full: Ema::new(Echo::new(), window_len),
            ema_final: Ema::new(Echo::new(), sqrt_len),
        }
    }
}

/// Rounds `x` (expected to be a small positive value) to the nearest
/// non-zero window length.
fn round_to_window_len<T: Float>(x: T) -> NonZeroUsize {
    let rounded = x.round();
    debug_assert!(rounded >= T::one(), "a window length rounds to at least 1");
    let len = rounded
        .to_usize()
        .expect("rounded window lengths used here are small positive integers");
    NonZeroUsize::new(len).expect("rounds to at least 1")
}

impl<T, V> View<T> for Ehma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.ema_half.update(val);
        self.ema_full.update(val);
        let (Some(half), Some(full)) = (self.ema_half.last(), self.ema_full.last()) else {
            return;
        };
        let two = T::from(2.0).expect("can convert");
        let diff = two * half - full;
        debug_assert!(diff.is_finite(), "value must be finite");
        self.ema_final.update(diff);
    }

    fn last(&self) -> Option<T> {
        self.ema_final.last().inspect(|v| {
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
    fn ehma_warms_up() {
        let mut ehma = Ehma::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        assert_eq!(ehma.last(), None);
        for v in &TEST_DATA {
            ehma.update(*v);
        }
        assert!(ehma.last().is_some());
    }

    #[test]
    fn ehma_constant_input_is_the_constant() {
        let mut ehma = Ehma::new(Echo::new(), NonZeroUsize::new(9).unwrap());
        for _ in 0..30 {
            ehma.update(6.0);
        }
        assert!((ehma.last().expect("ready") - 6.0).abs() < 1e-9);
    }

    #[test]
    fn ehma_derives_half_and_sqrt_lengths() {
        let ehma = Ehma::<f64, _>::new(Echo::new(), NonZeroUsize::new(9).unwrap());
        assert_eq!(ehma.half_len(), NonZeroUsize::new(5).unwrap()); // round(9/2) = round(4.5) = 5
        assert_eq!(ehma.sqrt_len(), NonZeroUsize::new(3).unwrap()); // round(sqrt(9)) = 3
    }

    /// Naive reference: three independent EMA recursions, combined as
    /// `ema(2*ema_half - ema_full, sqrt_len)`.
    #[test]
    fn ehma_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(77);
        let window_len = 16usize;
        let half_len = 8usize;
        let sqrt_len = 4usize;
        let alpha_half = 2.0 / (half_len as f64 + 1.0);
        let alpha_full = 2.0 / (window_len as f64 + 1.0);
        let alpha_final = 2.0 / (sqrt_len as f64 + 1.0);
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 50.0).collect();

        let mut ehma = Ehma::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let (mut e_half, mut e_full, mut e_final): (Option<f64>, Option<f64>, Option<f64>) =
            (None, None, None);
        for v in vals.iter() {
            ehma.update(*v);
            e_half = Some(match e_half {
                None => *v,
                Some(prev) => v * alpha_half + prev * (1.0 - alpha_half),
            });
            e_full = Some(match e_full {
                None => *v,
                Some(prev) => v * alpha_full + prev * (1.0 - alpha_full),
            });
            let diff = 2.0 * e_half.expect("set above") - e_full.expect("set above");
            e_final = Some(match e_final {
                None => diff,
                Some(prev) => diff * alpha_final + prev * (1.0 - alpha_final),
            });
        }
        let expected = e_final.expect("set");
        assert!((ehma.last().expect("ready") - expected).abs() < 1e-6);
    }

    /// EHMA follows a ramp with less lag than a plain EMA of the same length.
    #[test]
    fn ehma_lags_less_than_ema_on_a_ramp() {
        let window_len = NonZeroUsize::new(16).unwrap();
        let mut ehma = Ehma::new(Echo::new(), window_len);
        let mut ema = Ema::new(Echo::new(), window_len);
        let ramp: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for v in &ramp {
            ehma.update(*v);
            ema.update(*v);
        }
        let i = ramp.len() - 1;
        let ehma_lag = (i as f64 - ehma.last().expect("ready")).abs();
        let ema_lag = i as f64 - ema.last().expect("ready");
        assert!(ehma_lag < ema_lag, "ehma {ehma_lag} ema {ema_lag}");
    }

    #[test]
    #[should_panic(expected = "window length >= 2")]
    fn window_len_of_one_is_rejected() {
        let _ = Ehma::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn ehma_plot() {
        let mut ehma = Ehma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            ehma.update(*v);
            if let Some(val) = ehma.last() {
                out.push(val);
            }
        }
        let filename = "img/ehma.png";
        plot_values(out, filename).unwrap();
    }
}

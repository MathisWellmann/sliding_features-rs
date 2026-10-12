//! RMA - Wilder's Smoothing Moving Average (a.k.a. SMMA)

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// RMA - Wilder's Smoothing Moving Average (a.k.a. SMMA), as used by J.
/// Welles Wilder Jr. for RSI and ATR ("New Concepts in Technical Trading
/// Systems", 1978).
///
/// An exponential moving average with `alpha = 1 / window_len`, seeded with
/// the plain SMA of the first `window_len` values. This matches the TA-Lib
/// convention (`TA_MA` with `MAType::Ema` and a `1/n` smoothing factor is
/// SMA-seeded) rather than Wilder's original hand-computed tables, which
/// effectively seed with the first value; the difference washes out after a
/// few multiples of `window_len`.
///
/// `last()` is `None` until `window_len` values were observed, then in the
/// range spanned by the input.
///
/// Complexity: O(1) per update once warmed up (O(window_len) total for the
/// warm-up sum); O(window_len) memory during warm-up, O(1) after.
#[derive(Debug, Clone, CopyGetters)]
pub struct Rma<T, V> {
    view: V,
    /// The non-zero window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    seed_vals: VecDeque<T>,
    seed_sum: T,
    last_rma: Option<T>,
}

impl<T, V> Rma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new RMA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            seed_vals: VecDeque::with_capacity(window_len.get()),
            seed_sum: T::zero(),
            last_rma: None,
        }
    }
}

impl<T, V> View<T> for Rma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        match self.last_rma {
            None => {
                self.seed_vals.push_back(val);
                self.seed_sum = self.seed_sum + val;
                debug_assert!(self.seed_vals.len() <= self.window_len.get());
                if self.seed_vals.len() == self.window_len.get() {
                    let n = T::from(self.window_len.get()).expect("can convert");
                    self.last_rma = Some(self.seed_sum / n);
                }
            }
            Some(prev) => {
                let n = T::from(self.window_len.get()).expect("can convert");
                let alpha = T::one() / n;
                self.last_rma = Some(val * alpha + prev * (T::one() - alpha));
            }
        }
    }

    fn last(&self) -> Option<T> {
        self.last_rma.inspect(|v| {
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
    fn rma_warms_up() {
        let mut rma = Rma::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        assert_eq!(rma.last(), None);
        rma.update(1.0);
        assert_eq!(rma.last(), None);
        rma.update(2.0);
        assert_eq!(rma.last(), None);
        rma.update(3.0);
        assert_eq!(rma.last(), Some(2.0));
        rma.update(4.0);
        assert!(rma.last().is_some());
    }

    #[test]
    fn rma_constant_input_is_the_constant() {
        let mut rma = Rma::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..20 {
            rma.update(7.0);
        }
        assert!((rma.last().expect("ready") - 7.0).abs() < 1e-12);
    }

    /// Naive reference: SMA-seed then `rma = alpha * x + (1 - alpha) * rma_prev`.
    #[test]
    fn rma_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        let window_len = 7usize;
        let vals: Vec<f64> = (0..500).map(|_| rng.f64() * 200.0 - 100.0).collect();

        let mut rma = Rma::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let mut naive: Option<f64> = None;
        for (i, v) in vals.iter().enumerate() {
            rma.update(*v);
            if i + 1 < window_len {
                assert_eq!(rma.last(), None);
                continue;
            }
            naive = Some(match naive {
                None => vals[..window_len].iter().sum::<f64>() / window_len as f64,
                Some(prev) => {
                    let alpha = 1.0 / window_len as f64;
                    v * alpha + prev * (1.0 - alpha)
                }
            });
            let got = rma.last().expect("ready");
            assert!((got - naive.expect("set above")).abs() < 1e-9, "step {i}");
        }
    }

    #[test]
    fn rma_plot() {
        let mut rma = Rma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rma.update(*v);
            if let Some(val) = rma.last() {
                out.push(val);
            }
        }
        let filename = "img/rma.png";
        plot_values(out, filename).unwrap();
    }
}

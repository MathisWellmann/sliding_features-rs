//! TRIMA - Triangular Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Sma,
};

/// TRIMA - Triangular Moving Average (TA-Lib's `TRIMA`): a moving average
/// whose weights rise linearly to a peak at the center of the window and
/// fall linearly back down.
///
/// Implemented, like TA-Lib, as a double SMA rather than by computing
/// triangular weights directly: for an odd window length `n`, let
/// `p = (n + 1) / 2`; TRIMA is `SMA(SMA(x, p), p)`. For an even `n`, let
/// `p = n / 2`; TRIMA is `SMA(SMA(x, p), p + 1)`. Both give a triangular
/// weight shape of total width `n` (a box of width `p` convolved with a box
/// of width `p` or `p + 1`), without ever computing the weights explicitly.
///
/// `last()` is `None` until `window_len` values were observed; output stays
/// within the minimum and maximum of the window, like any weighted average
/// of it.
///
/// Complexity: O(1) per update (two chained O(1) SMA updates); O(n) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Trima<T, V> {
    view: V,
    /// The non-zero window length `n`.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    sma1: Sma<T, Echo<T>>,
    sma2: Sma<T, Echo<T>>,
}

impl<T, V> Trima<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new TRIMA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        let n = window_len.get();
        let (p, q) = if n % 2 == 1 {
            let p = n.div_ceil(2);
            (p, p)
        } else {
            let p = n / 2;
            (p, p + 1)
        };
        let p = NonZeroUsize::new(p).expect("p >= 1 for n >= 1");
        let q = NonZeroUsize::new(q).expect("q >= 1 for n >= 1");

        Self {
            view,
            window_len,
            sma1: Sma::new(Echo::new(), p),
            sma2: Sma::new(Echo::new(), q),
        }
    }
}

impl<T, V> View<T> for Trima<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.sma1.update(val);
        let Some(s1) = self.sma1.last() else { return };
        self.sma2.update(s1);
    }

    fn last(&self) -> Option<T> {
        self.sma2.last().inspect(|v| {
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
    fn trima_warms_up() {
        let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in [1.0, 2.0, 3.0] {
            trima.update(v);
            assert_eq!(trima.last(), None);
        }
        trima.update(4.0);
        assert!(trima.last().is_some());
    }

    /// Hand-computed example for an even window length (n=4, p=2, q=3):
    /// `sma1` is a 2-box average, `sma2` is a 3-box average of `sma1`.
    #[test]
    fn trima_even_window_matches_hand_computed_example() {
        let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        let mut out = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            trima.update(v);
            out.push(trima.last());
        }
        assert_eq!(out, [None, None, None, Some(2.5), Some(3.5), Some(4.5)]);
    }

    /// Hand-computed example for an odd window length (n=3, p=2, q=2).
    #[test]
    fn trima_odd_window_matches_hand_computed_example() {
        let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        let mut out = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            trima.update(v);
            out.push(trima.last());
        }
        // sma1(2): 1.5, 2.5, 3.5, 4.5; sma2(2) of those: 2.0, 3.0, 4.0
        assert_eq!(out, [None, None, Some(2.0), Some(3.0), Some(4.0)]);
    }

    #[test]
    fn trima_constant_input_is_the_constant() {
        for window_len in [3, 4, 5, 8] {
            let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
            for _ in 0..(2 * window_len) {
                trima.update(9.0);
            }
            assert!((trima.last().expect("ready") - 9.0).abs() < 1e-9);
        }
    }

    /// Naive reference: double box-average via direct window slicing, for
    /// both odd and even window lengths.
    #[test]
    fn trima_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(88);
        let direct_sma = |window: &[f64]| window.iter().sum::<f64>() / window.len() as f64;

        for window_len in [5usize, 8usize] {
            let (p, q) = if window_len % 2 == 1 {
                let p = window_len.div_ceil(2);
                (p, p)
            } else {
                let p = window_len / 2;
                (p, p + 1)
            };
            let vals: Vec<f64> = (0..200).map(|_| rng.f64() * 50.0).collect();
            let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
            let mut sma1_vals: Vec<f64> = Vec::new();
            for (i, v) in vals.iter().enumerate() {
                trima.update(*v);
                if i + 1 >= p {
                    sma1_vals.push(direct_sma(&vals[i + 1 - p..=i]));
                }
                if sma1_vals.len() < q {
                    assert_eq!(trima.last(), None, "window {window_len} step {i}");
                    continue;
                }
                let expected = direct_sma(&sma1_vals[sma1_vals.len() - q..]);
                let got = trima.last().expect("ready");
                assert!(
                    (got - expected).abs() < 1e-9,
                    "window {window_len} step {i}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn trima_plot() {
        let mut trima = Trima::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            trima.update(*v);
            if let Some(val) = trima.last() {
                out.push(val);
            }
        }
        let filename = "img/trima.png";
        plot_values(out, filename).unwrap();
    }
}

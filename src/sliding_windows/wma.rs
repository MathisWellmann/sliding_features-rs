// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! WMA - Linearly Weighted Moving Average

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// WMA - Linearly Weighted Moving Average.
///
/// `WMA = sum_{i=1..n} i * x_i / (n (n + 1) / 2)` over the last `n = window_len`
/// values, where `x_n` is the newest value (weight `n`) and `x_1` the oldest (weight 1).
/// See e.g. <https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average>.
///
/// `last()` is `None` until `window_len` values were observed.
/// Output range: within the minimum and maximum of the window.
///
/// Complexity: amortized O(1) per update, O(window_len) memory.
/// A running sum `S` and running weighted sum `W` are updated as
/// `W' = W - S + n * x`, `S' = S - x_oldest + x`. To stop floating point
/// rounding errors from accumulating, both sums are recomputed from the window
/// every `window_len` updates, which costs O(1) amortized.
#[derive(Debug, Clone, CopyGetters)]
pub struct Wma<T, V> {
    view: V,
    /// The non-zero window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    /// Sum of the values in the window.
    sum: T,
    /// Sum of the values in the window, weighted by their position (oldest = 1).
    weighted_sum: T,
    /// `n (n + 1) / 2`, the sum of the weights of a full window.
    weight_total: T,
    /// Evictions since `sum` and `weighted_sum` were last recomputed exactly.
    evictions: usize,
}

impl<T, V> Wma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new linearly weighted moving average with a chained `View`
    /// and a given sliding window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        let n = T::from(window_len.get()).expect("can convert");
        let weight_total = n * (n + T::one()) / T::from(2.0).expect("can convert");
        assert!(weight_total >= T::one(), "the weights sum to at least 1");
        Self {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            sum: T::zero(),
            weighted_sum: T::zero(),
            weight_total,
            evictions: 0,
        }
    }
}

/// Exact `(sum, weighted_sum)` of the window, oldest value has weight 1.
fn exact_sums<T: Float>(q_vals: &VecDeque<T>) -> (T, T) {
    q_vals
        .iter()
        .zip(1..)
        .fold((T::zero(), T::zero()), |(s, w), (v, i)| {
            (s + *v, w + *v * T::from(i).expect("can convert"))
        })
}

impl<T, V> View<T> for Wma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() < self.window_len.get() {
            self.q_vals.push_back(val);
            let weight = T::from(self.q_vals.len()).expect("can convert");
            self.sum = self.sum + val;
            self.weighted_sum = self.weighted_sum + weight * val;
            return;
        }
        let oldest = self.q_vals.pop_front().expect("the window is full");
        self.q_vals.push_back(val);
        self.evictions += 1;
        if self.evictions >= self.window_len.get() {
            (self.sum, self.weighted_sum) = exact_sums(&self.q_vals);
            self.evictions = 0;
        } else {
            // Every remaining value loses one unit of weight, the oldest drops to 0.
            let n = T::from(self.window_len.get()).expect("can convert");
            self.weighted_sum = self.weighted_sum - self.sum + n * val;
            self.sum = self.sum - oldest + val;
        }
        debug_assert_eq!(self.q_vals.len(), self.window_len.get());
    }

    fn last(&self) -> Option<T> {
        if self.q_vals.len() < self.window_len.get() {
            return None;
        }
        let wma = self.weighted_sum / self.weight_total;
        debug_assert!(wma.is_finite(), "value must be finite");
        Some(wma)
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

    fn wma(window_len: usize) -> Wma<f64, Echo<f64>> {
        Wma::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    fn naive(window: &[f64]) -> f64 {
        let num: f64 = window.iter().zip(1..).map(|(v, i)| v * i as f64).sum();
        let n = window.len() as f64;
        num / (n * (n + 1.0) / 2.0)
    }

    #[test]
    fn wma_plot() {
        let mut w = wma(16);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            w.update(*v);
            if let Some(val) = w.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/wma.png").unwrap();
    }

    #[test]
    fn wma_hand_computed() {
        let mut w = wma(3);
        assert_eq!(w.window_len().get(), 3);
        w.update(1.0);
        assert_eq!(w.last(), None);
        w.update(2.0);
        assert_eq!(w.last(), None);
        w.update(3.0);
        assert_eq!(w.last(), Some((1.0 + 4.0 + 9.0) / 6.0));
        w.update(4.0);
        assert_eq!(w.last(), Some((2.0 + 6.0 + 12.0) / 6.0));
        w.update(-2.0);
        assert_eq!(w.last(), Some((3.0 + 8.0 - 6.0) / 6.0));
    }

    #[test]
    fn wma_window_one_is_identity() {
        let mut w = wma(1);
        for v in [3.0, -1.5, 7.25] {
            w.update(v);
            assert_eq!(w.last(), Some(v));
        }
    }

    #[test]
    fn wma_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        for window_len in [1, 2, 3, 10, 64] {
            let vals: Vec<f64> = (0..5_000).map(|_| rng.f64() * 200.0 - 100.0).collect();
            let mut w = wma(window_len);
            for (i, v) in vals.iter().enumerate() {
                w.update(*v);
                if i + 1 < window_len {
                    assert_eq!(w.last(), None);
                    continue;
                }
                let expected = naive(&vals[i + 1 - window_len..=i]);
                let got = w.last().unwrap();
                assert!(
                    (got - expected).abs() < 1e-9,
                    "step {i}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn wma_no_drift_with_large_offset() {
        // Large values with small fluctuations stress the running sums.
        let rng = romu::Rng::from_seed_with_64bit(12);
        let window_len = 50;
        let vals: Vec<f64> = (0..200_000).map(|_| 1e6 + rng.f64()).collect();
        let mut w = wma(window_len);
        for v in &vals {
            w.update(*v);
        }
        let expected = naive(&vals[vals.len() - window_len..]);
        assert!((w.last().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn wma_constant_input() {
        let mut w = wma(7);
        for _ in 0..100 {
            w.update(2.5);
            if let Some(v) = w.last() {
                assert!((v - 2.5).abs() < 1e-12);
            }
        }
        assert!(w.last().is_some());
    }

    #[test]
    fn wma_output_within_window_range() {
        let rng = romu::Rng::from_seed_with_64bit(13);
        let vals: Vec<f64> = (0..2_000).map(|_| rng.f64()).collect();
        let mut w = wma(20);
        for (i, v) in vals.iter().enumerate() {
            w.update(*v);
            let Some(out) = w.last() else { continue };
            let window = &vals[i + 1 - 20..=i];
            let lo = window.iter().copied().fold(f64::MAX, f64::min);
            let hi = window.iter().copied().fold(f64::MIN, f64::max);
            assert!(out >= lo - 1e-12);
            assert!(out <= hi + 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "value must be finite")]
    fn wma_non_finite_input() {
        let mut w = wma(3);
        w.update(f64::NAN);
    }
}

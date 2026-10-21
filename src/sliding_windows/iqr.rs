// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Rolling interquartile range over a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Rolling interquartile range (`IQR = Q3 - Q1`) of the last `window_len`
/// values, a robust measure of dispersion unaffected by the tails of the
/// window's distribution.
///
/// `Q1` and `Q3` are the 0.25 and 0.75 quantiles with linear interpolation
/// between order statistics (Hyndman & Fan 1996 type 7, the same convention
/// as [`super::RollingQuantile`] and the default of R's and numpy's
/// `quantile`).
///
/// `last()` is `None` until `window_len` (>= 2, so `Q1` and `Q3` are
/// distinct quantile computations) values were observed. Output range:
/// `[0, max - min]` of the window.
///
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a
/// sorted `Vec` (fast for typical window sizes); O(1) for `last()`; O(n) memory.
#[derive(Debug, Clone)]
pub struct Iqr<T, V> {
    view: V,
    window: SortedWindow<T>,
}

impl<T, V> Iqr<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling interquartile range with a chained `View` and a
    /// sliding window length.
    ///
    /// # Panics
    /// If `window_len` is below 2.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 2, "window_len must be >= 2");
        Self {
            view,
            window: SortedWindow::new(window_len),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }
}

impl<T, V> View<T> for Iqr<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let _evicted = self.window.push(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.window.is_full() {
            return None;
        }
        let q1 = self
            .window
            .quantile(T::from(0.25).expect("can convert"))
            .expect("a full window is not empty");
        let q3 = self
            .window
            .quantile(T::from(0.75).expect("can convert"))
            .expect("a full window is not empty");
        let out = q3 - q1;
        debug_assert!(out >= T::zero(), "interquartile range is non-negative");
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

    fn iqr(window_len: usize) -> Iqr<f64, Echo<f64>> {
        Iqr::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Naive type 7 quantile: sort a copy of the window.
    fn naive_quantile(window: &[f64], q: f64) -> f64 {
        let mut s = window.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = (s.len() - 1) as f64 * q;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(s.len() - 1);
        s[lo] + (h - lo as f64) * (s[hi] - s[lo])
    }

    fn naive_iqr(window: &[f64]) -> f64 {
        naive_quantile(window, 0.75) - naive_quantile(window, 0.25)
    }

    #[test]
    fn iqr_plot() {
        let mut v = iqr(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/iqr.png").unwrap();
    }

    #[test]
    fn iqr_warmup_and_hand_computed() {
        let mut v = iqr(4);
        assert_eq!(v.window_len().get(), 4);
        for x in [3.0, 1.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(2.0);
        // numpy.quantile([3, 1, 4, 2], 0.75) - numpy.quantile([3, 1, 4, 2], 0.25) == 3.25 - 1.75
        assert_eq!(v.last(), Some(1.5));
    }

    #[test]
    fn iqr_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(51);
        for window_len in [2, 3, 10, 51] {
            // Mix of continuous values and frequent duplicates.
            let vals: Vec<f64> = (0..2_000)
                .map(|_| match rng.range_usize(0..3) {
                    0 => rng.f64() * 10.0,
                    _ => rng.range_usize(0..5) as f64,
                })
                .collect();
            let mut v = iqr(window_len);
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let expected = naive_iqr(&vals[i + 1 - window_len..=i]);
                assert_eq!(v.last(), Some(expected), "step {i}, n {window_len}");
            }
        }
    }

    #[test]
    fn iqr_constant_input_is_zero() {
        let mut v = iqr(5);
        for _ in 0..20 {
            v.update(7.5);
        }
        assert_eq!(v.last(), Some(0.0));
    }

    #[test]
    fn iqr_non_negative() {
        let rng = romu::Rng::from_seed_with_64bit(52);
        let mut v = iqr(8);
        for _ in 0..500 {
            v.update(rng.f64() * 100.0 - 50.0);
            if let Some(val) = v.last() {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be >= 2")]
    fn iqr_window_len_too_small() {
        let _ = iqr(1);
    }
}

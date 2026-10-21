// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Rolling median over a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Rolling median of the last `window_len` values.
///
/// For an odd window length this is the middle order statistic; for an even
/// window length it is the average of the two middle order statistics
/// (linear interpolation at `q = 0.5`, Hyndman & Fan 1996 type 7, the same
/// convention as [`super::RollingQuantile`] at `q = 0.5`). Unlike the mean,
/// the median is robust to outliers within the window.
///
/// `last()` is `None` until `window_len` values were observed.
/// Output range: within the minimum and maximum of the window.
///
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a
/// sorted `Vec` (fast for typical window sizes); O(1) for `last()`; O(n) memory.
#[derive(Debug, Clone)]
pub struct RollingMedian<T, V> {
    view: V,
    window: SortedWindow<T>,
}

impl<T, V> RollingMedian<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling median with a chained `View` and a sliding
    /// window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
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

impl<T, V> View<T> for RollingMedian<T, V>
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
        let out = self.window.median().expect("a full window is not empty");
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

    fn rm(window_len: usize) -> RollingMedian<f64, Echo<f64>> {
        RollingMedian::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Naive median: sort a copy of the window, interpolate the middle
    /// elements with the same type 7 formula as `SortedWindow::quantile` (at
    /// `q = 0.5`), so floating-point rounding matches exactly rather than
    /// only approximately.
    fn naive_median(window: &[f64]) -> f64 {
        let mut s = window.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        let h = (n - 1) as f64 * 0.5;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = h - lo as f64;
        s[lo] + frac * (s[hi] - s[lo])
    }

    #[test]
    fn rolling_median_plot() {
        let mut v = rm(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/rolling_median.png").unwrap();
    }

    #[test]
    fn rolling_median_warmup_and_hand_computed() {
        let mut v = rm(4);
        assert_eq!(v.window_len().get(), 4);
        for x in [3.0, 1.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(2.0);
        // median of [3, 1, 4, 2] sorted [1, 2, 3, 4] -> (2 + 3) / 2
        assert_eq!(v.last(), Some(2.5));
        v.update(10.0);
        // median of [1, 4, 2, 10] sorted [1, 2, 4, 10] -> (2 + 4) / 2
        assert_eq!(v.last(), Some(3.0));
    }

    #[test]
    fn rolling_median_odd_window_is_middle_element() {
        let mut v = rm(3);
        for x in [5.0, 1.0, 3.0] {
            v.update(x);
        }
        // sorted [1, 3, 5] -> middle is 3
        assert_eq!(v.last(), Some(3.0));
    }

    #[test]
    fn rolling_median_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(41);
        for window_len in [1, 2, 3, 10, 51] {
            // Mix of continuous values and frequent duplicates.
            let vals: Vec<f64> = (0..2_000)
                .map(|_| match rng.range_usize(0..3) {
                    0 => rng.f64() * 10.0,
                    _ => rng.range_usize(0..5) as f64,
                })
                .collect();
            let mut v = rm(window_len);
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let expected = naive_median(&vals[i + 1 - window_len..=i]);
                assert_eq!(v.last(), Some(expected), "step {i}, n {window_len}");
            }
        }
    }

    #[test]
    fn rolling_median_constant_input() {
        let mut v = rm(5);
        for _ in 0..20 {
            v.update(7.5);
        }
        assert_eq!(v.last(), Some(7.5));
    }
}

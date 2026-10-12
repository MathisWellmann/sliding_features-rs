//! Rolling percentile rank of the latest value within a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Rolling percentile rank of the most recent value within the last
/// `window_len` values (including itself).
///
/// Uses the "mean" convention of `scipy.stats.percentileofscore`: the
/// average of the "strict" percentage (share of window values strictly
/// below the latest value) and the "weak" percentage (share of window
/// values less than or equal to it). Averaging the two handles ties (and in
/// particular the latest value always being present in its own window)
/// without picking either direction, so a window of indistinguishable
/// values always ranks at the 50th percentile.
///
/// `last()` is `None` until `window_len` values were observed.
/// Output range: `[0, 100]`.
///
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a
/// sorted `Vec` (fast for typical window sizes); O(log n) for `last()`
/// (two binary searches); O(n) memory.
#[derive(Debug, Clone)]
pub struct PercentileRank<T, V> {
    view: V,
    window: SortedWindow<T>,
    latest: T,
}

impl<T, V> PercentileRank<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling percentile rank with a chained `View` and a
    /// sliding window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window: SortedWindow::new(window_len),
            latest: T::zero(),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }
}

impl<T, V> View<T> for PercentileRank<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.latest = val;
        let _evicted = self.window.push(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.window.is_full() {
            return None;
        }
        let n = T::from(self.window.window_len().get()).expect("can convert");
        let less = T::from(self.window.count_less(self.latest)).expect("can convert");
        let less_equal = T::from(self.window.count_less_equal(self.latest)).expect("can convert");
        let hundred = T::from(100.0).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let out = (less + less_equal) / (two * n) * hundred;
        debug_assert!(out >= T::zero(), "percentile rank is in [0, 100]");
        debug_assert!(out <= hundred, "percentile rank is in [0, 100]");
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

    fn pr(window_len: usize) -> PercentileRank<f64, Echo<f64>> {
        PercentileRank::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    fn naive_percentile_rank(window: &[f64], x: f64) -> f64 {
        let n = window.len() as f64;
        let less = window.iter().filter(|&&v| v < x).count() as f64;
        let less_equal = window.iter().filter(|&&v| v <= x).count() as f64;
        (less + less_equal) / (2.0 * n) * 100.0
    }

    #[test]
    fn percentile_rank_plot() {
        let mut v = pr(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/percentile_rank.png").unwrap();
    }

    #[test]
    fn percentile_rank_warmup_and_hand_computed() {
        let mut v = pr(4);
        assert_eq!(v.window_len().get(), 4);
        for x in [3.0, 1.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(2.0);
        // window [3, 1, 4, 2], latest = 2: less = 1 (the 1), less_equal = 2 (1 and 2)
        // (1 + 2) / (2 * 4) * 100 = 37.5
        assert_eq!(v.last(), Some(37.5));
    }

    #[test]
    fn percentile_rank_new_minimum_is_near_zero() {
        let mut v = pr(4);
        for x in [3.0, 4.0, 2.0] {
            v.update(x);
        }
        v.update(0.0);
        // 0.0 is below every other value in the window [0, 2, 3, 4]: strictly
        // below 0% of the window, at-or-below 25% of it (itself), so the
        // "mean" rank is (0 + 25) / 2 = 12.5, not exactly 0.
        assert_eq!(v.last(), Some(12.5));
    }

    #[test]
    fn percentile_rank_new_maximum_is_near_hundred() {
        let mut v = pr(4);
        for x in [3.0, 4.0, 2.0] {
            v.update(x);
        }
        v.update(10.0);
        // 10.0 is above every other value in the window [2, 3, 4, 10]:
        // strictly below 75% of it, at-or-below 100% of it (itself), so the
        // "mean" rank is (75 + 100) / 2 = 87.5, not exactly 100.
        assert_eq!(v.last(), Some(87.5));
    }

    #[test]
    fn percentile_rank_constant_input_is_fifty() {
        let mut v = pr(5);
        for _ in 0..20 {
            v.update(7.5);
        }
        assert_eq!(v.last(), Some(50.0));
    }

    #[test]
    fn percentile_rank_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(61);
        for window_len in [1, 2, 3, 10, 51] {
            // Mix of continuous values and frequent duplicates.
            let vals: Vec<f64> = (0..2_000)
                .map(|_| match rng.range_usize(0..3) {
                    0 => rng.f64() * 10.0,
                    _ => rng.range_usize(0..5) as f64,
                })
                .collect();
            let mut v = pr(window_len);
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let window = &vals[i + 1 - window_len..=i];
                let expected = naive_percentile_rank(window, *x);
                assert_eq!(v.last(), Some(expected), "step {i}, n {window_len}");
            }
        }
    }
}

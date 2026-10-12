//! A sliding window that also keeps its values in sorted order.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use num::Float;

/// A sliding window of the last `window_len` values, kept both in insertion
/// order (FIFO) and in sorted order, for order statistics such as quantiles,
/// the median or percentile ranks.
///
/// Per `push`: O(log n) binary search plus an O(n) `memmove` for the insert
/// and the removal in the sorted `Vec`, which is fast for typical window sizes
/// (a few thousand values). Queries are O(1) (`quantile`, `median`) or
/// O(log n) (`count_less`, `count_less_equal`). Memory is O(n).
///
/// Methods that are not used by library code yet are `#[cfg(test)]` to avoid
/// `dead_code` warnings; remove the attribute when using one.
#[derive(Debug, Clone)]
pub(crate) struct SortedWindow<T> {
    window_len: NonZeroUsize,
    /// Values in insertion order, oldest at the front.
    fifo: VecDeque<T>,
    /// The same values in ascending order.
    sorted: Vec<T>,
}

impl<T: Float> SortedWindow<T> {
    /// Create an empty window holding at most `window_len` values.
    pub(crate) fn new(window_len: NonZeroUsize) -> Self {
        Self {
            window_len,
            fifo: VecDeque::with_capacity(window_len.get()),
            sorted: Vec::with_capacity(window_len.get()),
        }
    }

    /// Append `val` and return the evicted oldest value, if the window was full.
    pub(crate) fn push(&mut self, val: T) -> Option<T> {
        debug_assert!(val.is_finite(), "value must be finite");
        let evicted = if self.fifo.len() == self.window_len.get() {
            let oldest = self.fifo.pop_front().expect("the window is full");
            let pos = self.sorted.partition_point(|p| *p < oldest);
            assert!(
                self.sorted.get(pos) == Some(&oldest),
                "the evicted value must be in the sorted window"
            );
            self.sorted.remove(pos);
            Some(oldest)
        } else {
            None
        };
        self.fifo.push_back(val);
        let pos = self.sorted.partition_point(|p| *p < val);
        self.sorted.insert(pos, val);

        debug_assert_eq!(self.fifo.len(), self.sorted.len());
        debug_assert!(self.fifo.len() <= self.window_len.get());
        evicted
    }

    /// The maximum number of values in the window.
    pub(crate) fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// The number of values currently in the window.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.fifo.len()
    }

    /// Whether the window holds `window_len` values.
    pub(crate) fn is_full(&self) -> bool {
        self.fifo.len() == self.window_len.get()
    }

    /// The `q`-quantile of the window with linear interpolation between order
    /// statistics (Hyndman & Fan type 7, the default of R and numpy):
    /// `h = (n - 1) q`, `Q = s[floor(h)] + (h - floor(h)) (s[floor(h) + 1] - s[floor(h)])`.
    /// `None` if the window is empty.
    ///
    /// # Panics
    /// If `q` is not in `[0, 1]`.
    pub(crate) fn quantile(&self, q: T) -> Option<T> {
        assert!(q >= T::zero(), "q must be >= 0");
        assert!(q <= T::one(), "q must be <= 1");
        let last_idx = self.sorted.len().checked_sub(1)?;
        let h = T::from(last_idx).expect("can convert") * q;
        let lo = h
            .floor()
            .to_usize()
            .expect("h is in [0, len - 1], so it fits into usize");
        debug_assert!(lo <= last_idx, "floor(h) is a valid index");
        let hi = (lo + 1).min(last_idx);
        let frac = h - T::from(lo).expect("can convert");
        let (s_lo, s_hi) = (self.sorted[lo], self.sorted[hi]);
        let out = s_lo + frac * (s_hi - s_lo);
        debug_assert!(
            out >= s_lo,
            "interpolation stays within its order statistics"
        );
        debug_assert!(
            out <= s_hi,
            "interpolation stays within its order statistics"
        );
        Some(out)
    }

    /// The median of the window, `None` if it is empty.
    #[cfg(test)]
    pub(crate) fn median(&self) -> Option<T> {
        self.quantile(T::from(0.5).expect("can convert"))
    }

    /// The number of values in the window strictly below `x`.
    pub(crate) fn count_less(&self, x: T) -> usize {
        debug_assert!(!x.is_nan(), "x must not be NaN");
        self.sorted.partition_point(|p| *p < x)
    }

    /// The number of values in the window below or equal to `x`.
    #[cfg(test)]
    pub(crate) fn count_less_equal(&self, x: T) -> usize {
        debug_assert!(!x.is_nan(), "x must not be NaN");
        self.sorted.partition_point(|p| *p <= x)
    }

    /// The values of the window in ascending order.
    #[cfg(test)]
    pub(crate) fn sorted(&self) -> &[T] {
        &self.sorted
    }

    /// The values of the window in insertion order, oldest first.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = T> + '_ {
        self.fifo.iter().copied()
    }

    /// The oldest value of the window, which is evicted next once the window is full.
    #[cfg(test)]
    pub(crate) fn oldest(&self) -> Option<T> {
        self.fifo.front().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive type 7 quantile of an already sorted slice.
    fn naive_quantile(sorted: &[f64], q: f64) -> f64 {
        let h = (sorted.len() - 1) as f64 * q;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
    }

    fn window(window_len: usize) -> SortedWindow<f64> {
        SortedWindow::new(NonZeroUsize::new(window_len).unwrap())
    }

    #[test]
    fn sorted_window_empty() {
        let w = window(3);
        assert_eq!(w.window_len().get(), 3);
        assert_eq!(w.len(), 0);
        assert!(!w.is_full());
        assert_eq!(w.quantile(0.5), None);
        assert_eq!(w.median(), None);
        assert_eq!(w.oldest(), None);
        assert_eq!(w.count_less(1.0), 0);
        assert!(w.sorted().is_empty());
    }

    #[test]
    fn sorted_window_hand_computed() {
        let mut w = window(4);
        assert_eq!(w.push(3.0), None);
        assert_eq!(w.push(1.0), None);
        assert_eq!(w.push(4.0), None);
        assert!(!w.is_full());
        assert_eq!(w.push(2.0), None);
        assert!(w.is_full());
        assert_eq!(w.sorted(), &[1.0, 2.0, 3.0, 4.0]);
        // numpy.quantile([1, 2, 3, 4], [0, .25, .5, 1]) == [1, 1.75, 2.5, 4]
        assert_eq!(w.quantile(0.0), Some(1.0));
        assert_eq!(w.quantile(0.25), Some(1.75));
        assert_eq!(w.median(), Some(2.5));
        assert_eq!(w.quantile(1.0), Some(4.0));
        assert_eq!(w.push(2.0), Some(3.0));
        assert_eq!(w.sorted(), &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(w.iter().collect::<Vec<_>>(), vec![1.0, 4.0, 2.0, 2.0]);
        assert_eq!(w.oldest(), Some(1.0));
        assert_eq!(w.count_less(2.0), 1);
        assert_eq!(w.count_less_equal(2.0), 3);
        assert_eq!(w.count_less(0.0), 0);
        assert_eq!(w.count_less_equal(4.0), 4);
    }

    #[test]
    fn sorted_window_vs_naive_with_duplicates() {
        let rng = romu::Rng::from_seed_with_64bit(21);
        for window_len in [1, 2, 5, 33] {
            let mut w = window(window_len);
            let mut naive: VecDeque<f64> = VecDeque::new();
            for _ in 0..3_000 {
                // Few distinct values, so duplicates are common.
                let val = rng.range_usize(0..8) as f64 * 0.5;
                let expected_evicted = if naive.len() == window_len {
                    naive.pop_front()
                } else {
                    None
                };
                naive.push_back(val);
                assert_eq!(w.push(val), expected_evicted);

                let mut sorted: Vec<f64> = naive.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(w.sorted(), sorted.as_slice());
                assert!(w.iter().eq(naive.iter().copied()));
                assert_eq!(w.len(), naive.len());
                assert_eq!(w.oldest(), naive.front().copied());
                for q in [0.0, 0.25, 0.5, 0.9, 1.0] {
                    assert_eq!(w.quantile(q), Some(naive_quantile(&sorted, q)));
                }
                let x = rng.range_usize(0..9) as f64 * 0.5 - 0.25 * rng.range_usize(0..2) as f64;
                assert_eq!(w.count_less(x), sorted.iter().filter(|v| **v < x).count());
                assert_eq!(
                    w.count_less_equal(x),
                    sorted.iter().filter(|v| **v <= x).count()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "q must be >= 0")]
    fn sorted_window_quantile_below_zero() {
        let mut w = window(3);
        w.push(1.0);
        let _ = w.quantile(-1e-9);
    }

    #[test]
    #[should_panic(expected = "q must be <= 1")]
    fn sorted_window_quantile_above_one() {
        let mut w = window(3);
        w.push(1.0);
        let _ = w.quantile(1.0 + 1e-9);
    }
}

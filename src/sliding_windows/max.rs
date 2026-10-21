// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Keep track of the maximum value observed over the sliding window
/// of the last `window_len` values.
///
/// `last()` is `Some` as soon as the first value was observed; until the window
/// is full, it is the maximum of all values observed so far.
///
/// Uses a monotonic deque (D. Lemire, "Streaming Maximum-Minimum Filter Using
/// No More than Three Comparisons per Element", 2006, <https://arxiv.org/abs/cs/0610046>):
/// the deque holds `(index, value)` pairs with strictly decreasing values,
/// so the front is always the window maximum.
/// Each value is pushed and popped at most once, giving amortized O(1)
/// (worst case O(window_len)) per update and O(window_len) memory.
#[derive(Clone, Debug, CopyGetters)]
pub struct Max<T, V> {
    view: V,
    /// Candidates `(index, value)`, values strictly decreasing from front to back.
    deque: VecDeque<(u64, T)>,
    /// Index assigned to the next value.
    next_idx: u64,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
}

impl<T, V> Max<T, V>
where
    T: Float,
    V: View<T>,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            deque: VecDeque::with_capacity(window_len.get()),
            next_idx: 0,
            window_len,
        }
    }
}

impl<T, V> View<T> for Max<T, V>
where
    T: Float,
    V: View<T>,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let idx = self.next_idx;
        let window_len = u64::try_from(self.window_len.get()).expect("usize fits into u64");
        // An older value that is not larger can never be the maximum again,
        // because `val` outlives it in the window. Dropping equal values too keeps
        // the deque small for repeated values without changing the result.
        while self.deque.back().is_some_and(|&(_, v)| v <= val) {
            self.deque.pop_back();
        }
        self.deque.push_back((idx, val));
        // The front is the oldest candidate, so at most one value expires per step.
        if self
            .deque
            .front()
            .is_some_and(|&(i, _)| i + window_len <= idx)
        {
            self.deque.pop_front();
        }
        self.next_idx += 1;

        debug_assert!(
            !self.deque.is_empty(),
            "the newest value is always a candidate"
        );
        debug_assert!(
            self.deque.len() <= self.window_len.get(),
            "at most one candidate per window slot"
        );
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.deque.front().map(|&(_, v)| {
            debug_assert!(v.is_finite(), "value must be finite");
            v
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::pure_functions::Echo;

    #[test]
    fn max() {
        const WINDOW_LEN: usize = 3;
        let mut v = Max::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        assert_eq!(v.last(), None);
        v.update(1.0);
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        assert_eq!(v.last(), Some(1.0));
        v.update(2.0);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        v.update(0.5);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        v.update(1.1);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        v.update(1.2);
        assert_eq!(v.last(), Some(1.2));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        v.update(1.3);
        assert_eq!(v.last(), Some(1.3));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        v.update(1.4);
        assert_eq!(v.last(), Some(1.4));
        assert_eq!(v.window_len(), NonZeroUsize::new(WINDOW_LEN).unwrap());
    }

    /// Feed `vals` and compare against a naive rescan of the trailing window.
    fn check_against_naive(vals: &[f64], window_len: usize) {
        let mut v = Max::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        assert_eq!(v.last(), None);
        for (i, val) in vals.iter().enumerate() {
            v.update(*val);
            let start = (i + 1).saturating_sub(window_len);
            let naive = vals[start..=i].iter().copied().fold(f64::MIN, f64::max);
            assert_eq!(v.last(), Some(naive), "step {i}, window_len {window_len}");
        }
    }

    #[test]
    fn max_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(42);
        for window_len in [1, 2, 3, 7, 32, 100] {
            let vals: Vec<f64> = (0..2_000).map(|_| rng.f64() * 200.0 - 100.0).collect();
            check_against_naive(&vals, window_len);
        }
    }

    #[test]
    fn max_vs_naive_duplicates() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        for window_len in [1, 2, 5, 16] {
            // Only 4 distinct values, so ties happen constantly.
            let vals: Vec<f64> = (0..2_000).map(|_| rng.range_usize(0..4) as f64).collect();
            check_against_naive(&vals, window_len);
        }
    }

    #[test]
    fn max_vs_naive_monotone() {
        let increasing: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let decreasing: Vec<f64> = increasing.iter().rev().copied().collect();
        let constant = vec![3.5; 500];
        for window_len in [1, 2, 10, 499, 500, 1000] {
            check_against_naive(&increasing, window_len);
            check_against_naive(&decreasing, window_len);
            check_against_naive(&constant, window_len);
        }
    }

    #[test]
    fn max_deque_bounded() {
        // A strictly decreasing sequence keeps every value as a candidate,
        // which is the worst case for the deque size.
        let window_len = 8;
        let mut v = Max::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        for i in 0..100 {
            v.update(-(i as f64));
            assert!(v.deque.len() <= window_len);
        }
        assert_eq!(v.deque.len(), window_len);
        assert_eq!(v.last(), Some(-92.0));
    }
}

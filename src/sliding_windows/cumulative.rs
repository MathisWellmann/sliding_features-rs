//! Cumulative sliding window

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Cumulative Sliding Window with a chained view
#[derive(Debug, Clone, CopyGetters)]
pub struct Cumulative<T, V> {
    view: V,
    /// The length of the sliding window.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Cumulative<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new cumulative sliding window with a chained view and a window length
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

impl<T, V> View<T> for Cumulative<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let out = self.out.get_or_insert(T::zero());
        if self.q_vals.len() >= self.window_len.get() {
            let old = self.q_vals.pop_front().expect("the window is full");
            *out = *out - old;
        }
        self.q_vals.push_back(val);
        *out = *out + val;
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(self.q_vals.len() <= self.window_len.get());
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
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
    fn cumulative_plot() {
        let mut cum = Cumulative::new(Echo::new(), NonZeroUsize::new(TEST_DATA.len()).unwrap());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            cum.update(*v);
            out.push(cum.last().unwrap());
        }
        let filename = "img/cumulative.png";
        plot_values(out, filename).unwrap();
    }

    /// The output is the sum of the last `window_len` values; the first
    /// value used to be counted twice, forever.
    #[test]
    fn cumulative_sums_the_window() {
        let mut cum = Cumulative::new(Echo::new(), NonZeroUsize::new(2).unwrap());
        assert_eq!(cum.last(), None);
        let mut sums = Vec::new();
        for v in [1.0, 2.0, 4.0, 8.0] {
            cum.update(v);
            sums.push(cum.last().unwrap());
        }
        assert_eq!(sums, [1.0, 3.0, 6.0, 12.0]);
    }
}

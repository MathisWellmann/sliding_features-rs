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

        if self.out.is_none() {
            self.out = Some(val);
        }

        if self.q_vals.len() >= self.window_len.get() {
            let old = self.q_vals.pop_front().unwrap();
            let out = self.out.as_mut().expect("Is some at this point");
            *out = *out - old;
        }
        self.q_vals.push_back(val);
        let out = self.out.as_mut().expect("Is some at this point");
        *out = *out + val;
        debug_assert!(out.is_finite(), "value must be finite");
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
}

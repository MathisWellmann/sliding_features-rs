//! Cumulative sliding window

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// Cumulative Sliding Window with a chained view
#[derive(Debug, Clone, CopyGetters)]
pub struct Cumulative<T, V> {
    view: V,
    /// The length of the sliding window.
    #[getset(get_copy = "pub")]
    window_len: usize,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Cumulative<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new cumulative sliding window with a chained view and a window length
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len),
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
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.out.is_none() {
            self.out = Some(val);
        }

        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.pop_front().unwrap();
            let out = self.out.as_mut().expect("Is some at this point");
            *out = *out - old;
        }
        self.q_vals.push_back(val);
        let out = self.out.as_mut().expect("Is some at this point");
        *out = *out + val;
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use crate::{plot::plot_values, pure_functions::Echo, test_data::TEST_DATA};

    use super::*;

    #[test]
    fn cumulative_plot() {
        let mut cum = Cumulative::new(Echo::new(), TEST_DATA.len());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            cum.update(*v);
            out.push(cum.last().unwrap());
        }
        let filename = "img/cumulative.png";
        plot_values(out, filename).unwrap();
    }
}

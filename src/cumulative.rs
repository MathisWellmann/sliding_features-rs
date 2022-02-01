//! Cumulative sliding window

use std::collections::VecDeque;

use crate::View;

/// Cumulative Sliding Window with a chained view
#[derive(Clone)]
pub struct Cumulative<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V> Cumulative<V>
where
    V: View,
{
    /// Create a new cumulative sliding window with a chained view and a window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl<V> View for Cumulative<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.pop_front().unwrap();
            self.out -= old;
        }
        self.q_vals.push_back(val);
        self.out += val;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use crate::{plot::plot_values, test_data::TEST_DATA, Echo};

    use super::*;

    #[test]
    fn cumulative_plot() {
        let mut cum = Cumulative::new(Echo::new(), TEST_DATA.len());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            cum.update(*v);
            out.push(cum.last());
        }
        let filename = "img/cumulative.png";
        plot_values(out, filename).unwrap();
    }
}

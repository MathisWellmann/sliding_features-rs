//! Cumulative sliding window

use std::collections::VecDeque;

use crate::View;

/// Cumulative Sliding Window with a chained view
#[derive(Debug, Clone)]
pub struct Cumulative<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: Option<f64>,
}

impl<V> Cumulative<V>
where
    V: View,
{
    /// Create a new cumulative sliding window with a chained view and a window length
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            out: None,
        }
    }
}

impl<V> View for Cumulative<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.out.is_none() {
            self.out = Some(val);
        }

        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.pop_front().unwrap();
            *self.out.as_mut().expect("Is some at this point") -= old;
        }
        self.q_vals.push_back(val);
        *self.out.as_mut().expect("Is some as this point") += val;
    }

    fn last(&self) -> Option<f64> {
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

//! Rate of Change Indicator

use std::collections::VecDeque;

use crate::View;

/// Rate of Change Indicator
#[derive(Debug, Clone)]
pub struct ROC<V> {
    view: V,
    window_len: usize,
    oldest: f64,
    q_vals: VecDeque<f64>,
    out: Option<f64>,
}

impl<V> ROC<V>
where
    V: View,
{
    /// Create a new Rate of Change Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        ROC {
            view,
            window_len,
            oldest: 0.0,
            q_vals: VecDeque::new(),
            out: None,
        }
    }
}

impl<V> View for ROC<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.is_empty() {
            self.oldest = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.front().unwrap();
            self.oldest = *old;
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let roc = ((val - self.oldest) / self.oldest) * 100.0;
        self.out = Some(roc);
    }

    fn last(&self) -> Option<f64> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn roc_plot() {
        let mut r = ROC::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            r.update(*v);
            if let Some(val) = r.last() {
                out.push(val);
            }
        }
        let filename = "img/roc.png";
        plot_values(out, filename).unwrap();
    }
}

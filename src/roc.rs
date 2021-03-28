use std::collections::VecDeque;

use super::sliding_window::View;
use crate::Echo;

/// Rate of Change Indicator
#[derive(Clone)]
pub struct ROC {
    view: Box<dyn View>,
    window_len: usize,
    oldest: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl ROC {
    /// Create a new Rate of Change Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Box<Self> {
        Box::new(ROC {
            view,
            window_len,
            oldest: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        })
    }

    /// Create a new Rate of Change Indicator with a given window length
    pub fn new_final(window_len: usize) -> Box<Self> {
        Self::new(Echo::new(), window_len)
    }
}

impl View for ROC {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() == 0 {
            self.oldest = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.front().unwrap();
            self.oldest = *old;
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let roc = ((val - self.oldest) / self.oldest) * 100.0;
        self.out = roc;
    }

    fn last(&self) -> f64 {
        return self.out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn roc_plot() {
        let mut r = ROC::new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            r.update(*v);
            out.push(r.last());
        }
        let filename = "img/roc.png";
        plot_values(out, filename).unwrap();
    }
}

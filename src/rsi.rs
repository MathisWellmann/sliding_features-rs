use std::collections::VecDeque;

use super::sliding_window::View;
use crate::Echo;

/// Relative Strength Index Indicator
#[derive(Clone)]
pub struct RSI {
    view: Box<dyn View>,
    window_len: usize,
    avg_gain: f64,
    avg_loss: f64,
    old_ref: f64,
    last_val: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl RSI {
    /// Create a Relative Strength Index Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        RSI {
            view,
            window_len,
            avg_gain: 0.0,
            avg_loss: 0.0,
            old_ref: 0.0,
            last_val: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }

    /// Create a new Relative Strength Index Indicator with a given window length
    pub fn new_final(window_len: usize) -> Self {
        Self::new(Box::new(Echo::new()), window_len)
    }
}

impl View for RSI {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() == 0 {
            self.old_ref = val;
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len {
            // remove old
            let old_val = self.q_vals.front().unwrap();
            let change = old_val - self.old_ref;
            self.old_ref = *old_val;
            self.q_vals.pop_front();
            if change > 0.0 {
                self.avg_gain -= change / self.window_len as f64;
            } else {
                self.avg_loss -= change.abs() / self.window_len as f64;
            }
        }
        self.q_vals.push_back(val);

        let change = val - self.last_val;
        self.last_val = val;
        if change > 0.0 {
            self.avg_gain += change / self.window_len as f64;
        } else {
            self.avg_loss += change.abs() / self.window_len as f64;
        }

        if self.q_vals.len() < self.window_len {
            self.out = 50.0;
        } else {
            if self.avg_loss == 0.0 {
                self.out = 100.0;
            } else {
                let rs = self.avg_gain / self.avg_loss;
                let rsi = 100.0 - 100.0 / (1.0 + rs);
                self.out = rsi;
            }
        }
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
    fn rsi_plot() {
        let mut rsi = RSI::new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rsi.update(*v);
            out.push(rsi.last());
        }
        let filename = "img/rsi.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn rsi_range() {
        let mut rsi = RSI::new_final(16);
        for v in &TEST_DATA {
            rsi.update(*v);
            let last = rsi.last();
            assert!(last >= 0.0);
            assert!(last <= 100.0);
        }
    }
}

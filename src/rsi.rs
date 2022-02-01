//! Relative Strength Index Indicator

use std::collections::VecDeque;

use super::View;

/// Relative Strength Index Indicator
#[derive(Clone)]
pub struct RSI<V> {
    view: V,
    window_len: usize,
    avg_gain: f64,
    avg_loss: f64,
    old_ref: f64,
    last_val: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V> std::fmt::Debug for RSI<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "RSI(window_len: {}, avg_gain: {}, avg_loss: {}, old_ref: {}, last_val: {}, q_vals: {:?}, out: {})",
               self.window_len, self.avg_gain, self.avg_loss, self.old_ref, self.last_val, self.q_vals, self.out)
    }
}

impl<V> RSI<V>
where
    V: View,
{
    /// Create a Relative Strength Index Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
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
}

impl<V> View for RSI<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.is_empty() {
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

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::Echo;

    #[test]
    fn rsi_plot() {
        let mut rsi = RSI::new(Echo::new(), 16);
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
        let mut rsi = RSI::new(Echo::new(), 16);
        for v in &TEST_DATA {
            rsi.update(*v);
            let last = rsi.last();
            assert!(last >= 0.0);
            assert!(last <= 100.0);
        }
    }
}

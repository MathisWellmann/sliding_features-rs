//! John Ehlers MyRSI
//! from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>

use crate::View;
use std::collections::VecDeque;

/// John Ehlers MyRSI
/// from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>
#[derive(Debug, Clone)]
pub struct MyRSI<V> {
    view: V,
    window_len: usize,
    cu: f64,
    cd: f64,
    out: f64,
    q_vals: VecDeque<f64>,
    last_val: f64,
    oldest_val: f64,
}

impl<V> MyRSI<V>
where
    V: View,
{
    /// Create a new MyRSI indicator with a chained View and a given window length
    pub fn new(view: V, window_len: usize) -> Self {
        MyRSI {
            view,
            window_len,
            cu: 0.0,
            cd: 0.0,
            out: 0.0,
            q_vals: VecDeque::new(),
            last_val: 0.0,
            oldest_val: 0.0,
        }
    }
}

impl<V> View for MyRSI<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.is_empty() {
            self.oldest_val = val;
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old_val: f64 = self.q_vals.pop_front().unwrap();
            if old_val > self.oldest_val {
                self.cu -= old_val - self.oldest_val;
            } else {
                self.cd -= self.oldest_val - old_val;
            }
            self.oldest_val = old_val;
        }
        self.q_vals.push_back(val);

        // accumulate 'closes up' and 'closes down'
        if val > self.last_val {
            self.cu += val - self.last_val;
        } else {
            self.cd += self.last_val - val;
        }
        self.last_val = val;

        if self.cu + self.cd != 0.0 {
            self.out = (self.cu - self.cd) / (self.cu + self.cd);
        }
    }

    fn last(&self) -> Option<f64> {
        if self.q_vals.len() < self.window_len {
            return None;
        }
        Some(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn my_rsi() {
        // TODO: don't be so lazy with this test.
        let mut my_rsi = MyRSI::new(Echo::new(), 16);
        for v in &TEST_DATA {
            my_rsi.update(*v);
            if let Some(val) = my_rsi.last() {
                assert!(val <= 1.0);
                assert!(val >= -1.0);
            }
        }
    }

    #[test]
    fn my_rsi_plot() {
        let mut my_rsi = MyRSI::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            my_rsi.update(*v);
            if let Some(rsi) = my_rsi.last() {
                out.push(rsi);
            }
        }
        println!("out: {:?}", out);
        let filename = "img/my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

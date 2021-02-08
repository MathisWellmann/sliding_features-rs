use std::collections::VecDeque;

use super::sliding_window::View;

#[derive(Debug, Clone)]
pub struct RSI {
    window_len: usize,
    avg_gain: f64,
    avg_loss: f64,
    old_ref: f64,
    last_val: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl RSI {
    pub fn new(window_len: usize) -> RSI {
        return RSI {
            window_len,
            avg_gain: 0.0,
            avg_loss: 0.0,
            old_ref: 0.0,
            last_val: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        };
    }
}

impl View for RSI {
    fn update(&mut self, val: f64) {
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
    extern crate rust_timeseries_generator;
    use self::rust_timeseries_generator::gaussian_process::gen;
    use self::rust_timeseries_generator::plt;
    use super::*;

    #[test]
    fn graph_rsi() {
        let vals = gen(1024, 100.0);
        let mut rsi = RSI::new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            rsi.update(vals[i]);
            out.push(rsi.last());
        }
        let filename = "img/rsi.png";
        plt::plt(out, filename).unwrap();
    }

    #[test]
    fn test_range() {
        let vals = gen(1024, 100.0);
        let mut rsi = RSI::new(16);
        for i in 0..vals.len() {
            rsi.update(vals[i]);
            let last = rsi.last();
            assert!(last >= 0.0);
            assert!(last <= 100.0);
        }
    }
}

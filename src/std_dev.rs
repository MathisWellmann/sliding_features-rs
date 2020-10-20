use crate::sliding_window::View;
use std::collections::VecDeque;


pub struct StdDev {
    window_len: usize,
    mean: f64,
    s: f64,
    q_vals: VecDeque<f64>,
}

impl StdDev {
    pub fn new(window_len: usize) -> StdDev {
        return StdDev {
            window_len,
            mean: 0.0,
            s: 0.0,
            q_vals: VecDeque::new(),
        }
    }
}

impl View for StdDev {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() >= self.window_len {
            // remove old value from std_dev estimation
            let old_val = self.q_vals.front().unwrap();
            let old_mean = self.mean;

            self.mean -= (old_val - self.mean) / self.q_vals.len() as f64;
            self.s -= (old_val - self.mean) * (old_val - old_mean);

            self.q_vals.pop_front();
            return
        }
        self.q_vals.push_back(val);

        let old_mean = self.mean;
        self.mean += (val - old_mean) / self.q_vals.len() as f64;
        self.s += (val - old_mean) * (val -self.mean);
    }

    fn last(&self) -> f64 {
        return self.variance().sqrt()
    }
}

impl StdDev {
    fn variance(&self) -> f64 {
        if self.q_vals.len() > 1 {
            return self.s / (self.q_vals.len() as f64 - 1.0);
        }
        return 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rust_timeseries_generator;
    use rust_timeseries_generator::{plt, gaussian_process};


    #[test]
    fn test_std_dev_graph() {
        let vals = gaussian_process::gen(10_000, 100.0);
        let mut std_dev = StdDev::new(64);
        let mut out: Vec<f64> = Vec::new();
        for v in &vals {
            std_dev.update(*v);
            out.push(std_dev.last());
        }
        let filename = "img/std_dev.png";
        plt::plt(out, filename).unwrap();
    }
}

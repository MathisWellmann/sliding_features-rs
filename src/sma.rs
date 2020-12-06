use std::collections::VecDeque;

use crate::sliding_window::View;

// SMA - Simple Moving Average
#[derive(Debug, Clone)]
pub struct SMA {
    window_len: usize,
    q_vals: VecDeque<f64>,
    sma: f64,
    last: f64,
}

impl SMA {
    pub fn new(window_len: usize) -> SMA {
        return SMA{
            window_len,
            q_vals: VecDeque::new(),
            sma: 0.0,
            last: 0.0,
        }
    }
}

impl View for SMA {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() > self.window_len {
            let old_val = self.q_vals.front().unwrap();
            self.last -= (old_val / self.window_len as f64);
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);
        self.sma += (val / self.window_len as f64);

        self.last = self.sma;
    }

    fn last(&self) -> f64 {
        return self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use rand::{thread_rng, Rng};
    extern crate rust_timeseries_generator;
    use rust_timeseries_generator::{plt, gaussian_process};

    #[test]
    fn test_sma() {
        let mut rng = thread_rng();
        let mut sma = SMA::new(16);
        for i in 0..1024 {
            let r = rng.gen::<f64>();
            sma.update(r);
            let last = sma.last();
            assert!(last >= 0.0);
            assert!(last <= 1.0);
        }
    }

    #[test]
    fn test_sma_graph() {
        let vals = gaussian_process::gen(1024, 100.0);
        let mut sma = SMA::new(128);
        let mut out: Vec<f64> = Vec::new();
        for v in &vals {
            sma.update(*v);
            out.push(sma.last());
        }
        let filename = "img/sma.png";
        plt::plt(out, filename);
    }
}
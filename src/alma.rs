use std::collections::VecDeque;

use crate::sliding_window::View;

// ALMA - Arnaud Legoux Moving Average
pub struct ALMA {
    window_len: usize,
    wtd_sum: f64,
    cum_wt: f64,
    m: f64,
    s: f64,
    q_vals: VecDeque<f64>,
    q_wtd: VecDeque<f64>,
    q_out: VecDeque<f64>,
}

impl ALMA {
    pub fn new(window_len: usize, sigma: f64, offset: f64) -> ALMA {
        let m = offset * (window_len as f64  + 1.0);
        let s = window_len as f64 / sigma;
        return ALMA {
            window_len,
            wtd_sum: 0.0,
            cum_wt: 0.0,
            m,
            s,
            q_vals: VecDeque::new(),
            q_wtd: VecDeque::new(),
            q_out: VecDeque::new(),
        }
    }

    pub fn default(window_len: usize) -> ALMA {
        return ALMA::new(window_len, 6.0, 0.85)
    }
}

impl View for ALMA {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() >= self.window_len {
            let old_val = self.q_vals.front().unwrap();
            let old_wtd = self.q_wtd.front().unwrap();
            self.wtd_sum -= old_wtd * old_val;
            self.cum_wt -= *old_wtd;

            self.q_vals.pop_front();
            self.q_wtd.pop_front();
            self.q_out.pop_front();
        }
        let count = self.q_vals.len() as f64;
        let wtd = (-(count - self.m).powi(2) / (2.0 * self.s * self.s) ).exp();
        self.wtd_sum += wtd * val;
        self.cum_wt += wtd;

        self.q_vals.push_back(val);
        self.q_wtd.push_back(wtd);

        if self.q_vals.len() < self.window_len {
            self.q_out.push_back(val);
        } else {
            let ala = self.wtd_sum / self.cum_wt;
            self.q_out.push_back(ala);
        }
    }

    fn last(&self) -> f64 {
        return *self.q_out.back().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use rand::{Rng, thread_rng};
    extern crate rust_timeseries_generator;
    use rust_timeseries_generator::{plt, gaussian_process};

    #[test]
    fn test_alma() {
        let mut rng = thread_rng();
        let mut alma = ALMA::default(16);
        for _i in 0..1_000_000 {
            let r = rng.gen::<f64>();
            alma.update(r);
            let last = alma.last();

            assert!(last >= 0.0);
            assert!(last <= 1.0);
        }
    }

    #[test]
    fn test_alma_graph() {
        let vals = gaussian_process::gen(1024, 100.0);
        let mut alma = ALMA::default(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &vals {
            alma.update(*v);
            out.push(alma.last())
        }
        let filename = "img/alma.png";
        plt::plt(out, filename).unwrap();
    }
}
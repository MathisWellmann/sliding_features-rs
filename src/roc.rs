use std::collections::VecDeque;

use super::sliding_window::View;

pub struct ROC {
    window_len: usize,
    oldest: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl ROC {
    pub fn new(window_len: usize) -> ROC {
        return ROC{
            window_len,
            oldest: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl View for ROC {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() == 0 {
            self.oldest = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.front().unwrap();
            self.oldest = *old;
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let roc = ((val - self.oldest)  / self.oldest) * 100.0;
        self.out = roc;
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
    fn graph_roc() {
        let vals = gen(1024, 100.0);
        let mut r = ROC::new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            r.update(vals[i]);
            out.push(r.last());
        }
        let filename = "img/roc.png";
        plt::plt(out, filename).unwrap();
    }
}

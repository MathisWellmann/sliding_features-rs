use std::collections::VecDeque;

use super::sliding_window::View;

pub struct CenterOfGravity {
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: f64,
}

pub fn new(window_len: usize) -> CenterOfGravity {
    return CenterOfGravity{
        window_len: window_len,
        q_vals: VecDeque::new(),
        out: 0.0,
    }
}

impl View for CenterOfGravity {
    // upate receives an observation in form of a trade and updates the view
    fn update(&mut self, val: f64) {
        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() < self.window_len {
            self.out = 0.0;
        } else {
            let mut denom: f64 = 0.0;
            let mut num: f64 = 0.0;
            let q_len = self.q_vals.len();
            for i in 0..q_len {
                let weight = q_len - i;
                let val_i = self.q_vals.get(i).unwrap();
                num += weight as f64 * val_i;
                denom += val_i;
            }
            if denom != 0.0 {
                self.out = -num / denom + (self.window_len as f64 + 1.0) / 2.0
            } else {
                self.out = 0.0;
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
    fn graph_center_of_gravity() {
        let vals = gen(1024, 100.0);
        let mut cgo = new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            cgo.update(vals[i]);
            out.push(cgo.last());
        }
        let filename = "img/center_of_gravity.png";
        plt::plt(out, filename);
    }
}

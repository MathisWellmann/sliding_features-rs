use std::collections::VecDeque;

use super::sliding_window::View;

pub struct ReFlex {
    window_len: usize,
    last_val: f64,
    last_m: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl ReFlex {
    pub fn new(window_len: usize) -> ReFlex {
        return ReFlex {
            window_len,
            last_val: 0.0,
            last_m: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl View for ReFlex {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() == 0 {
            self.last_val = val;
        }
        if self.q_vals.len() > self.window_len {
            self.q_vals.pop_front();
        }
        let a1 = 8.88442402435 / self.window_len as f64;
        let b1 = 2.0 * a1 * (4.44221201218 / self.window_len as f64).cos();
        let c3 = -a1 * a1;
        let c1 = 1.0 - b1 - c3;

        let l = self.q_vals.len();
        let mut filt: f64 = 0.0;
        if l == 0 {
            filt = c1 * (val + self.last_val) / 2.0;
        } else if l == 1 {
            let filt1 = self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 * filt1;
        } else if l > 1 {
            let filt2 = self.q_vals.get(l - 2).unwrap();
            let filt1 = self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 *filt1 + c3 * filt2;
        }
        self.last_val = val;
        self.q_vals.push_back(filt);

        let slope = (self.q_vals.get(0).unwrap() - filt) / self.window_len as f64;

        // sum the differences
        let mut d_sum: f64 = 0.0;
        for i in 0..self.q_vals.len() {
            let index = self.q_vals.len() - 1 - i;
            d_sum += (filt + i as f64 * slope) - *self.q_vals.get(index).unwrap();
        }
        d_sum /= self.window_len as f64;

        // normalize in termsn of standard deviation
        let ms0 = 0.04 * d_sum.powi(2) + 0.96 * self.last_m;
        self.last_m = ms0;
        if self.q_vals.len() < self.window_len {
            self.out = 0.0;
        } else {
            if ms0 > 0.0 {
                self.out = d_sum / ms0.sqrt();
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
    fn graph_re_flex() {
        let vals = gen(1024, 100.0);
        let mut rf = ReFlex::new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            rf.update(vals[i]);
            out.push(rf.last());
        }
        let filename = "img/re_flex.png";
        plt::plt(out, filename).unwrap();
    }
}

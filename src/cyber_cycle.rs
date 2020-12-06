use std::collections::VecDeque;

use super::sliding_window::View;

// CyberCycle with alpha set to default value of 2.0 / (windowLen + 1)
#[derive(Debug, Clone)]
pub struct CyberCycle {
    window_len: usize,
    alpha: f64,
    vals: VecDeque<f64>,
    out: VecDeque<f64>,
}

impl CyberCycle {
    pub fn new(window_len: usize) -> CyberCycle {
        return CyberCycle{
            window_len,
            alpha: 2.0 / (window_len as f64 + 1.0),
            vals: VecDeque::new(),
            out: VecDeque::new(),
        }
    }
}

impl View for CyberCycle {
    fn update(&mut self, val: f64) {
        if self.vals.len() >= self.window_len {
            self.vals.pop_front();
            self.out.pop_front();
        }
        self.vals.push_back(val);

        if self.vals.len() < self.window_len {
            self.out.push_back(0.0);
            return
        }
        let mut smooth: Vec<f64> = vec![0.0; self.vals.len()];
        let last = self.vals.len() - 1;
        for i in 3..self.vals.len() {
            smooth[i] = (val + 2.0*self.vals.get(i - 1).unwrap() + 2.0*self.vals.get(i - 2).unwrap() + *self.vals.get(i - 3).unwrap()) / 6.0
        }
        let cc = (1.0 - 0.5 * self.alpha).powi(2)
            * (smooth[last] - 2.0 * smooth[last - 1] + smooth[last - 2])
            + 2.0 * (1.0 - self.alpha) * self.out.get(last - 1).unwrap()
            - (1.0 - self.alpha).powi(2) * self.out.get(last - 2).unwrap();
        self.out.push_back(cc);
    }
    fn last(&self) -> f64 {
        return *self.out.get(self.out.len() - 1).unwrap();
    }
}

#[cfg(test)]
mod tests {
    extern crate rust_timeseries_generator;
    use self::rust_timeseries_generator::gaussian_process::gen;
    use self::rust_timeseries_generator::plt;
    use super::*;

    #[test]
    fn graph_cyber_cycle() {
        let vals = gen(1024, 100.0);
        let mut cc = CyberCycle::new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            cc.update(vals[i]);
            out.push(cc.last());
        }
        let filename = "img/cyber_cycle.png";
        plt::plt(out, filename).unwrap();
    }
}

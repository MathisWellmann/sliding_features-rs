use std::collections::VecDeque;

use super::sliding_window::View;

#[derive(Debug, Clone)]
pub struct LaguerreRSI {
    pub value: f64,
    gamma: f64,
    l0s: VecDeque<f64>,
    l1s: VecDeque<f64>,
    l2s: VecDeque<f64>,
    l3s: VecDeque<f64>,
}

impl LaguerreRSI {
    // laguerre_rsi with default gamma value of 0.5
    pub fn new(window_len: usize) -> LaguerreRSI {
        return LaguerreRSI{
            value: 0.0,
            gamma: 2.0 / (window_len as f64 + 1.0),
            l0s: VecDeque::new(),
            l1s: VecDeque::new(),
            l2s: VecDeque::new(),
            l3s: VecDeque::new(),
        }
    }
}

impl View for LaguerreRSI {
    fn update(&mut self, val: f64) {
        if self.l0s.len() >= 3 {
            self.l0s.pop_front();
            self.l1s.pop_front();
            self.l2s.pop_front();
            self.l3s.pop_front();
        }

        if self.l0s.len() < 2 {
            self.l0s.push_back(0.0);
            self.l1s.push_back(0.0);
            self.l2s.push_back(0.0);
            self.l3s.push_back(0.0);
            return
        } else {
            let last = self.l0s.len() - 1;
            self.l0s.push_back((1.0 - self.gamma) * val + self.gamma * self.l0s.get(last - 1).unwrap());
            self.l1s.push_back(-self.gamma * self.l0s.get(last).unwrap() + self.l0s.get(last - 1).unwrap() + self.gamma * self.l1s.get(last - 1).unwrap());
            self.l2s.push_back(-self.gamma * self.l1s.get(last).unwrap() + self.l1s.get(last - 1).unwrap() + self.gamma * self.l2s.get(last - 1).unwrap());
            self.l3s.push_back(-self.gamma * self.l2s.get(last).unwrap() + self.l2s.get(last - 1).unwrap() + self.gamma * self.l3s.get(last - 1).unwrap());
        }
        let last = self.l0s.len() - 1;

        let mut cu: f64 = 0.0;
        let mut cd: f64 = 0.0;
        if self.l0s.get(last) >= self.l1s.get(last) {
            cu = self.l0s.get(last).unwrap() - self.l1s.get(last).unwrap();
        } else {
            cd = self.l1s.get(last).unwrap() - self.l0s.get(last).unwrap();
        }
        if self.l1s.get(last) >= self.l2s.get(last) {
            cu += self.l1s.get(last).unwrap() - self.l2s.get(last).unwrap();
        } else {
            cd += self.l2s.get(last).unwrap() - self.l1s.get(last).unwrap();
        }
        if self.l2s.get(last) >= self.l3s.get(last) {
            cu += self.l2s.get(last).unwrap() - self.l3s.get(last).unwrap();
        } else {
            cd += self.l3s.get(last).unwrap() - self.l2s.get(last).unwrap();
        }

        if cu + cd != 0.0 {
            self.value = cu / (cu + cd);
        }
    }
    fn last(&self) -> f64 {
        return self.value;
    }
}

#[cfg(test)]
mod tests {
    extern crate rust_timeseries_generator;
    use self::rust_timeseries_generator::plt;
    use super::*;
    use self::rust_timeseries_generator::gaussian_process::gen;

    #[test]
    fn test_range() {
        let vals = gen(1024, 100.0);
        let mut lrsi = LaguerreRSI::new(16);
        for i in 0..vals.len() {
            lrsi.update(vals[i]);
            let last = lrsi.last();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }

    #[test]
    fn graph_laguerre_rsi() {
        let vals = gen(1024, 100.0);
        let mut lrsi = LaguerreRSI::new(16);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            lrsi.update(vals[i]);
            out.push(lrsi.last());
        }
        // graph the resutls
        let filename = "img/laguerre_rsi.png";
        plt::plt(out, filename).unwrap();
    }
}

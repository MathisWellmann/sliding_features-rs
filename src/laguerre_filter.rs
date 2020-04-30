use crate::sliding_window::View;

pub struct LaguerreFilter {
    gamma: f64,
    l0s: Vec<f64>,
    l1s: Vec<f64>,
    l2s: Vec<f64>,
    l3s: Vec<f64>,
    filts: Vec<f64>,
    init: bool,
}

impl LaguerreFilter {
    pub fn new(gamma: f64) -> LaguerreFilter {
        return LaguerreFilter {
            gamma,
            l0s: Vec::new(),
            l1s: Vec::new(),
            l2s: Vec::new(),
            l3s: Vec::new(),
            filts: Vec::new(),
            init: true,
        }
    }

    pub fn default() -> LaguerreFilter {
        return LaguerreFilter::new(0.8)
    }
}


impl View for LaguerreFilter {
    fn update(&mut self, val: f64) {
        if self.init {
            self.init = false;
            self.l0s.push(val);
            self.l1s.push(val);
            self.l2s.push(val);
            self.l3s.push(val);
            self.filts.push((self.l0s[0] + 2.0*self.l1s[0] + 2.0*self.l2s[0] + self.l3s[0]) / 6.0);
            return
        }
        self.l0s.push((1.0 - self.gamma) * val + self.gamma * self.l0s[self.l0s.len() - 1]);
        self.l1s.push(-self.gamma * self.l0s[self.l0s.len() - 1] + self.l0s[self.l0s.len() - 2] + self.gamma * self.l1s[self.l1s.len() - 1]);
        self.l2s.push(-self.gamma * self.l1s[self.l1s.len() - 1] + self.l1s[self.l1s.len() - 2] + self.gamma * self.l2s[self.l2s.len() - 1]);
        self.l3s.push(-self.gamma * self.l2s[self.l2s.len() - 1] + self.l2s[self.l2s.len() - 2] + self.gamma * self.l3s[self.l3s.len() - 1]);
        self.filts.push((self.l0s[self.l0s.len() - 1] + 2.0*self.l1s[self.l1s.len() - 1] + 2.0*self.l2s[self.l2s.len() - 1] + self.l3s[self.l3s.len() - 1]) / 6.0);
    }

    fn last(&self) -> f64 {
        return self.filts[self.filts.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    extern crate rust_timeseries_generator;
    use rust_timeseries_generator::gaussian_process;
    use rust_timeseries_generator::plt;
    use rand::{Rng, thread_rng};

    #[test]
    fn test_laguerre_filter() {
        let mut rng = thread_rng();
        let mut laguerre = LaguerreFilter::default();
        for _i in 0..1_000 {
            let r = rng.gen::<f64>();
            laguerre.update(r);
            let last = laguerre.last();

            assert!(last <= 1.0);
            assert!(last >= 0.0);
        }
    }

    #[test]
    fn test_laguerre_filter_graph() {
        let vals = gaussian_process::gen(1024, 100.0);
        let mut laguerre = LaguerreFilter::default();
        let mut out: Vec<f64> = Vec::new();
        for v in &vals {
            laguerre.update(*v);
            out.push(laguerre.last());
        }
        let filename = "img/laguerre_filter.png";
        plt::plt(out, filename).unwrap();
    }
}
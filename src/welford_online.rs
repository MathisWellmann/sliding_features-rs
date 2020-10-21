use crate::sliding_window::View;

#[derive(Debug)]
pub struct WelfordOnline {
    mean: f64,
    s: f64,
    n: usize,
}

impl WelfordOnline {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            s: 0.0,
            n: 0,
        }
    }

    fn variance(&self) -> f64 {
        return if self.n > 1 {
            self.s / (self.n as f64 - 1.0)
        } else {
            0.0
        }
    }
}

impl View for WelfordOnline {
    fn update(&mut self, val: f64) {
        self.n += 1;
        let old_mean = self.mean;
        self.mean += (val - old_mean) / self.n as f64;
        self.s += (val - old_mean) * (val - self.mean);
    }

    fn last(&self) -> f64 {
        let std_dev = self.variance().sqrt();
        std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_timeseries_generator::{gaussian_process};
    
    #[test]
    fn correct_std_dev() {
        let vals = gaussian_process::gen(10_000, 100.0);
        let mut wo = WelfordOnline::new();
        for v in &vals {
            wo.update(*v);
            assert!(!wo.last().is_nan());
        }
        let w_std_dev = wo.last();
        let avg: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let std_dev: f64 = ((1.0 / (vals.len() as f64 - 1.0)) * vals.iter().map(|v| (v - avg).powi(2)).sum::<f64>()).sqrt();

        assert_eq!(w_std_dev, std_dev);
    }
}

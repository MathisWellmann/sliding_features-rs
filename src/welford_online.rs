use crate::sliding_window::View;
use crate::Echo;

/// Welford online algorithm for computing mean and variance on-the-fly
#[derive(Clone)]
pub struct WelfordOnline {
    view: Box<dyn View>,
    mean: f64,
    s: f64,
    n: usize,
}

impl WelfordOnline {
    /// Create a WelfordOnline struct with a chained View
    pub fn new(view: Box<dyn View>) -> Self {
        Self {
            view,
            mean: 0.0,
            s: 0.0,
            n: 0,
        }
    }

    /// Create a new WelfordOnline Sliding Window without a chained View
    pub fn new_final() -> Self {
        Self::new(Box::new(Echo::new()))
    }

    /// Return the variance of the sliding window
    pub fn variance(&self) -> f64 {
        return if self.n > 1 {
            self.s / (self.n as f64 - 1.0)
        } else {
            0.0
        };
    }

    /// Return the mean of the sliding window
    pub fn mean(&self) -> f64 {
        self.mean
    }
}

impl View for WelfordOnline {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

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
    use crate::test_data::TEST_DATA;
    use round::round;

    #[test]
    fn correct_std_dev() {
        let mut wo = WelfordOnline::new_final();
        for v in &TEST_DATA {
            wo.update(*v);
            assert!(!wo.last().is_nan());
        }
        let w_std_dev = wo.last();

        // compute the standard deviation with the regular formula
        let avg: f64 = TEST_DATA.iter().sum::<f64>() / TEST_DATA.len() as f64;
        let std_dev: f64 = ((1.0 / (TEST_DATA.len() as f64 - 1.0))
            * TEST_DATA.iter().map(|v| (v - avg).powi(2)).sum::<f64>())
        .sqrt();

        assert_eq!(round(w_std_dev, 4), round(std_dev, 4));
    }
}

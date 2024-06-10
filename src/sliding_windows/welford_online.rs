//! Welford online algorithm for computing mean and variance on-the-fly
//! over a sliding window

use crate::View;
use getset::CopyGetters;
use std::collections::VecDeque;

/// Welford online algorithm for computing mean and variance on-the-fly
/// over a sliding window
#[derive(Debug, Clone, CopyGetters)]
pub struct WelfordOnline<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    /// The mean of the observed samples
    #[getset(get_copy = "pub")]
    mean: f64,
    s: f64,
    n: usize,
}

impl<V> WelfordOnline<V>
where
    V: View,
{
    /// Create a WelfordOnline struct with a chained View
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            mean: 0.0,
            s: 0.0,
            n: 0,
        }
    }

    /// Return the variance of the sliding window using biased estimator.
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.s / self.n as f64
        } else {
            0.0
        }
    }
}

impl<V> View for WelfordOnline<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            let old_val: f64 = self.q_vals.pop_front().unwrap();
            // remove old value from estimation
            let new_mean = (self.n as f64 * self.mean - old_val) / (self.n as f64 - 1.0);
            self.s -= (old_val - self.mean) * (old_val - new_mean);
            self.mean = new_mean;
            self.n -= 1;
        }
        self.q_vals.push_back(val);

        self.n += 1;
        let old_mean = self.mean;
        self.mean += (val - old_mean) / self.n as f64;
        self.s += (val - old_mean) * (val - self.mean);
    }

    fn last(&self) -> Option<f64> {
        if self.n < self.window_len {
            // To ensure we don't return anything when there are not enough samples.
            return None;
        }
        Some(self.variance().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;
    use round::round;

    #[test]
    fn welford_online() {
        let mut wo = WelfordOnline::new(Echo::new(), TEST_DATA.len());
        for v in &TEST_DATA {
            wo.update(*v);
            if let Some(val) = wo.last() {
                assert!(!val.is_nan());
            }
        }
        let w_std_dev = wo.last().expect("Is some");

        // compute the standard deviation with the regular formula
        let avg: f64 = TEST_DATA.iter().sum::<f64>() / TEST_DATA.len() as f64;
        let std_dev: f64 = ((1.0 / (TEST_DATA.len() as f64))
            * TEST_DATA.iter().map(|v| (v - avg).powi(2)).sum::<f64>())
        .sqrt();

        assert_eq!(round(w_std_dev, 4), round(std_dev, 4));
    }

    #[test]
    fn welford_online_plot() {
        let mut wo = WelfordOnline::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            wo.update(*v);
            if let Some(val) = wo.last() {
                out.push(val);
            }
        }
        let filename = "img/welford_online_sliding.png";
        plot_values(out, filename).unwrap();
    }
}

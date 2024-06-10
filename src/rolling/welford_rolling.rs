//! Welford online algorithm for computing mean and variance on-the-fly
//! over a sliding window

use crate::{pure_functions::Echo, View};
use getset::CopyGetters;

/// Welford online algorithm for computing mean and variance on-the-fly
/// over a sliding window
#[derive(Debug, Clone, CopyGetters)]
pub struct WelfordRolling<V> {
    view: V,
    /// The mean of the observed samples
    #[getset(get_copy = "pub")]
    mean: f64,
    s: f64,
    n: usize,
}

impl Default for WelfordRolling<Echo> {
    fn default() -> Self {
        Self::new(Echo::new())
    }
}

impl<V> WelfordRolling<V>
where
    V: View,
{
    /// Create a WelfordOnline struct with a chained View
    pub fn new(view: V) -> Self {
        Self {
            view,
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

impl<V> View for WelfordRolling<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.n += 1;
        let old_mean = self.mean;
        self.mean += (val - old_mean) / self.n as f64;
        self.s += (val - old_mean) * (val - self.mean);
    }

    fn last(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        Some(self.variance().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use round::round;

    #[test]
    fn welford_online() {
        let mut wo = WelfordRolling::default();
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
        let mut wo = WelfordRolling::default();
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

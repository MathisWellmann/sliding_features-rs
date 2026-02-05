//! Welford online algorithm for computing mean and variance on-the-fly
//! over a sliding window

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
};

/// Welford online algorithm for computing mean and variance on-the-fly
/// over a sliding window
#[derive(Debug, Clone, CopyGetters)]
pub struct WelfordRolling<T: Float, V> {
    view: V,
    /// The mean of the observed samples
    #[getset(get_copy = "pub")]
    mean: T,
    s: T,
    n: usize,
}

impl<T: Float> Default for WelfordRolling<T, Echo<T>> {
    fn default() -> Self {
        Self::new(Echo::new())
    }
}

impl<T, V> WelfordRolling<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a WelfordOnline struct with a chained View
    pub fn new(view: V) -> Self {
        Self {
            view,
            mean: T::zero(),
            s: T::zero(),
            n: 0,
        }
    }

    /// Return the variance of the sliding window using biased estimator.
    pub fn variance(&self) -> T {
        if self.n > 1 {
            self.s / T::from(self.n).expect("Can convert")
        } else {
            T::zero()
        }
    }
}

impl<T, V> View<T> for WelfordRolling<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.n += 1;
        let old_mean = self.mean;
        self.mean = self.mean + (val - old_mean) / T::from(self.n).expect("Can convert");
        self.s = self.s + (val - old_mean) * (val - self.mean);
    }

    fn last(&self) -> Option<T> {
        if self.n == 0 {
            return None;
        }
        let out = self.variance().sqrt();
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use round::round;

    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

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

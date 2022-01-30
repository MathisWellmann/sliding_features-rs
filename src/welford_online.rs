//! Welford online algorithm for computing mean and variance on-the-fly

use crate::Echo;
use crate::View;

/// Welford online algorithm for computing mean and variance on-the-fly
#[derive(Clone)]
pub struct WelfordOnline<V> {
    view: V,
    mean: f64,
    s: f64,
    n: usize,
}

impl<V> std::fmt::Debug for WelfordOnline<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "WelfordOnline(mean: {}, s: {}, n: {})",
            self.mean, self.s, self.n
        )
    }
}

/// Create a new WelfordOnline Sliding Window without a chained View
#[inline(always)]
pub fn new_final() -> WelfordOnline<Echo> {
    WelfordOnline::new(Echo::new())
}

impl<V> WelfordOnline<V>
where
    V: View,
{
    /// Create a WelfordOnline struct with a chained View
    #[inline]
    pub fn new(view: V) -> Self {
        Self {
            view,
            mean: 0.0,
            s: 0.0,
            n: 0,
        }
    }

    /// Return the variance of the sliding window
    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.s / (self.n as f64 - 1.0)
        } else {
            0.0
        }
    }

    /// Return the mean of the sliding window
    #[inline(always)]
    pub fn mean(&self) -> f64 {
        self.mean
    }
}

impl<V> View for WelfordOnline<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.n += 1;
        let old_mean = self.mean;
        self.mean += (val - old_mean) / self.n as f64;
        self.s += (val - old_mean) * (val - self.mean);
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use round::round;

    #[test]
    fn correct_std_dev() {
        let mut wo = new_final();
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

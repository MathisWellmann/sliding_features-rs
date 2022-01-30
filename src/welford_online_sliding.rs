//! Welford online algorithm for computing mean and variance on-the-fly
//! over a sliding window

use crate::Echo;
use crate::View;
use std::collections::VecDeque;

/// Welford online algorithm for computing mean and variance on-the-fly
/// over a sliding window
#[derive(Clone)]
pub struct WelfordOnlineSliding<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    mean: f64,
    s: f64,
    n: usize,
}

impl<V> std::fmt::Debug for WelfordOnlineSliding<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "WelfordOnlineSliding(window_len: {}, q_vals: {:?}, mean: {}, s: {}, n: {})",
            self.window_len, self.q_vals, self.mean, self.s, self.n
        )
    }
}

/// Create a new WelfordOnline Sliding Window without a chained View
#[inline(always)]
pub fn new_final(window_len: usize) -> WelfordOnlineSliding<Echo> {
    WelfordOnlineSliding::new(Echo::new(), window_len)
}

impl<V> WelfordOnlineSliding<V>
where
    V: View,
{
    /// Create a WelfordOnline struct with a chained View
    #[inline]
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

impl<V> View for WelfordOnlineSliding<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

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

    #[inline(always)]
    fn last(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use round::round;

    #[test]
    fn welford_online_sliding() {
        let mut wo = new_final(TEST_DATA.len());
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

    #[test]
    fn welford_online_sliding_plot() {
        let mut wo = new_final(16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            wo.update(*v);
            out.push(wo.last());
        }
        let filename = "img/welford_online_sliding.png";
        plot_values(out, filename).unwrap();
    }
}

//! Welford online algorithm for computing mean and variance on-the-fly
//! over a sliding window

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Welford online algorithm for computing mean and variance on-the-fly
/// over a sliding window
#[derive(Debug, Clone, CopyGetters)]
pub struct WelfordOnline<T: Float, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    /// The mean of the observed samples
    #[getset(get_copy = "pub")]
    mean: T,
    m2: T,
    count: usize,
}

impl<T, V> WelfordOnline<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a WelfordOnline struct with a chained View
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            mean: T::zero(),
            m2: T::zero(),
            count: 0,
        }
    }

    #[inline]
    fn update_stats_add(&mut self, x: T) {
        let delta = x - self.mean;
        self.mean = self.mean + (delta / T::from(self.count + 1).unwrap());
        self.m2 = self.m2 + (delta * (x - self.mean));
        self.count += 1;
    }

    #[inline]
    fn update_stats_remove(&mut self, old_value: T) {
        // Derive sum and sum-of-squares from Welford state:
        //   mean = sum / n,  m2 = sum_sq - n * mean^2
        //   => sum = n * mean,  sum_sq = m2 + n * mean^2
        let n = T::from(self.count).unwrap();
        let sum = self.mean * n;
        let sum_sq = self.m2 + sum * self.mean;

        let sum_new = sum - old_value;
        let sum_sq_new = sum_sq - old_value * old_value;
        self.count -= 1;

        if self.count > 0 {
            let n_new = T::from(self.count).unwrap();
            self.mean = sum_new / n_new;
            self.m2 = sum_sq_new - n_new * self.mean * self.mean;
            // Clamp floating-point noise that could push m2 slightly negative.
            if self.m2 < T::zero() {
                self.m2 = T::zero();
            }
        } else {
            self.mean = T::zero();
            self.m2 = T::zero();
        }
    }

    /// Return the variance of the sliding window
    #[inline]
    pub fn variance(&self) -> T {
        if self.count > 1 {
            self.m2 / T::from(self.count - 1).expect("can convert")
        } else {
            T::zero()
        }
    }
}

impl<T, V> View<T> for WelfordOnline<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.q_vals.push_back(val);

        if self.q_vals.len() > self.window_len.get() {
            let old_val = self.q_vals.pop_front().unwrap();
            self.update_stats_remove(old_val);
        }
        self.update_stats_add(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.count < self.window_len.get() - 1 {
            // To ensure we don't return anything when there are not enough samples.
            return None;
        }
        let var = self.variance();
        if var <= T::zero() {
            return Some(T::zero());
        }
        debug_assert!(var >= T::zero(), "Variance must be positive");
        let out = var.sqrt();
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
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    #[test]
    fn welford_online() {
        let mut wo = WelfordOnline::new(Echo::new(), NonZeroUsize::new(TEST_DATA.len()).unwrap());
        for v in &TEST_DATA {
            wo.update(*v);
            if let Some(val) = wo.last() {
                assert!(!val.is_nan());
            }
        }
        let w_std_dev = wo.last().expect("Is some");

        // compute the standard deviation with the regular formula
        let avg: f64 = TEST_DATA.iter().sum::<f64>() / TEST_DATA.len() as f64;
        let std_dev: f64 = ((1.0 / (TEST_DATA.len() as f64 - 1.0))
            * TEST_DATA.iter().map(|v| (v - avg).powi(2)).sum::<f64>())
        .sqrt();

        assert_eq!(round(w_std_dev, 4), round(std_dev, 4));
    }

    #[test]
    fn welford_online_sliding_matches_direct_computation() {
        // Feed [1, 2, 3, 4, 5, 6] through a window of length 3.
        // After each step, verify the std dev matches the direct formula for
        // the window contents.
        let mut wo = WelfordOnline::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        let all: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Direct std-dev helper (sample sd).
        let direct_sd = |vs: &[f64]| -> f64 {
            let n = vs.len() as f64;
            let m = vs.iter().sum::<f64>() / n;
            let var = vs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        };

        for i in 0..all.len() {
            wo.update(all[i]);
            let start = if i < 2 { 0 } else { i - 2 }; // i+1-window
            let window = &all[start..=i];
            let expected = direct_sd(&window);
            if let Some(got) = wo.last() {
                let diff = (got - expected).abs();
                if diff > 1e-12 {
                    panic!(
                        "step {} window {:?}: expected sd={}, got={}, diff={}",
                        i, window, expected, got, diff
                    );
                }
            } else {
                // window not full yet → None is correct for i < 2
                assert!(
                    i < 2,
                    "step {}: got None but window is full ({:?})",
                    i,
                    window
                );
            }
        }
    }

    #[test]
    fn welford_online_plot() {
        let mut wo = WelfordOnline::new(Echo::new(), NonZeroUsize::new(16).unwrap());
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

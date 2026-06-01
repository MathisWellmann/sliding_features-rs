//! Z-Score Standardization over a sliding window.
//!
//! Computes `(x - μ) / σ` where μ and σ are the sample mean and
//! standard deviation of the sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::WelfordOnline;
use crate::{
    View,
    pure_functions::Echo,
};

/// Z-Score Standardization over a sliding window.
///
/// Computes `(x - μ) / σ` where μ and σ are the sample mean and
/// sample standard deviation derived from the values in the sliding window.
#[derive(Debug, Clone)]
pub struct ZScoreStandardization<T: Float, V> {
    view: V,
    welford_online: WelfordOnline<T, Echo<T>>,
    last: T,
}

impl<T, V> ZScoreStandardization<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new ZScoreStandardization with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        ZScoreStandardization {
            view,
            welford_online: WelfordOnline::new(Echo::new(), window_len),
            last: T::zero(),
        }
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> NonZeroUsize {
        self.welford_online.window_len()
    }
}

impl<T, V> View<T> for ZScoreStandardization<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.welford_online.update(val);
        self.last = val;
    }

    fn last(&self) -> Option<T> {
        let std_dev = self.welford_online.last()?;
        if std_dev == T::zero() {
            return Some(T::zero());
        }
        let mean = self.welford_online.mean();
        let out = (self.last - mean) / std_dev;
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    #[test]
    fn z_score_plot() {
        let mut vsct = ZScoreStandardization::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            vsct.update(*v);
            if let Some(val) = vsct.last() {
                out.push(val);
            }
        }
        let filename = "img/vsct.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn z_score_matches_direct_computation() {
        // Feed [1, 2, 3, 4, 5, 6] through a z-score normalizer with window=3.
        // After each step, verify the z-score of the last value equals the
        // direct formula: (x - μ) / σ  where μ,σ are sample stats of the window.
        let mut zs = ZScoreStandardization::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        let all: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let direct_z_score = |vs: &[f64]| -> f64 {
            let n = vs.len() as f64;
            let m = vs.iter().sum::<f64>() / n;
            let var = vs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
            let last = vs.last().unwrap();
            (last - m) / var.sqrt()
        };

        for i in 0..all.len() {
            zs.update(all[i]);
            let start = if i < 2 { 0 } else { i - 2 };
            let window = &all[start..=i];
            let expected = direct_z_score(&window);
            if let Some(got) = zs.last() {
                let diff = (got - expected).abs();
                assert!(
                    diff < 1e-12,
                    "step {} window {:?}: expected z={}, got={}, diff={}",
                    i,
                    window,
                    expected,
                    got,
                    diff
                );
            } else {
                // window not full yet → None is correct for first 2 steps
                assert!(
                    i < 2,
                    "step {}: got None but window has {} elements",
                    i,
                    window.len()
                );
            }
        }
    }

    #[test]
    fn z_score_of_identical_values_is_zero() {
        // When all values in the window are the same, std_dev = 0 → z = 0.
        let mut zs = ZScoreStandardization::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for _ in 0..10 {
            zs.update(5.0);
        }
        assert_eq!(zs.last(), Some(0.0));
    }
}

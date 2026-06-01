//! Z-Score Standardization over a sliding window.
//!
//! Computes `(x - μ) / σ` where μ and σ are the sample mean and
//! standard deviation of the previous sliding window.

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
/// sample standard deviation derived from the previous sliding window.
///
/// The current value is intentionally excluded from the statistics used to
/// standardize it, so the transformation only depends on past values and does
/// not leak information from the current/future sample into the rolling mean or
/// standard deviation.
#[derive(Debug, Clone)]
pub struct ZScoreStandardization<T: Float, V> {
    view: V,
    welford_online: WelfordOnline<T, Echo<T>>,
    history_len: usize,
    out: Option<T>,
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
            history_len: 0,
            out: None,
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

        self.out = if self.history_len >= self.window_len().get() {
            let std_dev = self.welford_online.last().expect("history is warm");
            if std_dev == T::zero() {
                Some(T::zero())
            } else {
                let mean = self.welford_online.mean();
                let out = (val - mean) / std_dev;
                debug_assert!(out.is_finite(), "value must be finite");
                Some(out)
            }
        } else {
            None
        };

        self.welford_online.update(val);
        self.history_len = (self.history_len + 1).min(self.window_len().get());
    }

    fn last(&self) -> Option<T> {
        self.out
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
    fn z_score_matches_direct_past_window_computation() {
        // Feed [1, 2, 3, 4, 5, 6] through a z-score normalizer with window=3.
        // After warm-up, verify the z-score of the current value equals the
        // direct formula: (x - μ) / σ where μ,σ are sample stats of the three
        // previous values. The current value must not be part of the fitted
        // normalization window.
        let mut zs = ZScoreStandardization::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        let all: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let direct_z_score = |x: f64, history: &[f64]| -> f64 {
            let n = history.len() as f64;
            let m = history.iter().sum::<f64>() / n;
            let var = history.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
            (x - m) / var.sqrt()
        };

        for i in 0..all.len() {
            zs.update(all[i]);
            if i >= 3 {
                let history = &all[i - 3..i];
                let expected = direct_z_score(all[i], history);
                let got = zs.last().unwrap();
                let diff = (got - expected).abs();
                assert!(
                    diff < 1e-12,
                    "step {} history {:?}: expected z={}, got={}, diff={}",
                    i,
                    history,
                    expected,
                    got,
                    diff
                );
            } else {
                assert!(
                    zs.last().is_none(),
                    "step {}: got {:?} before {} past samples were available",
                    i,
                    zs.last(),
                    zs.window_len()
                );
            }
        }
    }

    #[test]
    fn z_score_does_not_leak_current_value_into_statistics() {
        let mut zs = ZScoreStandardization::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        for val in [1.0, 2.0, 3.0] {
            zs.update(val);
        }

        zs.update(100.0);

        // If the current value leaked into the statistics, this would be the
        // z-score against [2, 3, 100] (roughly 1.1545). The causal value uses
        // only the previous window [1, 2, 3].
        let expected = (100.0 - 2.0) / 1.0;
        let got = zs.last().unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "expected {expected}, got {got}"
        );
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

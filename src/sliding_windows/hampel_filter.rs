// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Hampel identifier: replaces outliers with the window median.

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// The MAD-to-standard-deviation scale factor for a normal distribution:
/// `1 / Phi^-1(3/4) ~= 1.4826`, making the scaled MAD a consistent estimator
/// of the standard deviation under normality (Hampel, 1974, "The Influence
/// Curve and its Role in Robust Estimation").
const MAD_SCALE: f64 = 1.4826;

/// The Hampel identifier (Hampel, 1974) over a trailing window of the last
/// `window_len` values, in its **causal** form: the window ends at the
/// current observation (rather than being centered on it, as in the
/// original offline filter), so it can run online without a future-data
/// buffer. A value is an outlier if it deviates from the window's median by
/// more than `t` times the scaled median absolute deviation (MAD) of the
/// window: `|x - median| > t * 1.4826 * MAD`. Outliers are replaced by the
/// median; other values pass through unchanged.
///
/// `t` is the threshold in robust-scale units (commonly 3, analogous to a
/// 3-sigma rule but robust to the outliers it is meant to detect).
///
/// `last()` is `None` until `window_len` values were observed. `is_outlier()`
/// reports whether the most recent observation was flagged (and therefore
/// replaced).
///
/// Complexity per update: O(log n) to maintain the sorted window plus
/// O(n log n) to sort the absolute deviations for the MAD (a preallocated
/// buffer, so no per-update allocation); O(n) memory.
#[derive(Debug, Clone)]
pub struct HampelFilter<T, V> {
    view: V,
    t: T,
    window: SortedWindow<T>,
    /// Scratch buffer for the absolute deviations from the median, reused
    /// every update to avoid allocating.
    abs_dev: Vec<T>,
    is_outlier: bool,
    out: Option<T>,
}

impl<T, V> HampelFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new filter with a chained `View`, a sliding window length
    /// and a threshold `t` in scaled-MAD units.
    ///
    /// # Panics
    /// If `t <= 0`.
    pub fn new(view: V, window_len: NonZeroUsize, t: T) -> Self {
        assert!(t > T::zero(), "t must be > 0");
        Self {
            view,
            t,
            window: SortedWindow::new(window_len),
            abs_dev: Vec::with_capacity(window_len.get()),
            is_outlier: false,
            out: None,
        }
    }

    /// The outlier threshold, in scaled-MAD units.
    #[inline]
    pub fn t(&self) -> T {
        self.t
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }

    /// Whether the most recently observed value was flagged as an outlier
    /// (and therefore replaced by the window median).
    #[inline]
    pub fn is_outlier(&self) -> bool {
        self.is_outlier
    }

    /// The median of `self.abs_dev`, which must already be sorted.
    fn median_of_sorted_abs_dev(&self) -> T {
        let n = self.abs_dev.len();
        debug_assert!(n > 0, "the window is full, so abs_dev is not empty");
        let mid = n / 2;
        if n % 2 == 1 {
            self.abs_dev[mid]
        } else {
            (self.abs_dev[mid - 1] + self.abs_dev[mid]) / (T::one() + T::one())
        }
    }
}

impl<T, V> View<T> for HampelFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.window.push(val);
        if !self.window.is_full() {
            self.is_outlier = false;
            self.out = None;
            return;
        }

        let median = self
            .window
            .quantile(T::from(0.5).expect("can convert"))
            .expect("a full window is not empty");
        self.abs_dev.clear();
        self.abs_dev
            .extend(self.window.iter().map(|x| (x - median).abs()));
        self.abs_dev
            .sort_by(|a, b| a.partial_cmp(b).expect("values are finite"));
        let mad = self.median_of_sorted_abs_dev();
        let scaled_mad = T::from(MAD_SCALE).expect("can convert") * mad;

        self.is_outlier = (val - median).abs() > self.t * scaled_mad;
        let out = if self.is_outlier { median } else { val };
        debug_assert!(out.is_finite(), "value must be finite");
        self.out = Some(out);
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
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn hampel(window_len: usize, t: f64) -> HampelFilter<f64, Echo<f64>> {
        HampelFilter::new(Echo::new(), NonZeroUsize::new(window_len).unwrap(), t)
    }

    #[test]
    fn hampel_plot() {
        let mut f = hampel(11, 3.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/hampel_filter.png").unwrap();
    }

    #[test]
    fn hampel_warmup() {
        let mut f = hampel(5, 3.0);
        for x in [1.0, 2.0, 3.0, 4.0] {
            f.update(x);
            assert_eq!(f.last(), None);
        }
        f.update(5.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn hampel_replaces_injected_spikes_and_leaves_clean_data_unchanged() {
        let mut f = hampel(9, 3.0);
        let mut clean = vec![10.0, 10.1, 9.9, 10.05, 9.95, 10.0, 10.1, 9.9, 10.0];
        // A large spike well outside the clean data's spread.
        clean.push(50.0);
        clean.extend([10.0, 9.95, 10.05, 10.0, 9.9, 10.1]);

        let mut outputs = Vec::new();
        let mut outlier_flags = Vec::new();
        for x in &clean {
            f.update(*x);
            outputs.push(f.last());
            outlier_flags.push(f.is_outlier());
        }

        let spike_idx = 9;
        assert!(
            outlier_flags[spike_idx],
            "the injected spike must be flagged as an outlier"
        );
        assert_ne!(
            outputs[spike_idx].unwrap(),
            50.0,
            "the spike must be replaced by the median, not passed through"
        );
        // Once the window has flushed the spike out, clean values pass
        // through unchanged.
        let last = clean.len() - 1;
        assert!(!outlier_flags[last], "clean data must not be flagged");
        assert_eq!(outputs[last], Some(clean[last]));
    }

    #[test]
    fn hampel_constant_input_never_flags_outliers() {
        let mut f = hampel(5, 3.0);
        for _ in 0..20 {
            f.update(7.5);
        }
        assert_eq!(f.last(), Some(7.5));
        assert!(!f.is_outlier());
    }

    #[test]
    fn hampel_vs_naive_reference() {
        // A direct, unoptimized transcription of the causal Hampel
        // identifier operating on a plain slice window.
        fn naive(window: &[f64], t: f64) -> (f64, bool) {
            let mut sorted = window.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            let n = sorted.len();
            let median = if n.is_multiple_of(2) {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                median
            };
            let mut dev: Vec<f64> = window.iter().map(|x| (x - median).abs()).collect();
            dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = if n.is_multiple_of(2) {
                (dev[n / 2 - 1] + dev[n / 2]) / 2.0
            } else {
                dev[n / 2]
            };
            let scaled_mad = 1.4826 * mad;
            let x = *window.last().unwrap();
            let is_outlier = (x - median).abs() > t * scaled_mad;
            (if is_outlier { median } else { x }, is_outlier)
        }

        let rng = romu::Rng::from_seed_with_64bit(21);
        let window_len = 7;
        let t = 3.0;
        let vals: Vec<f64> = (0..500)
            .map(|_| match rng.range_usize(0..10) {
                0 => rng.f64() * 1000.0, // occasional spike
                _ => 10.0 + (rng.f64() - 0.5) * 2.0,
            })
            .collect();
        let mut f = hampel(window_len, t);
        for (i, x) in vals.iter().enumerate() {
            f.update(*x);
            if i + 1 < window_len {
                assert_eq!(f.last(), None);
                continue;
            }
            let (expected_out, expected_outlier) = naive(&vals[i + 1 - window_len..=i], t);
            assert_eq!(f.last(), Some(expected_out), "step {i}");
            assert_eq!(f.is_outlier(), expected_outlier, "step {i}");
        }
    }

    #[test]
    #[should_panic(expected = "t must be > 0")]
    fn hampel_rejects_non_positive_t() {
        let _ = hampel(5, 0.0);
    }
}

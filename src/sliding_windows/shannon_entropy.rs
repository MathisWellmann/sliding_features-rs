// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Normalized Shannon entropy of the binned values of a sliding window.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use super::complexity_histogram::ComplexityHistogram;
use crate::View;

/// Normalized Shannon entropy (Shannon, 1948, "A Mathematical Theory of Communication")
/// of the last `window_len` values, binned into `bins` equal-width bins over the fixed range `[lo, hi]`.
///
/// `H = -sum_i p_i ln(p_i) / ln(bins)` where `p_i` is the fraction of window values in bin `i`.
///
/// Output range `[0, 1]`: 0 when all values fall in one bin, 1 when they are spread evenly.
/// When `window_len < bins` the maximum reachable value is `ln(window_len) / ln(bins)`.
///
/// Why a fixed range instead of the window's current `[min, max]`:
/// - The bin of each value never changes, so counts update incrementally in O(1)
///   (a min/max range would force re-binning the whole window, O(window_len), on every update).
/// - Bin edges do not jump when an extreme value enters or leaves the window,
///   so outputs at different times are comparable.
///
/// Values outside `[lo, hi]` are clamped into the first / last bin.
/// To use it on unbounded input (e.g. prices), chain it after a normalizing view
/// with a known output range.
///
/// Complexity per update: O(1) (amortized, see the shared histogram).
#[derive(Debug, Clone, CopyGetters)]
pub struct ShannonEntropy<T: Float, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The number of equal-width bins.
    #[getset(get_copy = "pub")]
    bins: NonZeroUsize,
    /// The lower end of the binned range.
    #[getset(get_copy = "pub")]
    lo: T,
    /// The upper end of the binned range.
    #[getset(get_copy = "pub")]
    hi: T,
    bin_idxs: VecDeque<usize>,
    histogram: ComplexityHistogram<T>,
    ln_bins: T,
}

impl<T, V> ShannonEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `ShannonEntropy` with a chained `View`.
    ///
    /// # Panics
    /// If `bins < 2`, if `lo` or `hi` is not finite, or if `lo >= hi`.
    pub fn new(view: V, window_len: NonZeroUsize, bins: NonZeroUsize, lo: T, hi: T) -> Self {
        assert!(bins.get() >= 2, "bins must be at least 2");
        assert!(lo.is_finite(), "lo must be finite");
        assert!(hi.is_finite(), "hi must be finite");
        assert!(lo < hi, "lo must be smaller than hi");
        Self {
            view,
            window_len,
            bins,
            lo,
            hi,
            bin_idxs: VecDeque::with_capacity(window_len.get()),
            histogram: ComplexityHistogram::new(bins.get()),
            ln_bins: T::from(bins.get()).expect("can convert").ln(),
        }
    }
}

/// Index of the equal-width bin of `val`, values outside `[lo, hi]` are clamped.
#[inline]
fn bin_index<T: Float>(val: T, lo: T, hi: T, bins: usize) -> usize {
    let pos = (val - lo) / (hi - lo) * T::from(bins).expect("can convert");
    // NaN must not panic in release builds (inputs are only debug-asserted finite).
    if pos.is_nan() || pos <= T::zero() {
        return 0;
    }
    let idx = pos.floor().to_usize().unwrap_or(bins - 1).min(bins - 1);
    debug_assert!(idx < bins, "bin index must be in range");
    idx
}

impl<T, V> View<T> for ShannonEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.bin_idxs.len() == self.window_len.get() {
            let old = self
                .bin_idxs
                .pop_front()
                .expect("window is full, so not empty");
            self.histogram.remove(old);
        }
        let idx = bin_index(val, self.lo, self.hi, self.bins.get());
        self.bin_idxs.push_back(idx);
        self.histogram.add(idx);
        debug_assert_eq!(self.histogram.total(), self.bin_idxs.len());
    }

    fn last(&self) -> Option<T> {
        if self.bin_idxs.len() < self.window_len.get() {
            return None;
        }
        let out = (self.histogram.entropy() / self.ln_bins).min(T::one());
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(out >= T::zero(), "entropy must be non-negative");
        Some(out)
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

    fn naive(window: &[f64], bins: usize, lo: f64, hi: f64) -> f64 {
        let mut counts = vec![0_usize; bins];
        for &v in window {
            let pos = ((v - lo) / (hi - lo) * bins as f64).floor();
            let idx = pos.clamp(0.0, (bins - 1) as f64) as usize;
            counts[idx] += 1;
        }
        let n = window.len() as f64;
        let h: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| -(c as f64 / n) * (c as f64 / n).ln())
            .sum();
        h / (bins as f64).ln()
    }

    fn new_f64(window_len: usize, bins: usize, lo: f64, hi: f64) -> ShannonEntropy<f64, Echo<f64>> {
        ShannonEntropy::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            NonZeroUsize::new(bins).unwrap(),
            lo,
            hi,
        )
    }

    #[test]
    fn shannon_entropy_warmup() {
        let mut e = new_f64(5, 4, 0.0, 1.0);
        for i in 0..4 {
            e.update(i as f64 * 0.2);
            assert_eq!(e.last(), None);
        }
        e.update(0.9);
        assert!(e.last().is_some());
    }

    #[test]
    fn shannon_entropy_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(42);
        let (window_len, bins, lo, hi) = (50, 7, -1.0, 1.0);
        let mut e = new_f64(window_len, bins, lo, hi);
        let mut vals = Vec::new();
        for i in 0..3_000 {
            // Some values fall outside [lo, hi] to exercise clamping.
            let v = rng.f64() * 2.6 - 1.3;
            vals.push(v);
            e.update(v);
            if i + 1 < window_len {
                assert!(e.last().is_none());
                continue;
            }
            let expected = naive(&vals[vals.len() - window_len..], bins, lo, hi);
            let got = e.last().unwrap();
            assert!((got - expected).abs() < 1e-9, "{got} vs {expected}");
        }
    }

    #[test]
    fn shannon_entropy_constant_is_zero() {
        let mut e = new_f64(20, 10, 0.0, 10.0);
        for _ in 0..100 {
            e.update(3.3);
        }
        assert_eq!(e.last().unwrap(), 0.0);
    }

    #[test]
    fn shannon_entropy_uniform_noise_near_max() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let mut e = new_f64(10_000, 10, 0.0, 1.0);
        for _ in 0..20_000 {
            e.update(rng.f64());
        }
        let h = e.last().unwrap();
        assert!(h > 0.99, "{h}");
        assert!(h <= 1.0, "{h}");
    }

    #[test]
    fn shannon_entropy_upper_edge_in_last_bin() {
        assert_eq!(bin_index(1.0, 0.0, 1.0, 4), 3);
        assert_eq!(bin_index(5.0, 0.0, 1.0, 4), 3);
        assert_eq!(bin_index(-5.0, 0.0, 1.0, 4), 0);
        assert_eq!(bin_index(0.0, 0.0, 1.0, 4), 0);
        assert_eq!(bin_index(0.25, 0.0, 1.0, 4), 1);
    }

    #[test]
    fn shannon_entropy_f32_stays_close_to_f64() {
        let rng = romu::Rng::from_seed_with_64bit(3);
        let w = NonZeroUsize::new(500).unwrap();
        let b = NonZeroUsize::new(16).unwrap();
        let mut e32 = ShannonEntropy::<f32, _>::new(Echo::new(), w, b, 0.0, 1.0);
        let mut e64 = ShannonEntropy::<f64, _>::new(Echo::new(), w, b, 0.0, 1.0);
        for _ in 0..100_000 {
            // Squared values give a skewed distribution with non-trivial entropy.
            let v = rng.f64().powi(2);
            e32.update(v as f32);
            e64.update(v);
        }
        let diff = (e32.last().unwrap() as f64 - e64.last().unwrap()).abs();
        assert!(diff < 1e-3, "{diff}");
    }

    #[test]
    #[should_panic(expected = "bins must be at least 2")]
    fn shannon_entropy_one_bin() {
        let _ = new_f64(10, 1, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "lo must be smaller than hi")]
    fn shannon_entropy_empty_range() {
        let _ = new_f64(10, 4, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "lo must be finite")]
    fn shannon_entropy_infinite_lo() {
        let _ = new_f64(10, 4, f64::NEG_INFINITY, 1.0);
    }

    #[test]
    fn shannon_entropy_plot() {
        let lo = TEST_DATA.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = TEST_DATA.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut e = new_f64(32, 8, lo, hi);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            e.update(*v);
            if let Some(val) = e.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/shannon_entropy.png").unwrap();
    }
}

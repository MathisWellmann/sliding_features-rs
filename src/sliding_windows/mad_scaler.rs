//! Robust MAD-based (mean absolute difference) scaler over a sliding window.
//!
//! Computes `(x - median) / (1.4826 * MAD)` where `median` and `MAD`
//! (median absolute deviation) are derived from the *previous* sliding
//! window.  The constant 1.4826 makes the estimator consistent with the
//! standard deviation for normally distributed data.
//!
//! Unlike `ZScoreStandardization`, the MAD scaler is resistant to outliers
//! and works well for skewed distributions.

use std::{
    num::NonZeroUsize,
    ops::{
        AddAssign,
        SubAssign,
    },
};

use num::{
    Float,
    FromPrimitive,
};
use watermill::sorted_window::SortedWindow;

use crate::View;

/// Robust MAD-based scaler over a sliding window.
///
/// Computes `(x - median) / (1.4826 * MAD)` using only *past* values so
/// that the transformation does not leak information about the current
/// observation.
///
/// No output is emitted until the sliding window is fully filled.
/// During warm‑up `last()` returns `None`.
pub struct MadScaler<T: Float + FromPrimitive + AddAssign + SubAssign, V> {
    view: V,
    /// Sliding window length (for warm-up tracking).
    window_len: NonZeroUsize,
    /// Sliding window storing the sorted values.
    sorted: SortedWindow<T>,
    /// Cached median of the window (recomputed when the sliding median changes).
    cached_median: T,
    /// Cached MAD (recomputed when the sliding median changes).
    cached_mad: T,
    /// Whether the caches are stale and need recomputation.
    cache_stale: bool,
    /// Most recent output of the scaler.
    out: Option<T>,
    /// Running count of values pushed into the sorted window (capped at window_len).
    count: usize,
}

/// MAD consistency constant: ~1.4826 makes MAD ≈ σ for normally distributed data.
const MAD_SCALE: f64 = 1.4826;

impl<T, V> MadScaler<T, V>
where
    V: View<T>,
    T: Float + FromPrimitive + AddAssign + SubAssign,
{
    /// Create a new `MadScaler` with a chained `View` and a given sliding
    /// window length.
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            sorted: SortedWindow::new(window_len.get()),
            cached_median: T::zero(),
            cached_mad: T::zero(),
            cache_stale: true,
            out: None,
            count: 0,
        }
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// Recompute the cached median and MAD from the sorted window.
    fn recompute_cache(&mut self) {
        let n = self.sorted.len();
        if n == 0 {
            self.cached_median = T::zero();
            self.cached_mad = T::zero();
            self.cache_stale = false;
            return;
        }

        // --- median ---
        let median = if n % 2 == 0 {
            let a = self.sorted[n / 2 - 1];
            let b = self.sorted[n / 2];
            (a + b) / (T::one() + T::one())
        } else {
            self.sorted[n / 2]
        };
        self.cached_median = median;

        // --- median absolute deviation ---
        // Collect absolute deviations, sort them.
        let mut abs_devs: Vec<T> = Vec::with_capacity(n);
        for i in 0..n {
            let diff = self.sorted[i] - median;
            abs_devs.push(if diff < T::zero() { -diff } else { diff });
        }
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mad = if n % 2 == 0 {
            (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / (T::one() + T::one())
        } else {
            abs_devs[n / 2]
        };
        self.cached_mad = mad;

        self.cache_stale = false;
    }
}

impl<T, V> View<T> for MadScaler<T, V>
where
    V: View<T>,
    T: Float + FromPrimitive + AddAssign + SubAssign,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else {
            return;
        };
        debug_assert!(val.is_finite(), "value must be finite");

        // Only emit output once the window is full (count >= window_len).
        // At that point `sorted` contains exactly `window_len` past values.
        if self.count >= self.window_len.get() {
            // The window (`sorted`) holds the *previous* window of values.
            // Normalize `val` against those.
            if self.cache_stale {
                self.recompute_cache();
            }

            self.out = {
                let scale_const = T::from(MAD_SCALE).expect("convert");
                if self.cached_mad <= T::zero() {
                    // No dispersion → all values identical.
                    Some(T::zero())
                } else {
                    let diff = val - self.cached_median;
                    Some(diff / (scale_const * self.cached_mad))
                }
            };
        } else {
            self.out = None;
        }

        // Slide window: push current value into the sorted window.
        // This makes it part of the *next* normalization's reference.
        let prev_len = self.sorted.len();
        self.sorted.push_back(val);
        // Only mark stale if the median may have changed.
        // SortedWindow insertion/removal may change the median, so always invalidate.
        if self.sorted.len() != prev_len {
            // A value was evicted (full window) → cached median/MAD always stale.
            self.cache_stale = true;
        } else {
            // A value was pushed without eviction (warm-up) → cache is stale too.
            self.cache_stale = true;
        }

        self.count = (self.count + 1).min(self.window_len.get());
    }

    #[inline]
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

    #[test]
    fn mad_scaler_plot() {
        let mut ms = MadScaler::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            ms.update(*v);
            if let Some(val) = ms.last() {
                out.push(val);
            }
        }
        let filename = "img/mad_scaler.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn mad_scaler_no_lookahead() {
        // Window = 3. Warm up with [10, 10, 10], then spike 100.
        // The spike must be normalized against [10, 10, 10],
        // where median = 10, MAD = 0 → output 0.
        let mut ms = MadScaler::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        ms.update(10.0);
        assert!(ms.last().is_none(), "warm-up 1");
        ms.update(10.0);
        assert!(ms.last().is_none(), "warm-up 2");
        ms.update(10.0);
        assert!(ms.last().is_none(), "warm-up 3");

        // 4th value: first real output. Window = [10, 10, 10].
        ms.update(100.0);
        let got = ms.last().unwrap();
        assert!(
            (got - 0.0).abs() < 1e-12,
            "spike must not leak into its own reference window, got {got}, expected 0.0"
        );

        // Now window = [10, 10, 100]. Next 10 is normalized against it.
        // median of [10, 10, 100] = 10.
        // absolute deviations: [0, 0, 90] → sorted [0, 0, 90] → median MAD = 0.
        // → output = 0.
        ms.update(10.0);
        let got = ms.last().unwrap();
        assert!(
            (got - 0.0).abs() < 1e-12,
            "after spike entered window (MAD=0), got {got}, expected 0.0"
        );
    }

    #[test]
    fn mad_scaler_disjoint_clean_and_outlier() {
        // Values: 5 clean (1,2,3,4,5) then outlier (100).
        // Window = 5.
        let mut ms = MadScaler::new(Echo::new(), NonZeroUsize::new(5).unwrap());

        // Warm-up: no output for first 5.
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            ms.update(v);
            assert!(ms.last().is_none(), "warm-up");
        }

        // 6th: 100 normalized against [1,2,3,4,5].
        // median of [1,2,3,4,5] = 3.
        // abs devs: [2,1,0,1,2] → sorted [0,1,1,2,2] → MAD = 1.
        // → output = (100 - 3) / (1.4826 * 1) ≈ 65.42
        ms.update(100.0);
        let got = ms.last().unwrap();
        let expected = (100.0 - 3.0) / (1.4826);
        assert!(
            (got - expected).abs() < 1e-6,
            "outlier scaling: got {got}, expected ~{expected}"
        );
    }
}

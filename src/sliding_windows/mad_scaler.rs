//! Robust MAD-based (median absolute deviation) scaler over a sliding window.
//!
//! Computes `(x - median) / (1.4826 * MAD)` where `median` and `MAD`
//! (median absolute deviation) are derived from the *previous* sliding
//! window.
//! `MAD = median((x - median(X)).abs())`
//! The constant 1.4826 makes the estimator consistent with the
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
    /// Most recent output of the scaler.
    out: Option<T>,
    /// Running count of values pushed into the sorted window (capped at window_len).
    count: usize,
    /// Pre-allocated buffer for merging absolute deviations in the hot-path.
    mad_buf: Vec<T>,
}

impl<F, V> std::fmt::Debug for MadScaler<F, V>
where
    F: Float + FromPrimitive + AddAssign + SubAssign + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MadScaler")
            .field("window_len", &self.window_len)
            .field("cached_median", &self.cached_median)
            .field("cached_mad", &self.cached_mad)
            .field("out", &self.out)
            .field("count", &self.count)
            .finish()
    }
}

/// MAD consistency constant: ~1.4826 makes MAD ≈ σ for normally distributed data.
const MAD_SCALE: f64 = 1.4826;

impl<F, V> MadScaler<F, V>
where
    V: View<F>,
    F: Float + FromPrimitive + AddAssign + SubAssign,
{
    /// Create a new `MadScaler` with a chained `View` and a given sliding
    /// window length.
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            sorted: SortedWindow::new(window_len.get()),
            cached_median: F::zero(),
            cached_mad: F::zero(),
            out: None,
            count: 0,
            mad_buf: vec![F::zero(); window_len.get()],
        }
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// Recompute median and MAD from the sorted window.
    ///
    /// Because `self.sorted` is already sorted, the absolute deviations
    /// `|s[i] - median|` form two monotonic sequences radiating outward
    /// from the median: one increasing to the left, one to the right.
    /// We merge these two "tails" in **O(w)** instead of paying the
    /// **O(w log w)** cost of a full sort.
    fn recompute_cache(&mut self) {
        let n = self.sorted.len();
        debug_assert!(n > 0, "recompute_cache called on empty window");
        debug_assert_eq!(n, self.mad_buf.len());

        // --- median (from already-sorted window) ---
        let median = if n % 2 == 0 {
            let a = self.sorted[n / 2 - 1];
            let b = self.sorted[n / 2];
            (a + b) / (F::one() + F::one())
        } else {
            self.sorted[n / 2]
        };
        self.cached_median = median;

        // --- median absolute deviation via O(w) merge ---
        // Walk the sorted window outward from the median, merging the
        // monotonically-increasing left and right absolute deviations.
        let mad_buf = &mut self.mad_buf;
        let mid = n / 2;
        let mut out_idx = 0usize;

        // For odd n, the median element itself has deviation 0 — emit it first.
        if n % 2 == 1 {
            mad_buf[0] = F::zero();
            out_idx = 1;
        }

        // left walks down from mid-1 to 0, right walks up from mid (even) or mid+1 (odd).
        let mut left: isize = mid.checked_sub(1).map_or(-1, |v| v as isize);
        let mut right: usize = if n % 2 == 0 { mid } else { mid + 1 };

        loop {
            let l_valid = left >= 0;
            let r_valid = right < n;
            match (l_valid, r_valid) {
                (true, true) => {
                    let l_abs = (self.sorted[left as usize] - median).abs();
                    let r_abs = (self.sorted[right] - median).abs();
                    if l_abs <= r_abs {
                        mad_buf[out_idx] = l_abs;
                        out_idx += 1;
                        left -= 1;
                    } else {
                        mad_buf[out_idx] = r_abs;
                        out_idx += 1;
                        right += 1;
                    }
                }
                (true, false) => {
                    mad_buf[out_idx] = (self.sorted[left as usize] - median).abs();
                    out_idx += 1;
                    left -= 1;
                }
                (false, true) => {
                    mad_buf[out_idx] = (self.sorted[right] - median).abs();
                    out_idx += 1;
                    right += 1;
                }
                (false, false) => break,
            }
        }
        debug_assert_eq!(out_idx, n, "merge must fill entire buffer");

        self.cached_mad = median_from_sorted(mad_buf);
    }
}

#[inline(always)]
fn median_from_sorted<F: Float>(vals: &[F]) -> F {
    let n = vals.len();
    if n % 2 == 0 {
        let a = vals[n / 2 - 1];
        let b = vals[n / 2];
        (a + b) / (F::one() + F::one())
    } else {
        vals[n / 2]
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
            self.recompute_cache();

            let scale_const = T::from(MAD_SCALE).expect("convert");
            self.out = if self.cached_mad <= T::zero() {
                // No dispersion → all values identical.
                Some(T::zero())
            } else {
                let diff = val - self.cached_median;
                Some(diff / (scale_const * self.cached_mad))
            };
        } else {
            self.out = None;
        }

        // Slide window: push current value into the sorted window.
        // This makes it part of the *next* normalization's reference.
        self.sorted.push_back(val);

        self.count = (self.count + 1).min(self.window_len.get());
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use ballpark::assert_approx_eq;

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

    fn median(vals: &mut [f64]) -> f64 {
        vals.sort_by(f64::total_cmp);
        let n = vals.len();
        if n.is_multiple_of(2) {
            let a = vals[n / 2 - 1];
            let b = vals[n / 2];
            (a + b) / 2.0
        } else {
            vals[n / 2]
        }
    }

    fn mad_scaled(vals: &mut [f64], buf: &mut [f64], current: f64) -> f64 {
        assert_eq!(vals.len(), buf.len());
        assert_ne!(*vals.last().unwrap(), current);

        let m = median(vals);
        dbg!(&m);
        buf.iter_mut()
            .zip(vals)
            .for_each(|(b, v)| *b = (*v - m).abs());
        let mad = median(buf);
        dbg!(&mad);
        (current - m) / (MAD_SCALE * mad)
    }

    #[test]
    fn mad_scaler() {
        const WINDOW_LEN: usize = 5;

        let r = romu::Rng::from_seed_with_64bit(0);
        let vals = Vec::from_iter((0..25).map(|_| r.f64()));
        dbg!(&vals);
        let mut buf = vec![0.0; WINDOW_LEN];

        let mut ms = MadScaler::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());

        // warmup
        for v in vals.iter().take(WINDOW_LEN) {
            ms.update(*v);
            assert!(ms.last().is_none())
        }

        for (i, v) in vals.iter().enumerate().skip(WINDOW_LEN) {
            ms.update(*v);
            dbg!(&v);
            dbg!(&ms);
            let start = i - WINDOW_LEN;
            let end = i;
            let mut window = vals[start..end].to_vec();
            assert_eq!(window.len(), WINDOW_LEN);
            let expected = mad_scaled(&mut window, &mut buf, *v);
            assert_approx_eq!(ms.last().expect("is warm"), expected);
        }
    }
}

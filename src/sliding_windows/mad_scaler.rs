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
    collections::VecDeque,
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

use crate::View;

/// Robust MAD-based scaler over a sliding window.
///
/// Computes `(x - median) / (1.4826 * MAD)` using only *past* values so
/// that the transformation does not leak information about the current
/// observation.
///
/// No output is emitted until the sliding window is fully filled.
/// During warm‑up `last()` returns `None`.
///
/// An update costs `O(log w)` comparisons plus moving at most `w / 2`
/// values of the sorted window (none for monotone input such as
/// timestamps).
pub struct MadScaler<T: Float + FromPrimitive + AddAssign + SubAssign, V> {
    view: V,
    /// Sliding window length (for warm-up tracking).
    window_len: NonZeroUsize,
    /// The window's values in ascending order.
    ///
    /// A ring buffer rather than a `Vec`: `insert` and `remove` move only
    /// the shorter side, so monotone input (always evicting the smallest
    /// value and inserting the largest) slides without moving anything, and
    /// any other input moves at most half the window.
    sorted: VecDeque<T>,
    /// FIFO insertion order for O(1) eviction decisions.
    order: VecDeque<T>,
    /// Median of the window, recomputed on every update.
    median: T,
    /// Median absolute deviation of the window, recomputed on every update.
    mad: T,
    /// Most recent output of the scaler.
    out: Option<T>,
    /// Running count of values pushed into the sorted window (capped at window_len).
    count: usize,
}

impl<F, V> std::fmt::Debug for MadScaler<F, V>
where
    F: Float + FromPrimitive + AddAssign + SubAssign + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MadScaler")
            .field("window_len", &self.window_len)
            .field("median", &self.median)
            .field("mad", &self.mad)
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
        let w = window_len.get();
        Self {
            view,
            window_len,
            sorted: VecDeque::with_capacity(w),
            order: VecDeque::with_capacity(w),
            median: F::zero(),
            mad: F::zero(),
            out: None,
            count: 0,
        }
    }

    /// Slide the window: evict the oldest value (if full), insert `val` in sorted position.
    #[inline]
    fn slide_window(&mut self, val: F) {
        let is_less = |p: &F, v: &F| p.partial_cmp(v).expect("NaN") == std::cmp::Ordering::Less;
        if self.sorted.len() >= self.window_len.get() {
            let oldest = self.order.pop_front().unwrap();
            // partition_point requires monotonic predicate; sorted[len..] >= oldest after this.
            let pos = self.sorted.partition_point(|p| is_less(p, &oldest));
            debug_assert_eq!(
                self.sorted[pos].partial_cmp(&oldest),
                Some(std::cmp::Ordering::Equal),
                "oldest value missing from sorted window"
            );
            self.sorted.remove(pos);
        }
        self.order.push_back(val);
        let pos = self.sorted.partition_point(|p| is_less(p, &val));
        self.sorted.insert(pos, val);
        debug_assert_eq!(self.sorted.len(), self.order.len());
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// Recompute the median and the MAD of the sorted window in `O(log w)`.
    ///
    /// The absolute deviations from the median form two non-decreasing
    /// sequences radiating outward from the middle of the sorted window `s`:
    /// `left(i) = |s[mid - 1 - i] - median|` and
    /// `right(j) = |s[mid + j] - median|`, with `mid = n / 2`. (Rounding is
    /// monotone, so the computed deviations are non-decreasing too.) For odd
    /// `n`, `right(0)` is the median's own deviation, 0.
    ///
    /// The MAD is the median of their union. With `c = n / 2` that is the
    /// `c`-th smallest deviation (0-based) for odd `n`, and the mean of the
    /// `(c - 1)`-th and `c`-th for even `n`. Both sit at the boundary of the
    /// split that takes the `c` smallest deviations, `i` from `left` and
    /// `c - i` from `right`. A binary search over `i` finds it while
    /// computing only the ~`2 log2(n)` deviations it compares, instead of
    /// merging all `n` of them.
    ///
    /// Every deviation is computed with the same expression as a full merge
    /// would, and an order statistic does not depend on how ties are
    /// ordered, so the result is bit-identical to merging (or sorting) all
    /// deviations.
    fn recompute_median_and_mad(&mut self) {
        let s = &self.sorted;
        let n = s.len();
        debug_assert!(n > 0, "recompute_median_and_mad called on empty window");
        let two = F::one() + F::one();
        let mid = n / 2;
        let median = if n.is_multiple_of(2) {
            (s[mid - 1] + s[mid]) / two
        } else {
            s[mid]
        };
        debug_assert!(s[0] <= median, "median below the window minimum");
        debug_assert!(median <= s[n - 1], "median above the window maximum");
        let left = |i: usize| (s[mid - 1 - i] - median).abs();
        let right = |j: usize| (s[mid + j] - median).abs();
        let (n_left, n_right) = (mid, n - mid);

        // `left` has exactly `c` deviations and `right` at least `c`, so `i`
        // ranges over `0..=c`. The split is the smallest `i` at which the
        // next `left` deviation is no smaller than the last `right` one
        // taken; the predicate is monotone in `i` because `left(i)` grows
        // and `right(c - 1 - i)` shrinks as `i` grows.
        let c = mid;
        let (mut lo, mut hi) = (0, c);
        while lo < hi {
            let i = lo + (hi - lo) / 2;
            if left(i) < right(c - 1 - i) {
                lo = i + 1;
            } else {
                hi = i;
            }
        }
        let (i, j) = (lo, c - lo);
        debug_assert!(i <= n_left);
        debug_assert!(j <= n_right);

        // The `c`-th smallest deviation: the smaller next candidate.
        let upper = match (i < n_left, j < n_right) {
            (true, true) => left(i).min(right(j)),
            (true, false) => left(i),
            (false, true) => right(j),
            (false, false) => unreachable!("c < n leaves a deviation untaken"),
        };
        let mad = if n.is_multiple_of(2) {
            // The `(c - 1)`-th smallest: the larger last-taken deviation.
            let lower = match (i > 0, j > 0) {
                (true, true) => left(i - 1).max(right(j - 1)),
                (true, false) => left(i - 1),
                (false, true) => right(j - 1),
                (false, false) => unreachable!("even n has c >= 1 taken deviations"),
            };
            debug_assert!(lower <= upper, "the split must be ordered");
            (lower + upper) / two
        } else {
            upper
        };
        debug_assert!(mad >= F::zero(), "a MAD is never negative");

        self.median = median;
        self.mad = mad;
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
            self.recompute_median_and_mad();

            let scale_const = T::from(MAD_SCALE).expect("convert");
            self.out = if self.mad <= T::zero() {
                // No dispersion → all values identical.
                Some(T::zero())
            } else {
                let diff = val - self.median;
                Some(diff / (scale_const * self.mad))
            };
        } else {
            self.out = None;
        }

        // Slide window: push current value into the sorted window.
        // This makes it part of the *next* normalization's reference.
        self.slide_window(val);

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

    /// Median of an already sorted slice, written independently of the
    /// implementation's helper but with the same floating-point expression.
    fn reference_median<F: Float>(sorted: &[F]) -> F {
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / (F::one() + F::one())
        } else {
            sorted[n / 2]
        }
    }

    /// Oracle: scale `current` against `window` by sorting the window and
    /// its absolute deviations from scratch. Every value goes through the
    /// same floating-point expressions as the scaler and an order
    /// statistic does not depend on how ties are ordered, so a correct
    /// scaler matches this bit for bit.
    fn reference_scaled<F: Float + FromPrimitive>(window: &[F], current: F) -> F {
        let by_value = |a: &F, b: &F| a.partial_cmp(b).expect("finite test data");
        let mut sorted = window.to_vec();
        sorted.sort_by(by_value);
        let median = reference_median(&sorted);
        let mut deviations = Vec::from_iter(sorted.iter().map(|v| (*v - median).abs()));
        deviations.sort_by(by_value);
        let mad = reference_median(&deviations);
        if mad <= F::zero() {
            F::zero()
        } else {
            (current - median) / (F::from(MAD_SCALE).expect("convert") * mad)
        }
    }

    /// Feed `vals` through a `MadScaler` of `window_len` and require the
    /// exact oracle value (same sign, exponent and mantissa) after warm-up,
    /// and `None` during it.
    fn assert_matches_reference<F>(name: &str, vals: &[F], window_len: usize)
    where
        F: Float + FromPrimitive + AddAssign + SubAssign + std::fmt::Debug,
    {
        let mut ms = MadScaler::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        for (i, v) in vals.iter().enumerate() {
            ms.update(*v);
            if i < window_len {
                assert!(ms.last().is_none(), "{name} w={window_len}: warm-up at {i}");
                continue;
            }
            let got = ms.last().expect("warm after window_len values");
            let expected = reference_scaled(&vals[i - window_len..i], *v);
            assert_eq!(
                got.integer_decode(),
                expected.integer_decode(),
                "{name} w={window_len} i={i}: got {got:?}, expected {expected:?}"
            );
        }
    }

    /// Streams that stress ties, a zero MAD, monotone input (where the
    /// sorted window equals insertion order) and outliers.
    fn reference_streams(seed: u64, len: usize) -> Vec<(&'static str, Vec<f64>)> {
        let rng = romu::Rng::from_seed_with_64bit(seed);
        let mut price = 100.0_f64;
        let walk = Vec::from_iter((0..len).map(|_| {
            price *= 1.0 + 0.002 * (rng.f64() - 0.5);
            price
        }));
        let ticks = Vec::from_iter(walk.iter().map(|p| (p * 10.0).round() / 10.0));
        let small_ints = Vec::from_iter((0..len).map(|_| rng.mod_u64(4) as f64));
        let constant = vec![7.25; len];
        let mut t = 1_516_320_000.0_f64;
        let timestamps = Vec::from_iter((0..len).map(|_| {
            t += rng.mod_u64(3) as f64 * 0.001;
            t
        }));
        let descending = Vec::from_iter((0..len).map(|i| -(i as f64) * 0.5));
        let outliers = Vec::from_iter((0..len).map(|_| {
            let v = rng.f64() * 2.0 - 1.0;
            if rng.mod_u64(50) == 0 { v * 1e6 } else { v }
        }));
        vec![
            ("walk", walk),
            ("ticks", ticks),
            ("small_ints", small_ints),
            ("constant", constant),
            ("timestamps", timestamps),
            ("descending", descending),
            ("outliers", outliers),
        ]
    }

    const REFERENCE_WINDOWS: &[usize] =
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 31, 32, 63, 64, 255, 256];

    #[test]
    fn mad_scaler_matches_reference_f64() {
        for &window_len in REFERENCE_WINDOWS {
            for seed in 0..3 {
                for (name, vals) in reference_streams(seed, 4 * window_len + 64) {
                    assert_matches_reference(name, &vals, window_len);
                }
            }
        }
    }

    #[test]
    fn mad_scaler_matches_reference_f32() {
        for &window_len in REFERENCE_WINDOWS {
            for seed in 0..3 {
                for (name, vals) in reference_streams(seed, 4 * window_len + 64) {
                    let vals = Vec::from_iter(vals.iter().map(|v| *v as f32));
                    assert_matches_reference(name, &vals, window_len);
                }
            }
        }
    }
}

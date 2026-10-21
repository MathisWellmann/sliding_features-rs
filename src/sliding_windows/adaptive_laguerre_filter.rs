// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Number of normalized tracking errors in the median that yields `alpha`, 5 in the book.
const MEDIAN_LEN: usize = 5;

/// John Ehlers Adaptive Laguerre Filter.
///
/// Reference: John Ehlers, "Cybernetic Analysis for Stocks and Futures" (2004),
/// Laguerre filter chapter ("Time Warp - Without Space Travel"), "Adaptive Laguerre Filter".
/// The fixed Laguerre filter alone is in <https://www.mesasoftware.com/papers/TimeWarp.pdf>,
/// the adaptive variant only in the book. The damping of the
/// [`LaguerreFilter`](super::LaguerreFilter) adapts to how well the filter tracks the data:
///
/// ```text
/// Diff  = |Price - Filt[1]|
/// HH, LL = highest and lowest Diff over the last `length` bars (20 in the book)
/// alpha = Median((Diff - LL) / (HH - LL), 5)      if HH - LL <> 0, else unchanged
/// L0 = alpha Price + (1 - alpha) L0[1]
/// L1 = -(1 - alpha) L0 + L0[1] + (1 - alpha) L1[1]
/// L2 = -(1 - alpha) L1 + L1[1] + (1 - alpha) L2[1]
/// L3 = -(1 - alpha) L2 + L2[1] + (1 - alpha) L3[1]
/// Filt = (L0 + 2 L1 + 2 L2 + L3) / 6
/// ```
///
/// A large tracking error (trend) drives alpha towards 1, so the filter follows the
/// data closely, a small one (range) drives alpha towards 0 for heavy smoothing.
///
/// Deviations: the Laguerre elements start at the first value instead of 0, which
/// avoids a long transient from zero; `alpha` is 0 (the book's variable initial value)
/// until the first median is available, so the filter holds the first value for
/// `length + 4` bars. The median is taken over the last 5 normalized values that
/// were computed, i.e. bars with `HH = LL` are skipped instead of evaluated.
/// `last()` is `None` for the first `length + 4` values. Constant input is passed
/// through unchanged (alpha keeps its value).
///
/// Output range: follows the input; for `alpha` in `[0, 1]`.
/// Complexity: O(length) per update for the highest and lowest `Diff`, O(length) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct AdaptiveLaguerreFilter<T, V> {
    view: V,
    /// The number of bars over which the tracking error is normalized.
    #[getset(get_copy = "pub")]
    length: NonZeroUsize,
    /// Ring buffer of the last `length` tracking errors.
    diffs: Vec<T>,
    /// Ring buffer of the last normalized tracking errors.
    ratios: [T; MEDIAN_LEN],
    n_ratios: usize,
    alpha: T,
    /// Laguerre elements L0..L3.
    l: [T; 4],
    filt: T,
    /// Number of values received from the inner view.
    n: usize,
}

impl<T, V> AdaptiveLaguerreFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance normalizing the tracking error over `length` bars (20 in the book).
    ///
    /// # Panics
    /// If `length < 2`, as the highest and lowest of a single value are always equal.
    pub fn new(view: V, length: NonZeroUsize) -> Self {
        assert!(length.get() >= 2, "length must be >= 2");
        Self {
            view,
            length,
            diffs: vec![T::zero(); length.get()],
            ratios: [T::zero(); MEDIAN_LEN],
            n_ratios: 0,
            alpha: T::zero(),
            l: [T::zero(); 4],
            filt: T::zero(),
            n: 0,
        }
    }

    /// The current smoothing factor `alpha` in `[0, 1]`, with Laguerre `gamma = 1 - alpha`.
    /// `None` while warming up.
    pub fn alpha(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.alpha >= T::zero(), "alpha must be >= 0");
        debug_assert!(self.alpha <= T::one(), "alpha must be <= 1");
        Some(self.alpha)
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.n >= self.length.get() + MEDIAN_LEN - 1
    }
}

impl<T, V> View<T> for AdaptiveLaguerreFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.n == 0 {
            // Start at rest on the first value, see struct docs.
            self.l = [val; 4];
            self.filt = val;
        }
        let len = self.length.get();
        let diff = (val - self.filt).abs();
        self.diffs[self.n % len] = diff;
        self.n = self.n.saturating_add(1);
        if self.n > len {
            let (hh, ll) = highest_lowest(&self.diffs);
            if hh > ll {
                self.ratios[self.n_ratios % MEDIAN_LEN] = (diff - ll) / (hh - ll);
                self.n_ratios += 1;
            }
            if self.n_ratios >= MEDIAN_LEN {
                self.alpha = median(self.ratios);
            }
        }
        self.l = laguerre(self.l, val, T::one() - self.alpha);
        let two = T::from(2.0).expect("can convert");
        self.filt = (self.l[0] + two * self.l[1] + two * self.l[2] + self.l[3])
            / T::from(6.0).expect("can convert");
        debug_assert!(self.filt.is_finite(), "value must be finite");
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.is_ready() {
            return None;
        }
        debug_assert!(self.filt.is_finite(), "value must be finite");
        Some(self.filt)
    }
}

/// Highest and lowest value of `vals`.
fn highest_lowest<T: Float>(vals: &[T]) -> (T, T) {
    debug_assert!(!vals.is_empty(), "values must not be empty");
    vals.iter()
        .fold((vals[0], vals[0]), |(hh, ll), &v| (hh.max(v), ll.min(v)))
}

/// Median of an odd number of finite values.
fn median<T: Float>(mut vals: [T; MEDIAN_LEN]) -> T {
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).expect("values are finite"));
    vals[MEDIAN_LEN / 2]
}

/// One step of the four element Laguerre filter with damping `gamma`.
fn laguerre<T: Float>(l: [T; 4], val: T, gamma: T) -> [T; 4] {
    debug_assert!(gamma >= T::zero(), "gamma must be >= 0");
    debug_assert!(gamma <= T::one(), "gamma must be <= 1");
    let l0 = (T::one() - gamma) * val + gamma * l[0];
    let l1 = -gamma * l0 + l[0] + gamma * l[1];
    let l2 = -gamma * l1 + l[1] + gamma * l[2];
    let l3 = -gamma * l2 + l[2] + gamma * l[3];
    [l0, l1, l2, l3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn alf(length: usize) -> AdaptiveLaguerreFilter<f64, Echo<f64>> {
        AdaptiveLaguerreFilter::new(Echo::new(), NonZeroUsize::new(length).unwrap())
    }

    #[test]
    fn adaptive_laguerre_filter_plot() {
        let mut f = alf(20);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/adaptive_laguerre_filter.png").unwrap();
    }

    #[test]
    fn adaptive_laguerre_filter_warmup() {
        let mut f = alf(20);
        assert_eq!(f.length().get(), 20);
        let rng = romu::Rng::from_seed_with_64bit(1);
        for _ in 0..23 {
            f.update(rng.f64());
            assert_eq!(f.last(), None);
            assert_eq!(f.alpha(), None);
        }
        f.update(rng.f64());
        assert!(f.last().is_some());
        assert!(f.alpha().is_some());
    }

    #[test]
    fn adaptive_laguerre_filter_alpha_in_range() {
        let mut f = alf(20);
        let rng = romu::Rng::from_seed_with_64bit(2);
        let mut price = 100.0;
        for _ in 0..5_000 {
            price += rng.f64() - 0.5;
            f.update(price);
            if let Some(a) = f.alpha() {
                assert!((0.0..=1.0).contains(&a), "alpha {a}");
                assert!(f.last().unwrap().is_finite());
            }
        }
    }

    #[test]
    fn adaptive_laguerre_filter_constant_input() {
        let mut f = alf(20);
        for _ in 0..100 {
            f.update(42.0);
        }
        assert_eq!(f.last(), Some(42.0));
        assert_eq!(f.alpha(), Some(0.0));
    }

    /// A jump in a quiet market is a large tracking error, so alpha rises and the
    /// filter catches up within a few bars.
    #[test]
    fn adaptive_laguerre_filter_adapts_to_step() {
        let mut f = alf(20);
        let rng = romu::Rng::from_seed_with_64bit(4);
        for _ in 0..100 {
            f.update(100.0 + 0.1 * rng.f64());
        }
        let quiet_alpha = f.alpha().unwrap();
        let mut max_alpha: f64 = 0.0;
        for _ in 0..5 {
            f.update(110.0 + 0.1 * rng.f64());
            max_alpha = max_alpha.max(f.alpha().unwrap());
        }
        assert!(max_alpha > 0.8, "alpha {max_alpha}");
        assert!(max_alpha > quiet_alpha);
        for _ in 0..5 {
            f.update(110.0 + 0.1 * rng.f64());
        }
        assert!(
            (f.last().unwrap() - 110.0).abs() < 1.0,
            "filter {}",
            f.last().unwrap()
        );
    }

    /// Direct transcription of the book code with the same initialization.
    #[test]
    fn adaptive_laguerre_filter_matches_book_transcription() {
        let length = 20;
        let mut f = alf(length);
        let rng = romu::Rng::from_seed_with_64bit(5);
        let (mut diffs, mut ratios) = (Vec::new(), Vec::new());
        let (mut alpha, mut l, mut filt) = (0.0, [0.0; 4], 0.0);
        let mut price = 100.0;
        for bar in 1..=1_000 {
            price += rng.f64() - 0.5;
            f.update(price);
            if bar == 1 {
                l = [price; 4];
                filt = price;
            }
            let diff = f64::abs(price - filt);
            diffs.insert(0, diff);
            let (mut hh, mut ll) = (diff, diff);
            for d in diffs.iter().take(length) {
                hh = hh.max(*d);
                ll = ll.min(*d);
            }
            if bar > length && hh - ll != 0.0 {
                ratios.insert(0, (diff - ll) / (hh - ll));
            }
            if ratios.len() >= 5 {
                let mut m: Vec<f64> = ratios[..5].to_vec();
                m.sort_by(|a, b| a.partial_cmp(b).unwrap());
                alpha = m[2];
            }
            let l0 = alpha * price + (1.0 - alpha) * l[0];
            let l1 = -(1.0 - alpha) * l0 + l[0] + (1.0 - alpha) * l[1];
            let l2 = -(1.0 - alpha) * l1 + l[1] + (1.0 - alpha) * l[2];
            let l3 = -(1.0 - alpha) * l2 + l[2] + (1.0 - alpha) * l[3];
            l = [l0, l1, l2, l3];
            filt = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0;
            if bar >= length + 4 {
                assert!((f.last().unwrap() - filt).abs() < 1e-9, "bar {bar}");
                // Rounding differs slightly as the crate computes `1 - gamma`.
                assert!((f.alpha().unwrap() - alpha).abs() < 1e-9, "bar {bar}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "length must be >= 2")]
    fn adaptive_laguerre_filter_length_too_small() {
        let _ = alf(1);
    }
}

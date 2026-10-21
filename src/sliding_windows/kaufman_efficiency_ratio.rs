// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Kaufman Efficiency Ratio

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Kaufman Efficiency Ratio (Perry J. Kaufman, "Smarter Trading", 1995; the
/// efficiency ratio inside Kaufman's Adaptive Moving Average).
///
/// `ER = |x - x[n]| / sum(|x[i] - x[i-1]|)`, the straight-line distance
/// travelled over `window_len` steps divided by the total distance actually
/// travelled (the sum of the `window_len` absolute step-to-step changes). By
/// the triangle inequality the numerator never exceeds the denominator, so
/// `ER` is in `[0, 1]`: `1` for a straight trend, close to `0` for noisy,
/// directionless movement.
///
/// If the denominator is exactly `0` (every value in the window is equal,
/// which also makes the numerator `0`), `ER` is defined as `0`: there is no
/// movement to be efficient about.
///
/// `last()` is `None` until `window_len + 1` values were observed.
///
/// Complexity: O(1) per update (two bounded `VecDeque`s); O(window_len)
/// memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct KaufmanEfficiencyRatio<T, V> {
    view: V,
    /// The non-zero window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    vals: VecDeque<T>,
    diffs: VecDeque<T>,
    diff_sum: T,
    prev_val: Option<T>,
}

impl<T, V> KaufmanEfficiencyRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Kaufman Efficiency Ratio with a chained `View` and a
    /// window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            vals: VecDeque::with_capacity(window_len.get() + 1),
            diffs: VecDeque::with_capacity(window_len.get()),
            diff_sum: T::zero(),
            prev_val: None,
        }
    }
}

impl<T, V> View<T> for KaufmanEfficiencyRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if let Some(prev) = self.prev_val {
            let diff = (val - prev).abs();
            if self.diffs.len() >= self.window_len.get() {
                let old = self.diffs.pop_front().expect("the window is full");
                // Rounding can leave a running sum of magnitudes slightly
                // below zero once they all left the window.
                self.diff_sum = (self.diff_sum - old).max(T::zero());
            }
            self.diffs.push_back(diff);
            self.diff_sum = self.diff_sum + diff;
        }
        self.prev_val = Some(val);

        if self.vals.len() > self.window_len.get() {
            self.vals.pop_front();
        }
        self.vals.push_back(val);
        debug_assert!(self.vals.len() <= self.window_len.get() + 1);
        debug_assert!(self.diffs.len() <= self.window_len.get());
    }

    fn last(&self) -> Option<T> {
        if self.vals.len() <= self.window_len.get() {
            return None;
        }
        let oldest = *self.vals.front().expect("checked length above");
        let newest = *self.vals.back().expect("checked length above");
        let numerator = (newest - oldest).abs();
        let er = if self.diff_sum == T::zero() {
            T::zero()
        } else {
            // The net change is at most the path length (triangle
            // inequality); only rounding of the running sum exceeds it.
            (numerator / self.diff_sum).min(T::one())
        };
        debug_assert!(er.is_finite(), "value must be finite");
        debug_assert!(er >= T::zero(), "ER is a ratio of magnitudes");
        debug_assert!(er <= T::one(), "ER is at most 1");
        Some(er)
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
    fn er_warms_up() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        for v in [1.0, 2.0, 3.0] {
            er.update(v);
            assert_eq!(er.last(), None);
        }
        er.update(4.0);
        assert!(er.last().is_some());
    }

    #[test]
    fn er_of_a_straight_trend_is_one() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
            er.update(v);
        }
        assert!((er.last().expect("ready") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn er_of_constant_input_is_zero_by_definition() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..10 {
            er.update(3.0);
        }
        assert_eq!(er.last(), Some(0.0));
    }

    #[test]
    fn er_of_a_zigzag_is_small() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in [0.0, 10.0, 0.0, 10.0, 0.0] {
            er.update(v);
        }
        let val = er.last().expect("ready");
        assert!(val.abs() < 1e-9, "{val}");
    }

    /// Naive reference: direct numerator / denominator over the trailing window.
    #[test]
    fn er_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(111);
        let window_len = 7usize;
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 100.0 - 50.0).collect();

        let mut er =
            KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        for (i, v) in vals.iter().enumerate() {
            er.update(*v);
            if i < window_len {
                assert_eq!(er.last(), None);
                continue;
            }
            let window = &vals[i - window_len..=i];
            let numerator = (window[window_len] - window[0]).abs();
            let denom: f64 = window.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
            let expected = if denom == 0.0 { 0.0 } else { numerator / denom };
            let got = er.last().expect("ready");
            assert!((got - expected).abs() < 1e-9, "step {i}");
        }
    }

    #[test]
    fn er_stays_in_unit_range() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for v in &TEST_DATA {
            er.update(*v);
            if let Some(val) = er.last() {
                assert!((0.0..=1.0).contains(&val), "{val}");
            }
        }
    }

    /// The rounded steps 0.7 and 0.19999999999999998 sum to
    /// 0.8999999999999999, less than the net change 0.9, which made the
    /// ratio exceed 1.
    #[test]
    fn er_rounding_does_not_exceed_one() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(2).unwrap());
        for v in [1.0, 0.3, 0.1] {
            er.update(v);
        }
        assert_eq!(er.last(), Some(1.0));
    }

    #[test]
    fn er_plot() {
        let mut er = KaufmanEfficiencyRatio::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            er.update(*v);
            if let Some(val) = er.last() {
                out.push(val);
            }
        }
        let filename = "img/kaufman_efficiency_ratio.png";
        plot_values(out, filename).unwrap();
    }
}

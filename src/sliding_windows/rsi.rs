// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Relative Strength Index Indicator

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Relative Strength Index Indicator
#[derive(Debug, Clone, CopyGetters)]
pub struct Rsi<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    avg_gain: T,
    avg_loss: T,
    old_ref: T,
    last_val: T,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Rsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a Relative Strength Index Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Rsi {
            view,
            window_len,
            avg_gain: T::zero(),
            avg_loss: T::zero(),
            old_ref: T::zero(),
            last_val: T::zero(),
            q_vals: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

impl<T, V> View<T> for Rsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value from `View` must be finite");

        if self.q_vals.is_empty() {
            self.old_ref = val;
            self.last_val = val;
        }
        let window_len = T::from(self.window_len.get()).expect("can convert");
        if self.q_vals.len() >= self.window_len.get() {
            // remove old
            let old_val = *self.q_vals.front().unwrap();
            let change = old_val - self.old_ref;
            debug_assert!(change.is_finite(), "`change` must be finite");
            self.old_ref = old_val;
            self.q_vals.pop_front();
            // Rounding can leave a running average of non-negative moves
            // slightly below zero once they all left the window, which
            // pushed the RSI below 0 or made `1 + rs` zero.
            if change > T::zero() {
                self.avg_gain = (self.avg_gain - change / window_len).max(T::zero());
            } else {
                self.avg_loss = (self.avg_loss - change.abs() / window_len).max(T::zero());
            }
        }
        self.q_vals.push_back(val);

        let change = val - self.last_val;
        debug_assert!(change.is_finite(), "`change` must be finite");
        self.last_val = val;
        if change > T::zero() {
            self.avg_gain = self.avg_gain + change / window_len;
        } else {
            self.avg_loss = self.avg_loss + change.abs() / window_len;
        }
        debug_assert!(self.avg_gain.is_finite(), "`avg_gain` must be finite");
        debug_assert!(self.avg_loss.is_finite(), "`avg_loss` must be finite");

        if self.q_vals.len() < self.window_len.get() {
            return;
        }

        let hundred = T::from(100.0).expect("can convert");
        if self.avg_loss == T::zero() {
            self.out = Some(hundred);
        } else {
            let rs = self.avg_gain / self.avg_loss;
            debug_assert!(rs.is_finite(), "`rs` must be finite");
            // Clamped as the running averages round, e.g. a gain that
            // should be 0 can be left at 1e-17.
            let rsi = (hundred - hundred / (T::one() + rs))
                .max(T::zero())
                .min(hundred);
            debug_assert!(rsi.is_finite(), "value must be finite");
            self.out = Some(rsi);
        }
    }

    #[inline(always)]
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
    fn rsi_plot() {
        let mut rsi = Rsi::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rsi.update(*v);
            if let Some(val) = rsi.last() {
                out.push(val);
            }
        }
        let filename = "img/rsi.png";
        plot_values(out, filename).unwrap();
    }

    /// The moves leave the window in another order than they entered it,
    /// so the rounded average gain ended at -5.6e-17 and the RSI at -1.4e-14.
    #[test]
    fn rsi_stays_in_range_when_rounded_moves_leave_the_window() {
        let mut rsi = Rsi::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        for v in [-2.0, 1.0, 2.0, 0.0, 0.0, 0.0] {
            rsi.update(v);
            if let Some(out) = rsi.last() {
                assert!((0.0..=100.0).contains(&out), "{out}");
            }
        }
    }

    #[test]
    fn rsi_range() {
        let mut rsi = Rsi::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            rsi.update(*v);
            if let Some(last) = rsi.last() {
                assert!(last >= 0.0);
                assert!(last <= 100.0);
            }
        }
    }
}

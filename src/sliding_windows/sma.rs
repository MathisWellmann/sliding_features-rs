// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! SMA - Simple Moving Average

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

#[derive(Debug, Clone, CopyGetters)]
/// SMA - Simple Moving Average
pub struct Sma<T, V> {
    view: V,
    /// The non-zero window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    sum: T,
}

impl<T, V> Sma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new simple moving average with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Sma {
            view,
            window_len,
            q_vals: VecDeque::new(),
            sum: T::zero(),
        }
    }
}

impl<T, V> View<T> for Sma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len.get() {
            let old_val = self.q_vals.pop_front().expect("the window is full");
            self.sum = self.sum - old_val;
        }
        self.q_vals.push_back(val);
        debug_assert!(self.q_vals.len() <= self.window_len.get());

        self.sum = self.sum + val;
    }

    fn last(&self) -> Option<T> {
        if self.q_vals.len() < self.window_len.get() {
            return None;
        }
        let sma = self.sum / T::from(self.q_vals.len()).expect("can convert");
        debug_assert!(sma.is_finite(), "value must be finite");
        Some(sma)
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        RngExt,
        rng,
    };

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    #[test]
    fn sma() {
        let mut rng = rng();

        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for _ in 0..1024 {
            let r = rng.random::<f64>();
            sma.update(r);
            if let Some(last) = sma.last() {
                assert!(last >= 0.0);
                assert!(last <= 1.0);
            }
        }
    }

    /// Exactly `window_len` values are averaged; one more used to stay in
    /// the window once it was full.
    #[test]
    fn sma_averages_the_window() {
        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(2).unwrap());
        let mut out = Vec::new();
        for v in [1.0, 3.0, 5.0, 7.0] {
            sma.update(v);
            out.push(sma.last());
        }
        assert_eq!(out, [None, Some(2.0), Some(4.0), Some(6.0)]);
    }

    #[test]
    fn sma_plot() {
        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            sma.update(*v);
            if let Some(val) = sma.last() {
                out.push(val);
            }
        }
        let filename = "img/sma.png";
        plot_values(out, filename).unwrap();
    }
}

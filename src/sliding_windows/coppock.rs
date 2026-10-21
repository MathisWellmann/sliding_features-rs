// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Coppock Curve - a long-term momentum oscillator by Edwin "Sedge" Coppock.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Roc,
    Wma,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// Coppock Curve (Edwin "Sedge" Coppock): a weighted moving average of the
/// sum of two rates of change: `WMA(wma_period)(ROC(long_period) +
/// ROC(short_period))`.
///
/// Classic defaults are `long_period = 14`, `short_period = 11` and
/// `wma_period = 10`, originally intended for monthly data but applicable
/// to any sampling period.
///
/// `last()` is `None` until `wma_period` observations: the crate's own
/// [`Roc`] is not gated on its own `window_len` (it starts comparing
/// against the first-ever observed value and only compares against a
/// properly `window_len`-ticks-old value once its window has filled), so
/// it feeds a (initially less accurate) value into the `Wma` from the very
/// first update; only the `Wma`'s own warm-up gates this indicator's
/// output.
///
/// Output range: unbounded, oscillates around `0`.
///
/// Complexity: amortized O(1) per update (`Wma`'s periodic resync),
/// O(`long_period.max(short_period) + wma_period`) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Coppock<T, V> {
    view: V,
    /// The longer rate-of-change period.
    #[getset(get_copy = "pub")]
    long_period: NonZeroUsize,
    /// The shorter rate-of-change period.
    #[getset(get_copy = "pub")]
    short_period: NonZeroUsize,
    /// The weighted-moving-average smoothing period.
    #[getset(get_copy = "pub")]
    wma_period: NonZeroUsize,
    roc_long: Roc<T, Echo<T>>,
    roc_short: Roc<T, Echo<T>>,
    wma: Wma<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Coppock<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Coppock Curve with a chained `View` and explicit
    /// periods for the two rates of change and the weighted moving average.
    pub fn with_periods(
        view: V,
        long_period: NonZeroUsize,
        short_period: NonZeroUsize,
        wma_period: NonZeroUsize,
    ) -> Self {
        Self {
            view,
            long_period,
            short_period,
            wma_period,
            roc_long: Roc::new(Echo::new(), long_period),
            roc_short: Roc::new(Echo::new(), short_period),
            wma: Wma::new(Echo::new(), wma_period),
            out: None,
        }
    }

    /// Create a new Coppock Curve with a chained `View` and the classic
    /// default periods: `ROC(14)`, `ROC(11)` and `WMA(10)`.
    pub fn new(view: V) -> Self {
        Self::with_periods(
            view,
            NonZeroUsize::new(14).expect("14 is non-zero"),
            NonZeroUsize::new(11).expect("11 is non-zero"),
            NonZeroUsize::new(10).expect("10 is non-zero"),
        )
    }
}

impl<T, V> View<T> for Coppock<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value from `View` must be finite");

        self.roc_long.update(val);
        self.roc_short.update(val);
        let Some((long, short)) = self.roc_long.last().zip(self.roc_short.last()) else {
            return;
        };
        let sum = long + short;
        debug_assert!(sum.is_finite(), "`sum` must be finite");

        self.wma.update(sum);
        let Some(wma) = self.wma.last() else { return };
        debug_assert!(wma.is_finite(), "`wma` must be finite");
        self.out = Some(wma);
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
        test_data::TEST_DATA,
    };

    #[test]
    fn coppock_plot() {
        let mut coppock = Coppock::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            coppock.update(*v);
            if let Some(val) = coppock.last() {
                out.push(val);
            }
        }
        let filename = "img/coppock.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn coppock_warms_up() {
        let mut coppock = Coppock::with_periods(
            Echo::new(),
            NonZeroUsize::new(6).unwrap(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        // The crate's `Roc` is not gated on its own `window_len` (see the
        // struct docs), so only `Wma`'s warm-up gates the output.
        let ready_at = 3;
        let mut seen = false;
        for (i, v) in TEST_DATA.iter().enumerate() {
            coppock.update(*v);
            if i + 1 < ready_at {
                assert_eq!(coppock.last(), None);
            }
            if coppock.last().is_some() {
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn coppock_constant_input_is_zero() {
        let mut coppock = Coppock::new(Echo::new());
        for _ in 0..50 {
            coppock.update(42.0);
        }
        assert_eq!(coppock.last(), Some(0.0));
    }

    #[test]
    fn coppock_finite() {
        let mut coppock = Coppock::new(Echo::new());
        for v in &TEST_DATA {
            coppock.update(*v);
            if let Some(val) = coppock.last() {
                assert!(val.is_finite(), "val = {val}");
            }
        }
    }

    /// Naive reference, independently reimplementing the crate's own `Roc`
    /// (including its not-gated-on-`window_len` warm-up, see the struct
    /// docs) and `Wma`, to validate the wiring of the real implementation.
    #[test]
    fn coppock_vs_naive_random() {
        use std::collections::VecDeque;

        /// Independent reimplementation of the crate's own `Roc`, matching
        /// its exact (not-gated) warm-up bookkeeping.
        struct NaiveRoc {
            period: usize,
            oldest: Option<f64>,
            q: VecDeque<f64>,
        }
        impl NaiveRoc {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    oldest: None,
                    q: VecDeque::new(),
                }
            }
            fn update(&mut self, val: f64) -> Option<f64> {
                if self.q.is_empty() {
                    self.oldest = Some(val);
                }
                if self.q.len() >= self.period {
                    let old = *self.q.front().expect("non-empty");
                    self.oldest = Some(old);
                    self.q.pop_front();
                }
                self.q.push_back(val);
                let oldest = self.oldest?;
                if oldest == 0.0 {
                    return None;
                }
                Some((val - oldest) / oldest * 100.0)
            }
        }

        let rng = romu::Rng::from_seed_with_64bit(41);
        let (long_period, short_period, wma_period) = (14_usize, 11_usize, 10_usize);
        let mut coppock = Coppock::with_periods(
            Echo::new(),
            NonZeroUsize::new(long_period).unwrap(),
            NonZeroUsize::new(short_period).unwrap(),
            NonZeroUsize::new(wma_period).unwrap(),
        );
        let mut naive_long = NaiveRoc::new(long_period);
        let mut naive_short = NaiveRoc::new(short_period);
        let mut wma_window: VecDeque<f64> = VecDeque::new();

        for _ in 0..3_000 {
            let x = rng.f64() * 100.0 + 1.0;
            coppock.update(x);

            let Some(sum) = naive_long
                .update(x)
                .zip(naive_short.update(x))
                .map(|(l, s)| l + s)
            else {
                continue;
            };

            if wma_window.len() >= wma_period {
                wma_window.pop_front();
            }
            wma_window.push_back(sum);

            let naive = if wma_window.len() == wma_period {
                let n = wma_window.len();
                let weighted: f64 = wma_window
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i + 1) as f64 * v)
                    .sum();
                let weight_total = (n * (n + 1)) as f64 / 2.0;
                Some(weighted / weight_total)
            } else {
                None
            };

            if let (Some(got), Some(expected)) = (coppock.last(), naive) {
                assert!(
                    (got - expected).abs() < 1e-9,
                    "got = {got}, expected = {expected}"
                );
            } else {
                assert_eq!(coppock.last(), naive);
            }
        }
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! ConnorsRSI - a composite momentum oscillator by Connors Research.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Rsi,
    sorted_window::SortedWindow,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// ConnorsRSI (Connors Research): a composite of three components, each in
/// `[0, 100]`, averaged into a single oscillator:
///
/// `CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(ROC(1), 100)) / 3`
///
/// - `RSI(close, 3)`: a plain 3-period RSI of the input values.
/// - `RSI(streak, 2)`: a 2-period RSI of the "streak", a running count of
///   consecutive up (positive) or down (negative) moves that resets to `0`
///   whenever consecutive values are equal, increments by `1` while the
///   input keeps rising and decrements by `1` while it keeps falling.
/// - `PercentRank(ROC(1), 100)`: the percentile rank of the most recent
///   1-bar rate of change `(x - x_prev) / x_prev * 100` against the last
///   `percent_rank_period` (default 100) such values: `100 * count_less /
///   percent_rank_period`, where `count_less` is the number of values in
///   that lookback (which includes the current one) strictly below it.
///
/// `last()` is `None` until all three components are ready, which requires
/// at least `percent_rank_period + 1` values (one extra to seed the first
/// rate of change).
///
/// Output range: `[0, 100]`.
///
/// Complexity per update: O(log n) (the `PercentRank` sorted window
/// dominates; the two `Rsi` legs are O(1)). Memory: O(`percent_rank_period`).
#[derive(Debug, Clone, CopyGetters)]
pub struct ConnorsRsi<T: Float, V> {
    view: V,
    /// The RSI period applied directly to the input.
    #[getset(get_copy = "pub")]
    rsi_period: NonZeroUsize,
    /// The RSI period applied to the up/down streak.
    #[getset(get_copy = "pub")]
    streak_rsi_period: NonZeroUsize,
    /// The lookback period for the `PercentRank` component.
    #[getset(get_copy = "pub")]
    percent_rank_period: NonZeroUsize,
    rsi: Rsi<T, Echo<T>>,
    streak_rsi: Rsi<T, Echo<T>>,
    percent_rank_window: SortedWindow<T>,
    prev_val: Option<T>,
    streak: i64,
    out: Option<T>,
}

impl<T, V> ConnorsRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `ConnorsRsi` with a chained `View` and explicit periods
    /// for the three components.
    pub fn with_params(
        view: V,
        rsi_period: NonZeroUsize,
        streak_rsi_period: NonZeroUsize,
        percent_rank_period: NonZeroUsize,
    ) -> Self {
        Self {
            view,
            rsi_period,
            streak_rsi_period,
            percent_rank_period,
            rsi: Rsi::new(Echo::new(), rsi_period),
            streak_rsi: Rsi::new(Echo::new(), streak_rsi_period),
            percent_rank_window: SortedWindow::new(percent_rank_period),
            prev_val: None,
            streak: 0,
            out: None,
        }
    }

    /// Create a new `ConnorsRsi` with a chained `View` and the classic
    /// default parameters: `RSI(3)`, `RSI(streak, 2)` and
    /// `PercentRank(ROC(1), 100)`.
    pub fn new(view: V) -> Self {
        Self::with_params(
            view,
            NonZeroUsize::new(3).expect("3 is non-zero"),
            NonZeroUsize::new(2).expect("2 is non-zero"),
            NonZeroUsize::new(100).expect("100 is non-zero"),
        )
    }

    /// Update the up/down streak given the previous and current value.
    fn next_streak(streak: i64, prev: T, val: T) -> i64 {
        if val > prev {
            if streak > 0 { streak + 1 } else { 1 }
        } else if val < prev {
            if streak < 0 { streak - 1 } else { -1 }
        } else {
            0
        }
    }
}

impl<T, V> View<T> for ConnorsRsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value from `View` must be finite");

        self.rsi.update(val);
        let rsi = self.rsi.last();

        let Some(prev) = self.prev_val else {
            self.prev_val = Some(val);
            return;
        };
        self.prev_val = Some(val);

        self.streak = Self::next_streak(self.streak, prev, val);
        let streak_val = T::from(self.streak).expect("streak fits into a float");
        self.streak_rsi.update(streak_val);
        let streak_rsi = self.streak_rsi.last();

        let roc = if prev == T::zero() {
            T::zero()
        } else {
            (val - prev) / prev * T::from(100.0).expect("can convert")
        };
        debug_assert!(roc.is_finite(), "`roc` must be finite");
        self.percent_rank_window.push(roc);
        let percent_rank = self.percent_rank_window.is_full().then(|| {
            let count_less = self.percent_rank_window.count_less(roc);
            T::from(100.0).expect("can convert") * T::from(count_less).expect("can convert")
                / T::from(self.percent_rank_period.get()).expect("can convert")
        });

        let Some(((rsi, streak_rsi), percent_rank)) = rsi.zip(streak_rsi).zip(percent_rank) else {
            return;
        };
        let three = T::from(3.0).expect("can convert");
        let out = (rsi + streak_rsi + percent_rank) / three;
        debug_assert!(out.is_finite(), "`out` must be finite");
        self.out = Some(out);
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
    fn connors_rsi_plot() {
        let mut crsi = ConnorsRsi::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            crsi.update(*v);
            if let Some(val) = crsi.last() {
                out.push(val);
            }
        }
        let filename = "img/connors_rsi.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn connors_rsi_warms_up() {
        let mut crsi = ConnorsRsi::with_params(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );
        let mut seen = false;
        for (i, v) in TEST_DATA.iter().enumerate() {
            crsi.update(*v);
            if i < 10 {
                assert_eq!(
                    crsi.last(),
                    None,
                    "must warm up for at least `percent_rank_period` values"
                );
            }
            if crsi.last().is_some() {
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn connors_rsi_range() {
        let mut crsi = ConnorsRsi::with_params(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(20).unwrap(),
        );
        const EPS: f64 = 1e-12;
        for v in &TEST_DATA {
            crsi.update(*v);
            if let Some(val) = crsi.last() {
                assert!((-EPS..=100.0 + EPS).contains(&val), "val = {val}");
            }
        }
    }

    #[test]
    fn connors_rsi_constant_input() {
        // A constant input has no change, so `streak` stays `0` and the
        // 1-bar `ROC` stays `0`: both RSI legs see a zero-loss sequence and
        // report `100` (the crate's `Rsi` zero-loss convention), while
        // `PercentRank` sees a window of identical values, so `count_less`
        // is always `0` (`100` is never strictly less than itself).
        // `CRSI = (100 + 100 + 0) / 3`.
        let mut crsi = ConnorsRsi::with_params(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );
        for _ in 0..50 {
            crsi.update(42.0);
        }
        let val = crsi.last().expect("ready");
        assert!((val - 200.0 / 3.0).abs() < 1e-9, "val = {val}");
    }

    #[test]
    fn connors_rsi_streak_resets_on_equal_and_flips_sign() {
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(0, 1.0, 2.0), 1);
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(1, 2.0, 3.0), 2);
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(2, 3.0, 3.0), 0);
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(0, 3.0, 2.0), -1);
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(-1, 2.0, 1.0), -2);
        // A down move right after an up streak starts a fresh down streak.
        assert_eq!(ConnorsRsi::<f64, Echo<f64>>::next_streak(2, 3.0, 2.0), -1);
    }

    /// Naive reference, independently reimplementing the streak and
    /// `PercentRank` logic, to validate the wiring of the real
    /// implementation (the two `Rsi` legs reuse the crate's own `Rsi`,
    /// already covered by its own tests).
    #[test]
    fn connors_rsi_vs_naive_percent_rank_and_streak() {
        use std::collections::VecDeque;

        /// Independent reimplementation of the crate's own sliding-window
        /// `Rsi`, matching its gain/loss bookkeeping exactly.
        struct NaiveRsi {
            period: usize,
            avg_gain: f64,
            avg_loss: f64,
            old_ref: f64,
            last_val: f64,
            q: VecDeque<f64>,
        }
        impl NaiveRsi {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    avg_gain: 0.0,
                    avg_loss: 0.0,
                    old_ref: 0.0,
                    last_val: 0.0,
                    q: VecDeque::new(),
                }
            }
            fn update(&mut self, val: f64) -> Option<f64> {
                if self.q.is_empty() {
                    self.old_ref = val;
                    self.last_val = val;
                }
                let n = self.period as f64;
                if self.q.len() >= self.period {
                    let old_val = *self.q.front().expect("non-empty");
                    let change = old_val - self.old_ref;
                    self.old_ref = old_val;
                    self.q.pop_front();
                    if change > 0.0 {
                        self.avg_gain -= change / n;
                    } else {
                        self.avg_loss -= change.abs() / n;
                    }
                }
                self.q.push_back(val);
                let change = val - self.last_val;
                self.last_val = val;
                if change > 0.0 {
                    self.avg_gain += change / n;
                } else {
                    self.avg_loss += change.abs() / n;
                }
                if self.q.len() < self.period {
                    return None;
                }
                Some(if self.avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
                })
            }
        }

        let rng = romu::Rng::from_seed_with_64bit(11);
        let percent_rank_period = 30_usize;
        let mut crsi = ConnorsRsi::with_params(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(percent_rank_period).unwrap(),
        );
        let mut naive_rsi = NaiveRsi::new(3);
        let mut naive_streak_rsi = NaiveRsi::new(2);

        let mut prev: Option<f64> = None;
        let mut streak = 0_i64;
        let mut window: VecDeque<f64> = VecDeque::new();

        for _ in 0..5_000 {
            let x = rng.f64() * 100.0 + 1.0;
            crsi.update(x);
            let rsi = naive_rsi.update(x);

            let Some(p) = prev else {
                prev = Some(x);
                continue;
            };
            prev = Some(x);

            streak = ConnorsRsi::<f64, Echo<f64>>::next_streak(streak, p, x);
            let streak_rsi = naive_streak_rsi.update(streak as f64);
            let roc = (x - p) / p * 100.0;
            if window.len() == percent_rank_period {
                window.pop_front();
            }
            window.push_back(roc);

            if window.len() < percent_rank_period {
                assert_eq!(crsi.last(), None);
                continue;
            }
            let count_less = window.iter().filter(|v| **v < roc).count();
            let percent_rank = 100.0 * count_less as f64 / percent_rank_period as f64;

            let Some(((rsi, streak_rsi), got)) = rsi.zip(streak_rsi).zip(crsi.last()) else {
                continue;
            };
            let expected = (rsi + streak_rsi + percent_rank) / 3.0;
            assert!(
                (got - expected).abs() < 1e-9,
                "got = {got}, expected = {expected}"
            );
        }
    }
}

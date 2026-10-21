// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! CMO - Chande Momentum Oscillator

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// CMO - Chande Momentum Oscillator (Tushar Chande, "The New Technical
/// Trader", 1994).
///
/// `CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down)`, where `sum_up`
/// and `sum_down` are the sums of the upward and downward step-to-step
/// changes over the last `window_len` steps. Output is in `[-100, 100]`:
/// `100` when every step moved up, `-100` when every step moved down.
///
/// If both sums are `0` (every value in the window is equal), `CMO` is
/// defined as `0`: there is no momentum either way.
///
/// `last()` is `None` until `window_len + 1` values were observed.
///
/// Complexity: O(1) per update (one bounded `VecDeque` of `(up, down)`
/// pairs); O(window_len) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Cmo<T, V> {
    view: V,
    /// The non-zero window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    moves: VecDeque<(T, T)>,
    sum_up: T,
    sum_down: T,
    prev_val: Option<T>,
}

impl<T, V> Cmo<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new CMO with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            moves: VecDeque::with_capacity(window_len.get()),
            sum_up: T::zero(),
            sum_down: T::zero(),
            prev_val: None,
        }
    }
}

impl<T, V> View<T> for Cmo<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(prev) = self.prev_val else {
            self.prev_val = Some(val);
            return;
        };
        let diff = val - prev;
        let up = diff.max(T::zero());
        let down = (-diff).max(T::zero());

        if self.moves.len() >= self.window_len.get() {
            let (old_up, old_down) = self.moves.pop_front().expect("the window is full");
            // Rounding can leave a running sum of non-negative moves
            // slightly below zero once they all left the window.
            self.sum_up = (self.sum_up - old_up).max(T::zero());
            self.sum_down = (self.sum_down - old_down).max(T::zero());
        }
        self.moves.push_back((up, down));
        self.sum_up = self.sum_up + up;
        self.sum_down = self.sum_down + down;
        self.prev_val = Some(val);

        debug_assert!(self.moves.len() <= self.window_len.get());
        debug_assert!(self.sum_up >= T::zero(), "sum of upward moves is >= 0");
        debug_assert!(self.sum_down >= T::zero(), "sum of downward moves is >= 0");
    }

    fn last(&self) -> Option<T> {
        if self.moves.len() < self.window_len.get() {
            return None;
        }
        let total = self.sum_up + self.sum_down;
        let cmo = if total == T::zero() {
            T::zero()
        } else {
            let hundred = T::from(100.0).expect("can convert");
            // The running sums round, which can push the ratio just past
            // the bound its definition guarantees.
            (hundred * (self.sum_up - self.sum_down) / total)
                .max(-hundred)
                .min(hundred)
        };
        debug_assert!(cmo.is_finite(), "value must be finite");
        let hundred = T::from(100.0).expect("can convert");
        debug_assert!(
            (-hundred..=hundred).contains(&cmo),
            "CMO is bounded by its definition"
        );
        Some(cmo)
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
    fn cmo_warms_up() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(3).unwrap());
        for v in [1.0, 2.0, 3.0] {
            cmo.update(v);
            assert_eq!(cmo.last(), None);
        }
        cmo.update(4.0);
        assert!(cmo.last().is_some());
    }

    #[test]
    fn cmo_of_a_strictly_rising_series_is_100() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            cmo.update(v);
        }
        assert_eq!(cmo.last(), Some(100.0));
    }

    #[test]
    fn cmo_of_a_strictly_falling_series_is_minus_100() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in [5.0, 4.0, 3.0, 2.0, 1.0] {
            cmo.update(v);
        }
        assert_eq!(cmo.last(), Some(-100.0));
    }

    #[test]
    fn cmo_of_constant_input_is_zero_by_definition() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..10 {
            cmo.update(3.0);
        }
        assert_eq!(cmo.last(), Some(0.0));
    }

    /// Naive reference: direct sums of up/down moves over the trailing window.
    #[test]
    fn cmo_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(222);
        let window_len = 7usize;
        let vals: Vec<f64> = (0..300).map(|_| rng.f64() * 100.0 - 50.0).collect();

        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        for (i, v) in vals.iter().enumerate() {
            cmo.update(*v);
            if i < window_len {
                assert_eq!(cmo.last(), None);
                continue;
            }
            let window = &vals[i - window_len..=i];
            let (mut up, mut down) = (0.0, 0.0);
            for w in window.windows(2) {
                let diff = w[1] - w[0];
                if diff > 0.0 {
                    up += diff;
                } else {
                    down += -diff;
                }
            }
            let expected = if up + down == 0.0 {
                0.0
            } else {
                100.0 * (up - down) / (up + down)
            };
            let got = cmo.last().expect("ready");
            assert!((got - expected).abs() < 1e-9, "step {i}");
        }
    }

    #[test]
    fn cmo_stays_in_range() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for v in &TEST_DATA {
            cmo.update(*v);
            if let Some(val) = cmo.last() {
                assert!((-100.0..=100.0).contains(&val), "{val}");
            }
        }
    }

    /// The running sums round: here the downward moves 0.6 and 0.1 leave
    /// `sum_down` at -2.8e-17, which made the CMO exceed 100.
    #[test]
    fn cmo_stays_in_range_when_rounded_moves_leave_the_window() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(2).unwrap());
        for v in [0.7, 0.1, 0.0, 0.0, 0.5] {
            cmo.update(v);
            if let Some(val) = cmo.last() {
                assert!((-100.0..=100.0).contains(&val), "{val}");
            }
        }
        assert_eq!(cmo.last(), Some(100.0));
    }

    #[test]
    fn cmo_plot() {
        let mut cmo = Cmo::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cmo.update(*v);
            if let Some(val) = cmo.last() {
                out.push(val);
            }
        }
        let filename = "img/cmo.png";
        plot_values(out, filename).unwrap();
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
//! it is also possible to use a custom moving average instead of the default EMA in the original

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

#[derive(Debug, Clone, CopyGetters)]
/// A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
/// it is also possible to use a custom moving average instead of the default EMA in the original
pub struct PolarizedFractalEfficiency<T, V, M> {
    view: V,
    moving_average: M, // defines which moving average to use, default is EMA
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V, M> PolarizedFractalEfficiency<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    /// Create a new PolarizedFractalEfficiency indicator with a chained view, custom moving
    /// average and a window length.
    ///
    /// # Panics
    ///
    /// If `window_len` is 1: the efficiency of a single value is undefined.
    pub fn new(view: V, moving_average: M, window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= 2,
            "a fractal efficiency needs at least two values"
        );
        Self {
            view,
            moving_average,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

impl<T, V, M> View<T> for PolarizedFractalEfficiency<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len.get() {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() >= self.window_len.get() {
            let p = efficiency(&self.q_vals);
            // apply a moving average
            self.moving_average.update(p);
            self.out = self.moving_average.last();
        }
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out.inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

/// The polarized efficiency of the path through `vals`, one unit of time
/// apart: the straight distance from the first to the last value over the
/// length of the path, negative when the last value is below the first.
///
/// Both span the same `n = len - 1` steps, so by the triangle inequality the
/// magnitude is at most 1 (up to rounding, which is clamped).
fn efficiency<T: Float>(vals: &VecDeque<T>) -> T {
    assert!(vals.len() >= 2, "a path has at least two values");
    let (first, last) = (
        *vals.front().expect("not empty"),
        *vals.back().expect("not empty"),
    );
    let steps = T::from(vals.len() - 1).expect("can convert");
    let path = vals
        .iter()
        .zip(vals.iter().skip(1))
        .fold(T::zero(), |sum, (a, b)| {
            sum + ((*b - *a).powi(2) + T::one()).sqrt()
        });
    debug_assert!(path >= steps, "every step is at least one unit long");
    let straight = ((last - first).powi(2) + steps.powi(2)).sqrt();
    let magnitude = (straight / path).min(T::one());
    if last < first { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    #[test]
    fn polarized_fractal_efficiency() {
        // TODO: don't be so lazy with this test. maybe compare against a reference implementation.
        let mut pfe = PolarizedFractalEfficiency::new(
            Echo::new(),
            Ema::new(Echo::new(), NonZeroUsize::new(16).unwrap()),
            NonZeroUsize::new(16).unwrap(),
        );
        for v in &TEST_DATA {
            pfe.update(*v);
            if let Some(val) = pfe.last() {
                assert!(val <= 1.0);
                assert!(val >= -1.0);
            }
        }
    }

    /// A straight path is fully efficient, in its direction; one segment
    /// too few in the path length used to push it beyond 1.
    #[test]
    fn efficiency_of_reference_paths() {
        let path = |vals: &[f64]| efficiency(&VecDeque::from_iter(vals.iter().copied()));
        assert!((path(&[1.0, 2.0, 3.0, 4.0]) - 1.0).abs() < 1e-12);
        assert!((path(&[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(path(&[5.0, 5.0, 5.0]), 1.0, "a flat path is straight");
        // Up 3 then down 2: the direction is that of the whole window, not
        // of the last step. Straight sqrt(1 + 4), path sqrt(10) + sqrt(5).
        let expected = 5.0_f64.sqrt() / (10.0_f64.sqrt() + 5.0_f64.sqrt());
        assert!((path(&[1.0, 4.0, 2.0]) - expected).abs() < 1e-12);
        let zigzag = path(&[0.0, 10.0, 0.0, 10.0, 0.0]);
        assert!(zigzag.abs() < 0.11, "{zigzag}");
    }

    #[test]
    fn pfe_without_smoothing_stays_in_range() {
        for window in [2, 3, 10] {
            let mut pfe = PolarizedFractalEfficiency::new(
                Echo::new(),
                Echo::new(),
                NonZeroUsize::new(window).unwrap(),
            );
            for v in &TEST_DATA {
                pfe.update(*v);
                if let Some(val) = pfe.last() {
                    assert!((-1.0..=1.0).contains(&val), "window {window}: {val}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "at least two values")]
    fn window_of_one_is_rejected() {
        let _ = PolarizedFractalEfficiency::<f64, _, _>::new(
            Echo::new(),
            Echo::new(),
            NonZeroUsize::new(1).unwrap(),
        );
    }

    #[test]
    fn polarized_fractal_efficiency_plot() {
        let mut pfe = PolarizedFractalEfficiency::new(
            Echo::new(),
            Ema::new(Echo::new(), NonZeroUsize::new(16).unwrap()),
            NonZeroUsize::new(16).unwrap(),
        );
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            pfe.update(*v);
            if let Some(val) = pfe.last() {
                out.push(val);
            }
        }
        let filename = "img/polarized_fractal_efficiency.png";
        plot_values(out, filename).unwrap();
    }
}

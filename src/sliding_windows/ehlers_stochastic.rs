// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! John Ehlers Stochastic Indicator

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use super::{
    RoofingFilter,
    SuperSmoother,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// John Ehlers Stochastic Indicator.
///
/// Runs a [`RoofingFilter`] over the chained view to remove both slow trend
/// and fast noise, rescales the result to `[0, 1]` using the highest and
/// lowest filtered values over `stochastic_len` samples (a classic
/// stochastic `%K`), then smooths that ratio with another [`SuperSmoother`]
/// of the same length. Output is close to `[0, 1]`, though the final
/// `SuperSmoother` pass can slightly overshoot those bounds (IIR filter
/// ringing), same as any Ehlers two-pole smoother applied to a bounded
/// series.
///
/// When the rescaling window has zero range (every filtered value equal),
/// the ratio is defined as the neutral midpoint `0.5` rather than left
/// undefined.
///
/// Reference: John Ehlers, "Decyclers",
/// <http://www.stockspotter.com/files/PredictiveIndicators.pdf>
///
/// # Complexity
/// O(`stochastic_len`) per update: the highest/lowest filtered values are
/// recomputed from the raw window buffer every step.
#[derive(Debug, Clone, CopyGetters)]
pub struct EhlersStochastic<T, V> {
    view: V,
    roofing_filter: RoofingFilter<T, Echo<T>>,
    smoother: SuperSmoother<T, Echo<T>>,
    /// The lookback length used to rescale the filtered value to `[0, 1]`
    /// and to smooth that ratio.
    #[getset(get_copy = "pub")]
    stochastic_len: NonZeroUsize,
    buf: VecDeque<T>,
}

impl<T, V> EhlersStochastic<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View`, the two `RoofingFilter`
    /// window lengths, and the stochastic rescaling/smoothing length.
    ///
    /// # Panics
    /// Panics if `stochastic_len < 2`, since a highest/lowest range needs at
    /// least two samples.
    pub fn new(
        view: V,
        window_len_low_pass: NonZeroUsize,
        super_smoother_len_high_pass: NonZeroUsize,
        stochastic_len: NonZeroUsize,
    ) -> Self {
        assert!(
            stochastic_len.get() >= 2,
            "stochastic_len must be at least 2, got {}",
            stochastic_len.get()
        );
        Self {
            view,
            roofing_filter: RoofingFilter::new(
                Echo::new(),
                window_len_low_pass,
                super_smoother_len_high_pass,
            ),
            smoother: SuperSmoother::new(Echo::new(), stochastic_len),
            stochastic_len,
            buf: VecDeque::with_capacity(stochastic_len.get()),
        }
    }
}

impl<T, V> View<T> for EhlersStochastic<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.roofing_filter.update(val);
        let Some(filt) = self.roofing_filter.last() else {
            return;
        };
        debug_assert!(filt.is_finite(), "value must be finite");

        if self.buf.len() >= self.stochastic_len.get() {
            self.buf.pop_front();
        }
        self.buf.push_back(filt);
        if self.buf.len() < self.stochastic_len.get() {
            return;
        }

        let stoc = stochastic_ratio(&self.buf, filt);
        debug_assert!(
            stoc >= T::zero() && stoc <= T::one(),
            "stochastic ratio must be in [0, 1]"
        );
        self.smoother.update(stoc);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.smoother.last().inspect(|out| {
            debug_assert!(out.is_finite(), "value must be finite");
        })
    }
}

/// Rescale `filt` to `[0, 1]` using the highest and lowest value in `buf`.
/// Defined as `0.5` when the window has zero range.
fn stochastic_ratio<T: Float>(buf: &VecDeque<T>, filt: T) -> T {
    assert!(!buf.is_empty(), "buffer must not be empty");
    let mut highest = T::neg_infinity();
    let mut lowest = T::infinity();
    for v in buf {
        highest = highest.max(*v);
        lowest = lowest.min(*v);
    }
    let range = highest - lowest;
    debug_assert!(range >= T::zero(), "range must be non-negative");
    if range <= T::zero() {
        return T::from(0.5).expect("can convert");
    }
    (filt - lowest) / range
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn new_default(view: Echo<f64>) -> EhlersStochastic<f64, Echo<f64>> {
        EhlersStochastic::new(
            view,
            NonZeroUsize::new(48).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        )
    }

    #[test]
    fn warmup_then_ready() {
        let mut es = new_default(Echo::new());
        for (i, v) in TEST_DATA.iter().enumerate() {
            es.update(*v);
            if es.last().is_some() {
                assert!(i > 48, "should not be ready before roofing filter warms up");
                return;
            }
        }
        panic!("never warmed up over the whole test data set");
    }

    #[test]
    fn constant_input_produces_finite_bounded_output() {
        let mut es = new_default(Echo::new());
        for _ in 0..200 {
            es.update(5.0);
        }
        // A flat input decays the roofing filter's high-pass stage towards
        // (but never exactly to) zero, so the rescaling window's range
        // shrinks to floating-point noise: the ratio then divides noise by
        // noise and is not meaningfully defined. This is an inherent
        // property of the reference formula (not a bug here), so this test
        // only checks the output stays finite and within the documented
        // overshoot-tolerant band, rather than asserting a neutral value.
        let out = es.last().expect("warmed up");
        assert!(out.is_finite(), "{out}");
        assert!((-0.5..=1.5).contains(&out), "{out}");
    }

    #[test]
    fn output_stays_roughly_bounded_on_real_data() {
        let mut es = new_default(Echo::new());
        for v in &TEST_DATA {
            es.update(*v);
            if let Some(out) = es.last() {
                // allow a margin for IIR filter overshoot, see module docs.
                assert!((-0.5..=1.5).contains(&out), "out of range: {out}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "stochastic_len must be at least 2")]
    fn stochastic_len_one_panics() {
        let _ = EhlersStochastic::<f64, _>::new(
            Echo::new(),
            NonZeroUsize::new(48).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        );
    }

    #[test]
    fn ehlers_stochastic_plot() {
        let mut es = new_default(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            es.update(*v);
            if let Some(val) = es.last() {
                out.push(val);
            }
        }
        let filename = "img/ehlers_stochastic.png";
        plot_values(out, filename).unwrap();
    }
}

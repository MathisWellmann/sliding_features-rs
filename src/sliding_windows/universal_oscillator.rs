// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! John Ehlers Universal Oscillator

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::SuperSmoother;
use crate::{
    View,
    pure_functions::Echo,
};

/// Decay factor of the Automatic Gain Control's peak tracker. Fixed by
/// Ehlers as the value that minimizes distortion without introducing lag.
const AGC_DECAY: f64 = 0.991;

/// John Ehlers Universal Oscillator.
///
/// Builds a momentum term `WhiteNoise = (x - x[2]) / 2` from the chained
/// view, smooths it with a [`SuperSmoother`] of period `band_edge` to get
/// `Filt`, then normalizes `Filt` with an Automatic Gain Control: a tracked
/// peak `Peak` that decays by `0.991` every step and jumps up to
/// `|Filt|` whenever that is larger. Output is `Filt / Peak`, which by
/// construction (`Peak >= |Filt|` always) is exactly in `[-1, 1]`.
///
/// Reference: John Ehlers, "Whiter Is Brighter", TASC January 2015.
///
/// # Complexity
/// O(1) amortized per update.
#[derive(Debug, Clone, CopyGetters)]
pub struct UniversalOscillator<T, V> {
    view: V,
    smoother: SuperSmoother<T, Echo<T>>,
    /// The SuperSmoother period (the "band edge" critical period).
    #[getset(get_copy = "pub")]
    band_edge: NonZeroUsize,
    i: usize,
    val_1: T,
    val_2: T,
    peak: T,
    out: Option<T>,
}

impl<T, V> UniversalOscillator<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View` and a band edge period.
    pub fn new(view: V, band_edge: NonZeroUsize) -> Self {
        Self {
            view,
            smoother: SuperSmoother::new(Echo::new(), band_edge),
            band_edge,
            i: 0,
            val_1: T::zero(),
            val_2: T::zero(),
            // seeded with a tiny positive epsilon, as in Ehlers' reference
            // code, so the first `Filt / Peak` division is well-defined;
            // it is immediately overtaken by the first real `|Filt|`.
            peak: T::from(1e-7).expect("can convert"),
            out: None,
        }
    }
}

impl<T, V> View<T> for UniversalOscillator<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.i < 2 {
            self.val_2 = self.val_1;
            self.val_1 = val;
            self.i += 1;
            return;
        }
        let two = T::from(2.0).expect("can convert");
        let white_noise = (val - self.val_2) / two;
        self.val_2 = self.val_1;
        self.val_1 = val;

        self.smoother.update(white_noise);
        let Some(filt) = self.smoother.last() else {
            return;
        };
        debug_assert!(filt.is_finite(), "value must be finite");

        let decay = T::from(AGC_DECAY).expect("can convert");
        self.peak = (decay * self.peak).max(filt.abs());
        debug_assert!(self.peak > T::zero(), "peak must stay strictly positive");

        // Clamped defensively: by construction `peak >= |filt|`, so the
        // ratio is mathematically in [-1, 1] already; the clamp only guards
        // against a last-bit rounding overshoot from the division itself.
        let universal = (filt / self.peak).max(-T::one()).min(T::one());
        debug_assert!(universal.is_finite(), "value must be finite");
        self.out = Some(universal);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out.inspect(|out| {
            debug_assert!(out.is_finite(), "value must be finite");
            debug_assert!(
                *out >= -T::one() && *out <= T::one(),
                "output must be in [-1, 1]"
            );
        })
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

    /// Naive O(n) reference replicating the full recurrence directly.
    fn naive_universal_oscillator(vals: &[f64], band_edge: usize) -> Vec<f64> {
        let wl = band_edge as f64;
        // 4.4422 is `SuperSmoother`'s own truncated constant for 1.414*pi
        // radians; using the same truncated value here (rather than full
        // precision) is required to match it bit-for-bit over many steps.
        let a1 = (-1.414 * std::f64::consts::PI / wl).exp();
        let b1 = 2.0 * a1 * (4.4422 / wl).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        let mut out = Vec::with_capacity(vals.len());
        let (mut filt_1, mut filt_2) = (0.0, 0.0);
        let mut wn_1 = 0.0;
        let mut peak = 1e-7;
        for (i, _) in vals.iter().enumerate() {
            if i < 2 {
                out.push(None);
                continue;
            }
            let wn = (vals[i] - vals[i - 2]) / 2.0;
            let filt = c1 * (wn + wn_1) / 2.0 + c2 * filt_1 + c3 * filt_2;
            filt_2 = filt_1;
            filt_1 = filt;
            wn_1 = wn;

            if i + 1 < band_edge + 2 {
                // mirrors SuperSmoother's own warmup gate.
                out.push(None);
                continue;
            }
            peak = (0.991 * peak).max(filt.abs());
            out.push(Some((filt / peak).clamp(-1.0, 1.0)));
        }
        out.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()
    }

    #[test]
    fn warmup_then_ready() {
        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        for (i, v) in TEST_DATA.iter().enumerate() {
            uo.update(*v);
            if uo.last().is_some() {
                assert!(i >= 20, "should not be ready before band_edge warmup");
                return;
            }
        }
        panic!("never warmed up over the whole test data set");
    }

    #[test]
    fn output_is_bounded_in_unit_interval_on_real_data() {
        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        for v in &TEST_DATA {
            uo.update(*v);
            if let Some(out) = uo.last() {
                assert!((-1.0..=1.0).contains(&out), "out of range: {out}");
            }
        }
    }

    #[test]
    fn steady_uptrend_converges_to_positive_one() {
        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..100 {
            uo.update(f64::from(i));
        }
        assert!((uo.last().expect("warmed up") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn steady_downtrend_converges_to_negative_one() {
        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for i in 0..100 {
            uo.update(-f64::from(i));
        }
        assert!((uo.last().expect("warmed up") + 1.0).abs() < 1e-9);
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(17);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const BAND_EDGE: usize = 20;
        let expected = naive_universal_oscillator(&vals, BAND_EDGE);

        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(BAND_EDGE).unwrap());
        for (i, v) in vals.iter().enumerate() {
            uo.update(*v);
            match (uo.last(), expected[i].is_nan()) {
                (None, true) => {}
                (Some(got), false) => {
                    assert!(
                        (got - expected[i]).abs() < 1e-6,
                        "at {i}: {} vs {got}",
                        expected[i]
                    );
                }
                (got, _) => panic!(
                    "readiness mismatch at {i}: got {got:?}, expected {:?}",
                    expected[i]
                ),
            }
        }
    }

    #[test]
    fn universal_oscillator_plot() {
        let mut uo = UniversalOscillator::new(Echo::new(), NonZeroUsize::new(20).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            uo.update(*v);
            if let Some(val) = uo.last() {
                out.push(val);
            }
        }
        let filename = "img/universal_oscillator.png";
        plot_values(out, filename).unwrap();
    }
}

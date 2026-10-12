//! John Ehlers Reversion Index

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use super::SuperSmoother;
use crate::{
    View,
    pure_functions::Echo,
};

/// John Ehlers Reversion Index.
///
/// Sums the bar-to-bar changes of the chained view over `length` samples and
/// normalizes that sum by the sum of their absolute values: `Ratio =
/// sum(delta) / sum(|delta|)`, which by the triangle inequality is always in
/// `[-1, 1]` (`0` when there is no history for a delta yet, or when the
/// window has no change at all). `Ratio` is then smoothed with a
/// [`SuperSmoother`] of period `smoother_len`, since the raw ratio over a
/// short window is too erratic to read directly. Ehlers suggests `length`
/// cover about half of the dominant cycle in the data, and recommends
/// instantiating this indicator twice, with `smoother_len` set to `4` and
/// `8`, to get a "trigger" and "smooth" line whose crossings mark peaks and
/// valleys.
///
/// The raw (unsmoothed) ratio is exposed via [`ReversionIndex::ratio`] for
/// callers who want it directly.
///
/// Reference: John Ehlers, "The Reversion Index", Stocks & Commodities,
/// January 2026. This is a very recently published indicator (at the time
/// of writing); this implementation follows the formula in the published
/// EasyLanguage listing as closely as possible, but, unlike the crate's
/// long-established Ehlers indicators, it has not been cross-checked
/// against a widely used independent implementation.
///
/// # Complexity
/// O(1) amortized per update.
#[derive(Debug, Clone, CopyGetters)]
pub struct ReversionIndex<T, V> {
    view: V,
    smoother: SuperSmoother<T, Echo<T>>,
    /// The rolling window length used to sum the bar-to-bar changes.
    #[getset(get_copy = "pub")]
    length: NonZeroUsize,
    prev_val: Option<T>,
    deltas: VecDeque<T>,
    delta_sum: T,
    abs_delta_sum: T,
    ratio: Option<T>,
}

impl<T, V> ReversionIndex<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View`, the rolling sum length,
    /// and the `SuperSmoother` period applied to the ratio.
    pub fn new(view: V, length: NonZeroUsize, smoother_len: NonZeroUsize) -> Self {
        Self {
            view,
            smoother: SuperSmoother::new(Echo::new(), smoother_len),
            length,
            prev_val: None,
            deltas: VecDeque::with_capacity(length.get()),
            delta_sum: T::zero(),
            abs_delta_sum: T::zero(),
            ratio: None,
        }
    }

    /// The raw, unsmoothed ratio `sum(delta) / sum(|delta|)` over `length`
    /// samples, or `None` before the first sample.
    #[must_use]
    pub fn ratio(&self) -> Option<T> {
        self.ratio
    }
}

impl<T, V> View<T> for ReversionIndex<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        // The very first sample has no prior value to diff against; follow
        // the published reference implementations and treat that delta as 0.
        let delta = self.prev_val.map_or(T::zero(), |prev| val - prev);
        self.prev_val = Some(val);

        if self.deltas.len() >= self.length.get() {
            let old = self.deltas.pop_front().expect("buffer is at full capacity");
            self.delta_sum = self.delta_sum - old;
            self.abs_delta_sum = self.abs_delta_sum - old.abs();
        }
        self.deltas.push_back(delta);
        self.delta_sum = self.delta_sum + delta;
        self.abs_delta_sum = self.abs_delta_sum + delta.abs();

        let ratio = if self.abs_delta_sum > T::zero() {
            // Clamped defensively: by the triangle inequality
            // `|delta_sum| <= abs_delta_sum` always holds mathematically,
            // so the ratio is in [-1, 1] already; the clamp only guards
            // against a last-bit rounding overshoot from the division.
            (self.delta_sum / self.abs_delta_sum)
                .max(-T::one())
                .min(T::one())
        } else {
            T::zero()
        };
        debug_assert!(ratio.is_finite(), "value must be finite");
        self.ratio = Some(ratio);

        self.smoother.update(ratio);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.smoother.last().inspect(|out| {
            debug_assert!(out.is_finite(), "value must be finite");
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

    /// Naive O(n) reference: ratio recomputed from a plain slice of deltas.
    fn naive_ratio(vals: &[f64], length: usize) -> f64 {
        let deltas: Vec<f64> = (0..vals.len())
            .map(|i| if i == 0 { 0.0 } else { vals[i] - vals[i - 1] })
            .collect();
        let start = deltas.len().saturating_sub(length);
        let window = &deltas[start..];
        let delta_sum: f64 = window.iter().sum();
        let abs_delta_sum: f64 = window.iter().map(|d| d.abs()).sum();
        if abs_delta_sum > 0.0 {
            delta_sum / abs_delta_sum
        } else {
            0.0
        }
    }

    #[test]
    fn warmup_then_ready() {
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(20).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );
        ri.update(1.0);
        assert_eq!(ri.ratio(), Some(0.0));
        assert_eq!(ri.last(), None);
        for _ in 0..4 {
            ri.update(2.0);
        }
        assert!(ri.last().is_some());
    }

    #[test]
    fn monotonic_increase_has_ratio_one() {
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );
        for i in 0..30 {
            ri.update(f64::from(i));
        }
        assert!((ri.ratio().expect("ready") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn monotonic_decrease_has_ratio_negative_one() {
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );
        for i in 0..30 {
            ri.update(-f64::from(i));
        }
        assert!((ri.ratio().expect("ready") + 1.0).abs() < 1e-9);
    }

    #[test]
    fn constant_input_has_ratio_zero() {
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );
        for _ in 0..30 {
            ri.update(5.0);
        }
        assert_eq!(ri.ratio(), Some(0.0));
    }

    #[test]
    fn output_is_bounded_in_unit_interval_on_real_data() {
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(20).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        );
        for v in &TEST_DATA {
            ri.update(*v);
            assert!((-1.0..=1.0).contains(&ri.ratio().expect("ready since the first sample")));
            if let Some(out) = ri.last() {
                // allow a margin for IIR filter overshoot.
                assert!((-1.5..=1.5).contains(&out), "out of range: {out}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(31);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const LENGTH: usize = 20;
        let mut ri = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(LENGTH).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        );
        for (i, v) in vals.iter().enumerate() {
            ri.update(*v);
            let expected = naive_ratio(&vals[..=i], LENGTH);
            let got = ri.ratio().expect("ready since the first sample");
            assert!((expected - got).abs() < 1e-9, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    fn reversion_index_plot() {
        let mut smooth = ReversionIndex::new(
            Echo::new(),
            NonZeroUsize::new(20).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        );
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            smooth.update(*v);
            if let Some(val) = smooth.last() {
                out.push(val);
            }
        }
        let filename = "img/reversion_index.png";
        plot_values(out, filename).unwrap();
    }
}

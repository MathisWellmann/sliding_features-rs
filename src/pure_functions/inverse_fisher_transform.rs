//! John Ehlers Inverse Fisher Transform

use num::Float;

use crate::View;

/// John Ehlers Inverse Fisher Transform.
///
/// Applies `IFish(x) = (exp(2*gain*x) - 1) / (exp(2*gain*x) + 1)` to the
/// output of the chained view, compressing an unbounded (or wide-range)
/// oscillator into `[-1, 1]`: values far from zero are pushed close to the
/// bounds (reaching them exactly once floating-point precision saturates),
/// values near zero pass through almost unchanged. `gain` scales the input
/// before compression; the usual choice is a value that maps the
/// oscillator's typical range to roughly `[-2, 2]` before the transform
/// sharpens it. This formula is mathematically `tanh(gain * x)`; it is
/// computed via [`Float::tanh`] rather than the raw exponential to avoid
/// overflow for large `gain * x`.
///
/// Stateless: the output only depends on the chained view's current value.
///
/// Reference: John Ehlers, "Using The Fisher Transform",
/// <http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf>
#[derive(Debug, Clone)]
pub struct InverseFisherTransform<T, V> {
    view: V,
    gain: T,
}

impl<T, V> InverseFisherTransform<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View` and a gain factor.
    ///
    /// # Panics
    /// Panics if `gain` is not finite or not strictly positive.
    pub fn new(view: V, gain: T) -> Self {
        assert!(gain.is_finite(), "gain must be finite");
        assert!(gain > T::zero(), "gain must be strictly positive");
        Self { view, gain }
    }

    /// The gain factor applied to the chained view's value before compression.
    #[inline]
    #[must_use]
    pub fn gain(&self) -> T {
        self.gain
    }
}

impl<T, V> View<T> for InverseFisherTransform<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
    }

    fn last(&self) -> Option<T> {
        self.view.last().map(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
            let out = (self.gain * v).tanh();
            debug_assert!(out.is_finite(), "value must be finite");
            debug_assert!(
                out >= -T::one() && out <= T::one(),
                "output must be in [-1, 1]"
            );
            out
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

    #[test]
    fn zero_maps_to_zero() {
        let mut ift = InverseFisherTransform::new(Echo::new(), 1.0);
        ift.update(0.0);
        assert_eq!(ift.last(), Some(0.0));
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(13);
        const GAIN: f64 = 1.5;
        let mut ift = InverseFisherTransform::new(Echo::new(), GAIN);
        for _ in 0..300 {
            let v = r.f64() * 10.0 - 5.0;
            ift.update(v);
            let x = 2.0 * GAIN * v;
            let expected = (x.exp() - 1.0) / (x.exp() + 1.0);
            let got = ift.last().expect("stateless, always ready");
            assert!((expected - got).abs() < 1e-9, "{expected} vs {got}");
        }
    }

    #[test]
    fn output_is_bounded_in_unit_interval_on_real_data() {
        let mut ift = InverseFisherTransform::new(Echo::new(), 2.0);
        for v in &TEST_DATA {
            ift.update(*v);
            let out = ift.last().expect("stateless, always ready");
            assert!((-1.0..=1.0).contains(&out), "out of range: {out}");
        }
    }

    #[test]
    fn larger_gain_pushes_output_closer_to_the_bound() {
        let mut low_gain = InverseFisherTransform::new(Echo::new(), 0.5);
        let mut high_gain = InverseFisherTransform::new(Echo::new(), 3.0);
        low_gain.update(1.0);
        high_gain.update(1.0);
        let low = low_gain.last().expect("stateless, always ready");
        let high = high_gain.last().expect("stateless, always ready");
        assert!(high > low, "{high} should exceed {low}");
    }

    #[test]
    #[should_panic(expected = "gain must be strictly positive")]
    fn zero_gain_panics() {
        let _ = InverseFisherTransform::<f64, _>::new(Echo::new(), 0.0);
    }

    #[test]
    #[should_panic(expected = "gain must be finite")]
    fn non_finite_gain_panics() {
        let _ = InverseFisherTransform::<f64, _>::new(Echo::new(), f64::NAN);
    }

    #[test]
    fn inverse_fisher_transform_plot() {
        let mut ift = InverseFisherTransform::new(Echo::new(), 0.1);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            ift.update(*v);
            if let Some(val) = ift.last() {
                out.push(val);
            }
        }
        let filename = "img/inverse_fisher_transform.png";
        plot_values(out, filename).unwrap();
    }
}

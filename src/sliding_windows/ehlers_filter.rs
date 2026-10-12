use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// John Ehlers nonlinear "Ehlers Filter".
///
/// A nonlinear FIR filter whose weights are the squared distances (momentum)
/// of each smoothed value to its own predecessors. Values far from the recent
/// past get large weights, so the filter jumps to new price levels quickly,
/// while it averages quietly in sideways markets.
///
/// Reference: John Ehlers, "Ehlers Filters" (2001),
/// <https://www.mesasoftware.com/papers/EhlersFilters.pdf>:
///
/// ```text
/// Smooth  = (x + 2x[1] + 2x[2] + x[3]) / 6
/// Coef[i] = sum_{k=1}^{L} (Smooth[i] - Smooth[i + k])^2,   i = 0..L-1
/// Filt    = sum_i Coef[i] Smooth[i] / sum_i Coef[i]
/// ```
///
/// where `L` is `window_len`. The weights are non-negative, so the output is
/// a convex combination of the last `L` smoothed values.
///
/// Deviation: when all weights are 0 (the smoothed input was constant for
/// `2L` bars), the published code keeps the previous output (`If SumCoef <> 0`);
/// here the output falls back to the plain average of the last `L` smoothed
/// values, which then equals that constant.
/// `last()` is `None` for the first `2 window_len + 2` values (3 for the
/// smoother, `2L` smoothed values for `L` full weights).
///
/// Output range: within the min and max of the last `2 window_len + 3` inputs.
/// Constant input yields the constant.
/// Complexity: O(window_len) per update (each new weight is a sum over
/// `window_len` distances; the weights of older values are cached),
/// O(window_len) memory. An O(1) form via running sums of `Smooth` and
/// `Smooth^2` exists but cancels catastrophically for prices far from 0,
/// especially with `f32`.
#[derive(Debug, Clone, CopyGetters)]
pub struct EhlersFilter<T, V> {
    view: V,
    /// The number of weights `L` and the lookback of each weight.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The last up to 3 inputs, newest first.
    inputs: VecDeque<T>,
    /// The last up to `L + 1` smoothed values, newest first.
    smooths: VecDeque<T>,
    /// The last up to `L` pairs of `(Coef, Smooth)`, newest first.
    weighted: VecDeque<(T, T)>,
    filt: T,
}

impl<T, V> EhlersFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Ehlers Filter with a chained `View`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        let l = window_len.get();
        Self {
            view,
            window_len,
            inputs: VecDeque::with_capacity(3),
            smooths: VecDeque::with_capacity(l + 1),
            weighted: VecDeque::with_capacity(l),
            filt: T::zero(),
        }
    }

    /// `(x + 2x[1] + 2x[2] + x[3]) / 6`, `None` until 3 earlier inputs exist.
    fn smooth(&self, val: T) -> Option<T> {
        if self.inputs.len() < 3 {
            return None;
        }
        let two = T::from(2.0).expect("can convert");
        let sum = val + two * self.inputs[0] + two * self.inputs[1] + self.inputs[2];
        Some(sum / T::from(6.0).expect("can convert"))
    }
}

/// Weighted average of `(weight, value)` pairs, or the plain average if all weights are 0.
fn weighted_average<T: Float>(pairs: &VecDeque<(T, T)>) -> T {
    debug_assert!(!pairs.is_empty(), "need at least one value");
    let (num, sum_coef, sum) = pairs.iter().fold(
        (T::zero(), T::zero(), T::zero()),
        |(num, sum_coef, sum), (coef, smooth)| {
            debug_assert!(*coef >= T::zero(), "weights are sums of squares");
            (num + *coef * *smooth, sum_coef + *coef, sum + *smooth)
        },
    );
    if sum_coef > T::zero() {
        num / sum_coef
    } else {
        sum / T::from(pairs.len()).expect("can convert")
    }
}

impl<T, V> View<T> for EhlersFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let smooth = self.smooth(val);
        self.inputs.push_front(val);
        self.inputs.truncate(3);
        let Some(smooth) = smooth else { return };

        let l = self.window_len.get();
        if self.smooths.len() == l {
            let coef = self
                .smooths
                .iter()
                .fold(T::zero(), |acc, s| acc + (smooth - *s) * (smooth - *s));
            self.weighted.push_front((coef, smooth));
            self.weighted.truncate(l);
        }
        self.smooths.push_front(smooth);
        self.smooths.truncate(l);
        if self.weighted.len() == l {
            self.filt = weighted_average(&self.weighted);
            debug_assert!(self.filt.is_finite(), "filter output must be finite");
        }
        debug_assert!(self.smooths.len() <= l);
        debug_assert!(self.weighted.len() <= l);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.weighted.len() < self.window_len.get() {
            return None;
        }
        debug_assert!(self.filt.is_finite(), "value must be finite");
        Some(self.filt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::Sma,
        test_data::TEST_DATA,
    };

    fn ef(len: usize) -> EhlersFilter<f64, Echo<f64>> {
        EhlersFilter::new(Echo::new(), NonZeroUsize::new(len).unwrap())
    }

    /// Direct transcription of the paper's double loop over full price history.
    fn reference(prices: &[f64], l: usize) -> Option<f64> {
        let n = prices.len();
        let smooth = |i: usize| -> f64 {
            // `i` bars ago.
            let p = |k: usize| prices[n - 1 - i - k];
            (p(0) + 2.0 * p(1) + 2.0 * p(2) + p(3)) / 6.0
        };
        if n < 2 * l + 3 {
            return None;
        }
        let (mut num, mut sum_coef) = (0.0, 0.0);
        for count in 0..l {
            let mut d2 = 0.0;
            for lookback in 1..=l {
                d2 += (smooth(count) - smooth(count + lookback)).powi(2);
            }
            num += d2 * smooth(count);
            sum_coef += d2;
        }
        Some(num / sum_coef)
    }

    #[test]
    fn ehlers_filter_plot() {
        let mut f = ef(15);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(val) = f.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/ehlers_filter.png").unwrap();
    }

    #[test]
    fn ehlers_filter_warmup() {
        let mut f = ef(5);
        assert_eq!(f.window_len().get(), 5);
        for i in 0..12 {
            f.update(i as f64);
            assert_eq!(f.last(), None, "bar {i}");
        }
        f.update(12.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn ehlers_filter_matches_paper_code() {
        for l in [1, 3, 15] {
            let mut f = ef(l);
            for i in 0..TEST_DATA.len() {
                f.update(TEST_DATA[i]);
                let want = reference(&TEST_DATA[..=i], l);
                assert_eq!(f.last().is_some(), want.is_some(), "len {l} bar {i}");
                if let (Some(got), Some(want)) = (f.last(), want) {
                    assert!(
                        (got - want).abs() < 1e-9,
                        "len {l} bar {i}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn ehlers_filter_constant_input_uses_fallback() {
        let mut f = ef(10);
        for _ in 0..100 {
            f.update(42.0);
        }
        assert_eq!(f.last(), Some(42.0));
    }

    #[test]
    fn ehlers_filter_output_within_input_range() {
        let rng = romu::Rng::from_seed_with_64bit(13);
        let l = 8;
        let mut f = ef(l);
        let mut vals: Vec<f64> = Vec::new();
        for _ in 0..2_000 {
            let v = 100.0 + 10.0 * rng.f64();
            vals.push(v);
            f.update(v);
            let Some(out) = f.last() else { continue };
            let recent = &vals[vals.len() - (2 * l + 3)..];
            let lo = recent.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(out >= lo - 1e-9, "{out} < {lo}");
            assert!(out <= hi + 1e-9, "{out} > {hi}");
        }
    }

    #[test]
    fn ehlers_filter_follows_steps_faster_than_sma() {
        let l = 10;
        // Five bars after a step, halfway through the SMA's transition.
        let mut f = ef(l);
        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(l).unwrap());
        for i in 0..55 {
            let v = if i < 50 { 100.0 } else { 110.0 };
            f.update(v);
            sma.update(v);
        }
        let (e, s) = (f.last().unwrap(), sma.last().unwrap());
        assert!(e > s + 2.0, "ehlers {e}, sma {s}");
        assert!(e <= 110.0);
    }
}

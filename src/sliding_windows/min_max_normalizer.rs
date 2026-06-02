//! A sliding Min - Max Normalizer

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// A sliding Min - Max Normalizer
///
/// Normalizes values to the [-1, 1] range using the min and max of a sliding
/// window of *past* values.  The current value is intentionally excluded from
/// the normalization window to avoid lookahead / data-leakage bias.
///
/// No output is emitted until the sliding window is completely filled.
/// During warm-up `last()` returns `None`.
#[derive(Clone, Debug, CopyGetters)]
pub struct MinMaxNormalizer<T, V> {
    view: V,
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    min: T,
    max: T,
    out: Option<T>,
    init: bool,
}

impl<T, V> MinMaxNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View
    /// and a given sliding window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        MinMaxNormalizer {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            min: T::zero(),
            max: T::zero(),
            out: None,
            init: true,
        }
    }
}

fn extent_queue<T: Float>(q: &VecDeque<T>) -> (T, T) {
    let mut min = *q.front().unwrap();
    let mut max = *q.front().unwrap();

    for i in 1..q.len() {
        let val = *q.get(i).unwrap();
        if val > max {
            max = val;
        }
        if val < min {
            min = val;
        }
    }

    (min, max)
}

impl<T, V> View<T> for MinMaxNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(view_last) = self.view.last() else {
            return;
        };
        debug_assert!(view_last.is_finite(), "value must be finite");

        if self.init {
            self.init = false;
            self.min = view_last;
            self.max = view_last;
            self.q_vals.push_back(view_last);
            // Warm-up: first value goes into the queue but no output yet.
            // We need window_len values before min/max are meaningful.
            return;
        }

        // Only emit output once the window is full.
        // Up to this point `out` remains None.
        if self.q_vals.len() >= self.window_len.get() {
            // Normalize the incoming value against the *previous* window's
            // min/max. This avoids lookahead bias.
            if self.min == self.max {
                self.out = Some(T::zero());
            } else {
                self.out = Some(
                    -T::one()
                        + ((view_last - self.min) * T::from(2.0).expect("can convert"))
                            / (self.max - self.min),
                );
            }
            debug_assert!(self.out.unwrap().is_finite(), "output must be finite");

            // Slide the window.
            let old = self.q_vals.pop_front().expect("Its checked above that the length is >= the non-zero window length, therefore this must be `Some`");
            self.q_vals.push_back(view_last);

            if old <= self.min || old >= self.max {
                let (min, max) = extent_queue(&self.q_vals);
                self.min = min;
                self.max = max;
            } else {
                if view_last > self.max {
                    self.max = view_last;
                }
                if view_last < self.min {
                    self.min = view_last;
                }
            }
        } else {
            // Still filling the window — no output yet.
            self.q_vals.push_back(view_last);
            if view_last > self.max {
                self.max = view_last;
            }
            if view_last < self.min {
                self.min = view_last;
            }
        }
    }

    fn last(&self) -> Option<T> {
        self.out
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
    fn normalizer() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            n.update(*v);
            if let Some(last) = n.last() {
                assert!(last.is_finite());
            }
        }
    }

    #[test]
    fn min_max_normalizer_plot() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            n.update(*v);
            if let Some(val) = n.last() {
                out.push(val);
            }
        }
        let filename = "img/min_max_normalizer.png";
        plot_values(out, filename).unwrap();
    }

    // ── Warm-up tests ──

    /// Outputs must be suppressed until the sliding window is fully filled.
    #[test]
    fn min_max_normalizer_warmup() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        // First 5 updates: window not yet full → None.
        // (The 1st goes through init, 2nd–5th have <N in queue.)
        for i in 0..5 {
            n.update(i as f64);
            assert!(n.last().is_none(), "warmup step {i}: expected None");
        }
        // 6th update has a full window → first output.
        n.update(5.0);
        assert!(
            n.last().is_some(),
            "first output after warmup should be Some"
        );
    }

    // ── Data-leakage / lookahead-bias tests ──

    /// The normalization of the current value must only use *past* values.
    /// When a spike arrives, the spike itself must not widen the min/max range
    /// used to normalize it — that would be lookahead bias.
    #[test]
    fn min_max_normalizer_no_lookahead_on_spike() {
        // Window = 3.
        // After warmup (3 values), the history is [10, 10, 10].
        // The 4th value (spike of 100) is normalized against [10,10,10]
        // (min=10, max=10 → output 0).
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        // Warm up: 3 values fill the window, output is None for all.
        for _ in 0..3 {
            n.update(10.0);
            assert!(n.last().is_none(), "warmup should suppress output");
        }

        // 4th value: first real normalization.
        n.update(100.0);
        let got = n.last().unwrap();
        // Leakage would give ≈ 1.0.  Correct causal output is 0.0.
        assert!(
            got.abs() < 1e-12,
            "lookahead bias detected: spike widened its own range, got {got}, expected 0.0"
        );
    }

    /// After the spike leaves the window, the normalizer should recover.
    #[test]
    fn min_max_normalizer_recovers_after_spike_leaves_window() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        // Warm up (3 updates, all output None).
        for _ in 0..3 {
            n.update(10.0);
        }

        // Spike (4th update — first real output, normalized against [10,10,10] → 0).
        n.update(100.0);
        let _ = n.last();

        // Push spike through the window: three more 10s are needed
        // (100 moves to index 2→1→0→popped).
        n.update(10.0);
        let _ = n.last();
        n.update(10.0);
        let _ = n.last();
        n.update(10.0);
        let _ = n.last();

        // Now the window is [10, 10, 10], min=max=10, output=0.
        n.update(10.0);
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "after spike left window, expected 0.0, got {got}"
        );
    }

    /// A steadily rising sequence should produce consistent normalization.
    #[test]
    fn min_max_normalizer_rising_sequence() {
        let window = 4;
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(window).unwrap());
        let values: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        // [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for v in &values {
            n.update(*v);
        }

        // After all updates, the last value (100) was normalized against
        // the previous window [60, 70, 80, 90] (min=60, max=90).
        // normalized(100) = -1 + 2*(100-60)/(90-60) = -1 + 2*40/30 = 5/3
        let got = n.last().unwrap();
        let expected = 5.0 / 3.0;
        let diff = (got - expected).abs();
        assert!(
            diff < 1e-12,
            "rising sequence last: expected {expected}, got {got}, diff {diff}"
        );
    }

    // ── Stale min/max bug tests ──

    /// When the singular minimum leaves the window, the min must be recalculated.
    #[test]
    fn min_max_normalizer_min_recalculated_when_singular_min_leaves() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(4).unwrap());

        // Fill the window to warm up.
        n.update(1.0);
        n.update(100.0);
        n.update(100.0);
        n.update(100.0);
        // Queue: [1, 100, 100, 100], min=1, max=100.

        // Next 100: pops 1. This is normalized against [1,100,100,100] → 1.0.
        n.update(100.0);
        let _ = n.last();
        // Queue now: [100, 100, 100, 100].

        // Next 100: normalized against [100,100,100,100] → min=max=100 → 0.
        n.update(100.0);
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "after singular min left + one more update, expected 0.0, got {got}"
        );
    }

    /// Symmetric test: when the singular maximum leaves, max must be recalculated.
    #[test]
    fn min_max_normalizer_max_recalculated_when_singular_max_leaves() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(4).unwrap());

        // Fill to warm up.
        n.update(100.0);
        n.update(1.0);
        n.update(1.0);
        n.update(1.0);
        // Queue: [100, 1, 1, 1], min=1, max=100.

        // Next 1: pops 100. Normalized against [100,1,1,1] → -1.0.
        n.update(1.0);
        let _ = n.last();
        // Queue now: [1, 1, 1, 1].

        // Next 1: normalized against [1,1,1,1] → min=max=1 → 0.
        n.update(1.0);
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "after singular max left + one more update, expected 0.0, got {got}"
        );
    }

    // ── General correctness tests ──

    /// When the window contains identical values, output is always 0.
    #[test]
    fn min_max_normalizer_identical_values_yield_zero() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        // Warm up: first 5 values produce None.
        for _ in 0..5 {
            n.update(42.0);
            assert!(n.last().is_none(), "warmup should suppress output");
        }
        // From the 6th onward, every output should be 0.
        for _ in 0..20 {
            n.update(42.0);
            assert!(
                n.last().unwrap().abs() < 1e-12,
                "identical values should yield normalized output 0"
            );
        }
    }

    /// Values at exactly the min normalize to -1.0; at exactly the max to +1.0.
    #[test]
    fn min_max_normalizer_bounds() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        // Warm up the window.
        n.update(0.0);
        n.update(100.0);
        n.update(100.0);
        // Queue: [0, 100, 100], min=0, max=100.
        // After 3rd update we now have a full window, first output coming next.

        // Update with a value at the min.
        n.update(0.0);
        let got_min = n.last().unwrap();
        assert!(
            (got_min - (-1.0)).abs() < 1e-12,
            "expected -1.0, got {got_min}"
        );

        // Queue is now [100, 100, 0], min=0, max=100.
        // Update with a value at the max.
        n.update(100.0);
        let got_max = n.last().unwrap();
        assert!((got_max - 1.0).abs() < 1e-12, "expected 1.0, got {got_max}");
    }

    /// Ensure MinMaxNormalizer works when chained after another View.
    #[test]
    fn min_max_normalizer_chained() {
        use crate::sliding_windows::Ema;
        let mut n = MinMaxNormalizer::new(
            Ema::new(Echo::new(), NonZeroUsize::new(5).unwrap()),
            NonZeroUsize::new(8).unwrap(),
        );
        for v in &TEST_DATA {
            n.update(*v);
            if let Some(val) = n.last() {
                assert!(
                    val.is_finite(),
                    "chained output should be finite, got {val}"
                );
            }
        }
    }

    /// The normalizer should handle the case where window_len == 1.
    #[test]
    fn min_max_normalizer_window_len_one() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(1).unwrap());

        // First update goes through init, no output yet.
        n.update(5.0);
        assert!(n.last().is_none(), "first update: init, no output");

        // Second update: queue has 1 value, window full → output.
        n.update(10.0);
        // Normalize 10 against [5] → min=max=5 → 0.0
        assert_eq!(n.last().unwrap(), 0.0);
    }

    /// Verify the normalizer produces finite outputs for all test data.
    #[test]
    fn min_max_normalizer_all_outputs_finite() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            n.update(*v);
            if let Some(out) = n.last() {
                assert!(out.is_finite(), "output should be finite, got {out}");
            }
        }
    }

    /// Regression: the normalizer should not panic and should produce
    /// valid output when fed many values.
    #[test]
    fn min_max_normalizer_stress_test() {
        let mut n = MinMaxNormalizer::new(Echo::new(), NonZeroUsize::new(64).unwrap());
        for i in 0..1000 {
            let val = (i as f64).sin();
            n.update(val);
            if let Some(out) = n.last() {
                assert!(out.is_finite(), "output should be finite, got {out}");
            }
        }
    }
}

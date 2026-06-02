//! A sliding High - Low Normalizer

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// A sliding High - Low Normalizer
///
/// Normalizes values to the [-1, 1] range using the min and max of a sliding
/// window of *past* values.  The current value is intentionally excluded from
/// the normalization window to avoid lookahead / data-leakage bias.
#[derive(Clone, Debug, CopyGetters)]
pub struct HLNormalizer<T, V> {
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

impl<T, V> HLNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new HLNormalizer with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        HLNormalizer {
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

impl<T, V> View<T> for HLNormalizer<T, V>
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
            self.out = Some(T::zero()); // single value → min==max → 0
            self.q_vals.push_back(view_last);
            return;
        }

        // Normalize the incoming value against the *previous* window's min/max.
        // This avoids lookahead bias — the current value does not widen its own
        // normalization range.
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

        // Now update the sliding window and min/max to include the new value.
        if self.q_vals.len() >= self.window_len.get() {
            // Pop the oldest value first, then push the new one.
            let old = self.q_vals.pop_front().expect("Has some value");
            self.q_vals.push_back(view_last);

            if old <= self.min || old >= self.max {
                // The removed value was at a boundary — full rescan needed.
                let (min, max) = extent_queue(&self.q_vals);
                self.min = min;
                self.max = max;
            } else {
                // Old value was interior — only the new value can shift bounds.
                if view_last > self.max {
                    self.max = view_last;
                }
                if view_last < self.min {
                    self.min = view_last;
                }
            }
        } else {
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
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            n.update(*v);
            let last = n.last().unwrap();
            assert!(last.is_finite());
        }
    }

    #[test]
    fn hl_normalizer_plot() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last().unwrap());
        }
        let filename = "img/hl_normalizer.png";
        plot_values(out, filename).unwrap();
    }

    // ── Data-leakage / lookahead-bias tests ──

    /// The normalization of the current value must only use *past* values.
    /// When a spike arrives, the spike itself must not widen the min/max range
    /// used to normalize it — that would be lookahead bias.
    #[test]
    fn hl_normalizer_no_lookahead_on_spike() {
        // Feed: steady 10s, then a spike of 100, then back to 10.
        // Window = 3.
        //
        // Before the spike the history is [10, 10, 10]  →  min=10, max=10.
        // Normalizing 100 against a range of zero *should* yield 0
        // (since min==max → division-by-zero → return 0).
        // If the implementation leaks 100 into its own normalization window
        // the range becomes [10, 100] and the output is 1.0 — a sign of leakage.
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        for _ in 0..3 {
            n.update(10.0);
            let _ = n.last();
        }

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
    fn hl_normalizer_recovers_after_spike_leaves_window() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        // Warm up
        for _ in 0..3 {
            n.update(10.0);
        }

        // Spike
        n.update(100.0);
        let _ = n.last();

        // Push spike through the window: three more 10s are needed because
        // the output is normalized against the *previous* window, so after
        // the spike leaves the queue the *next* update is the first one
        // normalized against a clean window.
        n.update(10.0);
        let _ = n.last();
        n.update(10.0);
        let _ = n.last();
        n.update(10.0);
        let _ = n.last();

        // Now the window should be [10, 10, 10] again,
        // so normalization of the next 10 should be 0.
        n.update(10.0);
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "after spike left window, expected 0.0, got {got}"
        );
    }

    /// A steadily rising sequence should produce consistent normalization.
    /// For a strictly increasing sequence with window=N, each new value
    /// should be the new max and the oldest value is the min.
    /// With a causal window (current value excluded), the new value is always
    /// one step ahead of the max, giving a consistent ratio.
    #[test]
    fn hl_normalizer_rising_sequence() {
        let window = 4;
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(window).unwrap());
        let values: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        // [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for v in &values {
            n.update(*v);
        }

        // After all updates, the last value (100) was normalized against
        // the previous window [60, 70, 80, 90] (min=60, max=90).
        // normalized(100) = -1 + 2*(100-60)/(90-60) = -1 + 2*40/30 = 1.666...
        let got = n.last().unwrap();
        let expected = 5.0 / 3.0; // ≈ 1.666...
        let diff = (got - expected).abs();
        assert!(
            diff < 1e-12,
            "rising sequence last: expected {expected}, got {got}, diff {diff}"
        );
    }

    // ── Stale min/max bug tests ──

    /// When the singular minimum leaves the window, the min must be recalculated.
    /// The *next* update after the min leaves must see the recomputed min.
    #[test]
    fn hl_normalizer_min_recalculated_when_singular_min_leaves() {
        // Window = 4.
        // Fill: [1, 100, 100, 100]  → min=1, max=100
        // Next: 100  → 1 should leave, leaving [100, 100, 100, 100]
        // (This step still normalizes against the old window → output 1.0)
        // Next: 100  → now normalized against [100,100,100,100] → min=max=100 → 0
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(4).unwrap());

        n.update(1.0);
        for _ in 0..4 {
            n.update(100.0);
        }
        // 5th update (4th 100) pops the 1; normalization was against [1,100,100,100].
        // The stale min cleared but output reflects the old window. One more:
        n.update(100.0);
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "stale min bug: after singular min left + one more update, expected 0.0, got {got}"
        );
    }

    /// Symmetric test: when the singular maximum leaves, max must be recalculated.
    #[test]
    fn hl_normalizer_max_recalculated_when_singular_max_leaves() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(4).unwrap());

        n.update(100.0);
        for _ in 0..5 {
            n.update(1.0);
        }

        // After 6 total updates:
        // - 5th update (4th 1.0) popped 100 from queue
        // - 6th update (5th 1.0) gets normalized against [1,1,1,1] → 0
        let got = n.last().unwrap();
        assert!(
            got.abs() < 1e-12,
            "stale max bug: after singular max left + one more update, expected 0.0, got {got}"
        );
    }

    // ── General correctness tests ──

    /// When the window contains identical values, output is always 0.
    #[test]
    fn hl_normalizer_identical_values_yield_zero() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..20 {
            n.update(42.0);
            assert!(
                n.last().unwrap().abs() < 1e-12,
                "identical values should yield normalized output 0"
            );
        }
    }

    /// Values at exactly the min normalize to -1.0; at exactly the max to +1.0.
    /// Because we normalize against the *previous* window, we need to fill the
    /// window first, then test with values at the boundaries.
    #[test]
    fn hl_normalizer_bounds() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(3).unwrap());

        // Fill the window with range [0, 100]
        n.update(0.0);
        n.update(100.0);
        n.update(100.0);

        // Now queue is [0, 100, 100], min=0, max=100.
        // Update with a value at the min.
        n.update(0.0);
        let got_min = n.last().unwrap();
        assert!(
            (got_min - (-1.0)).abs() < 1e-12,
            "expected -1.0, got {got_min}"
        );

        // After that update, queue is [100, 100, 0], min=0, max=100.
        // Update with a value at the max.
        n.update(100.0);
        let got_max = n.last().unwrap();
        assert!((got_max - 1.0).abs() < 1e-12, "expected 1.0, got {got_max}");
    }

    /// Ensure HLNormalizer works when chained after another View.
    #[test]
    fn hl_normalizer_chained() {
        use crate::sliding_windows::Ema;
        let mut n = HLNormalizer::new(
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
    fn hl_normalizer_window_len_one() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(1).unwrap());

        n.update(5.0);
        assert_eq!(n.last().unwrap(), 0.0); // only one value → min==max

        n.update(10.0);
        // Window is [10], min=max=10 → 0.0
        assert_eq!(n.last().unwrap(), 0.0);
    }

    /// Verify the normalizer produces finite outputs for all test data.
    #[test]
    fn hl_normalizer_all_outputs_finite() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            n.update(*v);
            let out = n.last().unwrap();
            assert!(out.is_finite(), "output should be finite, got {out}");
        }
    }

    /// Regression: the normalizer should not panic and should produce
    /// valid output when fed many values.
    #[test]
    fn hl_normalizer_stress_test() {
        let mut n = HLNormalizer::new(Echo::new(), NonZeroUsize::new(64).unwrap());
        for i in 0..1000 {
            let val = (i as f64).sin();
            n.update(val);
            let out = n.last().unwrap();
            assert!(out.is_finite(), "output should be finite, got {out}");
        }
    }
}

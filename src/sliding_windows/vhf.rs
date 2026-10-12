//! Vertical Horizontal Filter

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use num::Float;

use crate::View;

/// Vertical Horizontal Filter (VHF), by Adam White.
///
/// Measures whether a market is trending or ranging: the absolute range of
/// the last `window_len` samples of the chained view, divided by the sum of
/// their `window_len` one-step absolute changes. A high value indicates a
/// strong trend (the path moved mostly in one direction), a low value
/// indicates a ranging market (the path moved back and forth). Output is
/// non-negative and unbounded above, though in practice it rarely exceeds 1.
///
/// When every sample in the window is equal, both the range and the sum of
/// changes are zero; this is defined as `VHF = 0` (no trend), rather than
/// left undefined.
///
/// Reference: <https://www.investopedia.com/terms/v/verticalhorizontalfilter.asp>
///
/// # Complexity
/// O(`window_len`) per update: the range and the sum of changes are
/// recomputed from the raw window buffer every step.
#[derive(Debug, Clone)]
pub struct Vhf<T, V> {
    view: V,
    window_len: NonZeroUsize,
    buf: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Vhf<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    ///
    /// # Panics
    /// Panics if `window_len < 2`, since at least two samples are needed to
    /// measure a net change.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(
            window_len.get() >= 2,
            "window_len must be at least 2, got {}",
            window_len.get()
        );
        Self {
            view,
            window_len,
            // one extra sample beyond the window: the oldest one is only
            // used to compute the oldest net change, not the high/low range.
            buf: VecDeque::with_capacity(window_len.get() + 1),
            out: None,
        }
    }

    /// The sliding window length.
    #[inline]
    #[must_use]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }
}

impl<T, V> View<T> for Vhf<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let capacity = self.window_len.get() + 1;
        if self.buf.len() >= capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(val);

        if self.buf.len() == capacity {
            self.out = Some(vhf(&self.buf));
        }
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out.inspect(|out| {
            debug_assert!(out.is_finite(), "value must be finite");
            debug_assert!(*out >= T::zero(), "VHF must be non-negative");
        })
    }
}

/// Vertical Horizontal Filter over a buffer of `window_len + 1` raw samples:
/// the high/low range of the newest `window_len` samples, divided by the sum
/// of all `window_len` consecutive one-step absolute changes in the buffer.
fn vhf<T: Float>(buf: &VecDeque<T>) -> T {
    assert!(
        buf.len() >= 3,
        "buffer must hold window_len + 1 >= 3 samples"
    );

    let mut hcp = T::neg_infinity();
    let mut lcp = T::infinity();
    for v in buf.iter().skip(1) {
        hcp = hcp.max(*v);
        lcp = lcp.min(*v);
    }
    let range = hcp - lcp;
    debug_assert!(range >= T::zero(), "range must be non-negative");

    let sum_changes = buf
        .iter()
        .zip(buf.iter().skip(1))
        .fold(T::zero(), |sum, (a, b)| sum + (*b - *a).abs());
    debug_assert!(
        sum_changes >= T::zero(),
        "sum of changes must be non-negative"
    );

    if sum_changes <= T::zero() {
        return T::zero();
    }
    range / sum_changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    /// Naive O(n) reference, operating on the same `window_len + 1` slice
    /// convention as the indicator under test.
    fn naive_vhf(vals: &[f64]) -> f64 {
        let hcp = vals[1..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lcp = vals[1..].iter().copied().fold(f64::INFINITY, f64::min);
        let sum_changes: f64 = vals.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        if sum_changes <= 0.0 {
            return 0.0;
        }
        (hcp - lcp) / sum_changes
    }

    #[test]
    fn warmup_then_ready() {
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(6.0);
        assert!(v.last().is_some());
    }

    #[test]
    fn constant_input_is_zero() {
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(8).unwrap());
        for _ in 0..20 {
            v.update(7.0);
        }
        assert_eq!(v.last(), Some(0.0));
    }

    #[test]
    fn monotonic_trend_approaches_one() {
        let window_len = 10;
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        for i in 0..20 {
            v.update(f64::from(i));
        }
        // A strictly monotonic unit-step path has range = window_len - 1 (it
        // spans window_len samples) and sum of changes = window_len (it sums
        // window_len unit steps across window_len + 1 samples), so the
        // ratio approaches but never reaches 1.
        let expected = (window_len - 1) as f64 / window_len as f64;
        assert!((v.last().expect("warmed up") - expected).abs() < 1e-9);
    }

    #[test]
    fn output_is_non_negative_on_real_data() {
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(out) = v.last() {
                assert!(out >= 0.0, "must be non-negative: {out}");
            }
        }
    }

    #[test]
    fn matches_naive_reference_on_random_data() {
        let r = romu::Rng::from_seed_with_64bit(55);
        let vals: Vec<f64> = (0..300).map(|_| r.f64() * 50.0 - 25.0).collect();
        const WINDOW_LEN: usize = 14;
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(WINDOW_LEN).unwrap());
        for (i, x) in vals.iter().enumerate() {
            v.update(*x);
            if i + 1 < WINDOW_LEN + 1 {
                continue;
            }
            let expected = naive_vhf(&vals[i + 1 - (WINDOW_LEN + 1)..=i]);
            let got = v.last().expect("warmed up");
            assert!((expected - got).abs() < 1e-9, "at {i}: {expected} vs {got}");
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 2")]
    fn window_len_one_panics() {
        let _ = Vhf::<f64, _>::new(Echo::new(), NonZeroUsize::new(1).unwrap());
    }

    #[test]
    fn vhf_plot() {
        let mut v = Vhf::new(Echo::new(), NonZeroUsize::new(28).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        let filename = "img/vhf.png";
        plot_values(out, filename).unwrap();
    }
}

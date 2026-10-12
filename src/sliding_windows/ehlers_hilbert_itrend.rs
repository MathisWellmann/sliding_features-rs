//! Ehlers' instantaneous trendline, shared by `InstantaneousTrendline` and `HilbertTrendMode`.
//! Reference: "Rocket Science for Traders" (2001), and TA-Lib's `HT_TRENDLINE`.

use num::Float;

use super::{
    ehlers_hilbert_core::c,
    ehlers_hilbert_dc_phase::{
        SIZE,
        dominant_cycle_bars,
    },
};

/// Sum of the last `bars` values (newest first, ending at `head` in the ring buffer `buf`).
#[inline(always)]
fn sum_last<T: Float>(buf: &[T; SIZE], head: usize, bars: usize) -> T {
    debug_assert!(head < SIZE, "head out of bounds");
    debug_assert!(bars <= SIZE, "bars out of bounds");
    (0..bars).fold(T::zero(), |acc, i| acc + buf[(head + SIZE - i) % SIZE])
}

/// Streaming instantaneous trendline: the simple average of the raw price over one
/// dominant cycle (which cancels that cycle), smoothed by a 4-bar WMA.
/// O(dominant cycle) <= O(50) per update.
#[derive(Debug, Clone)]
pub(super) struct ITrend<T> {
    /// Ring buffer of raw prices.
    prices: [T; SIZE],
    /// Index of the newest value in `prices`.
    head: usize,
    n_updates: usize,
    /// Previous three cycle averages, newest first.
    averages: [T; 3],
    trendline: T,
}

impl<T: Float> ITrend<T> {
    pub(super) fn new() -> Self {
        Self {
            prices: [T::zero(); SIZE],
            head: 0,
            n_updates: 0,
            averages: [T::zero(); 3],
            trendline: T::zero(),
        }
    }

    /// Feed the raw price and the current smoothed dominant cycle period.
    pub(super) fn update(&mut self, price: T, smooth_period: T) {
        debug_assert!(price.is_finite(), "value must be finite");
        debug_assert!(
            smooth_period >= T::zero(),
            "smooth period must not be negative"
        );
        if self.n_updates == 0 {
            // Pre-fill instead of zeros, so the average never mixes in an artificial zero level.
            self.prices = [price; SIZE];
        }
        self.n_updates = self.n_updates.saturating_add(1);
        self.head = (self.head + 1) % SIZE;
        self.prices[self.head] = price;

        let bars = dominant_cycle_bars(smooth_period);
        // TA-Lib reports a zero average for a zero length cycle, which only happens in warm-up.
        let avg = if bars > 0 {
            sum_last(&self.prices, self.head, bars) / T::from(bars).expect("can convert")
        } else {
            T::zero()
        };
        let [a1, a2, a3] = self.averages;
        self.trendline = (c::<T>(4.0) * avg + c::<T>(3.0) * a1 + c::<T>(2.0) * a2 + a3) / c(10.0);
        self.averages = [avg, a1, a2];
        debug_assert!(self.trendline.is_finite(), "trendline must be finite");
    }

    /// The latest trendline value.
    #[inline(always)]
    pub(super) fn trendline(&self) -> T {
        self.trendline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ehlers_hilbert_itrend_sum_last_wraps() {
        let mut buf = [0.0; SIZE];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = i as f64;
        }
        assert_eq!(sum_last(&buf, 2, 0), 0.0);
        assert_eq!(sum_last(&buf, 2, 3), 2.0 + 1.0);
        // Wraps around to the end of the buffer.
        assert_eq!(sum_last(&buf, 1, 4), 1.0 + 0.0 + 49.0 + 48.0);
        assert_eq!(sum_last(&buf, 7, SIZE), (0..SIZE).sum::<usize>() as f64);
    }

    #[test]
    fn ehlers_hilbert_itrend_cycle_average() {
        let mut it = ITrend::new();
        // A 4 bar cycle averages to its mean over a 4 bar window.
        for i in 0..20 {
            it.update([1.0, 2.0, 3.0, 2.0][i % 4], 4.0);
        }
        assert!((it.trendline() - 2.0).abs() < 1e-12);
    }
}

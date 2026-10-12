//! Ulcer Index over a sliding window of prices.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Peter Martin's Ulcer Index: the root mean square of the percentage drawdowns over the
/// last `window_len` periods, a measure of the depth and duration of drawdowns.
///
/// Reference: P. G. Martin, B. B. McCann (1989), "The Investor's Guide to Fidelity Funds";
/// classic form as in Martin's definition and StockCharts.
///
/// With the window length `n`:
/// - `R_i = 100 (P_i - max(P_{i-n+1..=i})) / max(P_{i-n+1..=i})`, the percent drawdown of
///   each price from the maximum of the `n` prices ending at it (classic: the rolling
///   window max at each point, not the max within the averaging window),
/// - `UI = sqrt(sum(R_i^2 for the last n periods) / n)`.
///
/// Input: prices (strictly positive), not returns.
/// Output: the Ulcer Index in percent (`>= 0`, `0` without drawdowns).
///
/// Warm-up: `None` until `2 n - 1` prices were observed (`n` for the first drawdown,
/// `n - 1` more for the first full average).
/// Complexity: amortized O(1) per update, a monotonic deque gives the window max and a
/// running sum gives the mean of the squared drawdowns (recomputed from the window every
/// `n` updates to bound floating point drift, and reset to exactly zero without drawdowns);
/// O(n) memory, preallocated.
#[derive(Debug, Clone, CopyGetters)]
pub struct UlcerIndex<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// `(index, price)` with decreasing prices; the front is the window max.
    maxima: VecDeque<(usize, T)>,
    /// The squared percent drawdowns of the last `window_len` periods.
    squared_drawdowns: VecDeque<T>,
    sum_squared: T,
    /// The number of non-zero drawdowns in `squared_drawdowns`.
    num_drawdowns: usize,
    count: usize,
    evictions: usize,
}

impl<T, V> UlcerIndex<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Ulcer Index with a chained `View` of prices and the window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            maxima: VecDeque::with_capacity(window_len.get()),
            squared_drawdowns: VecDeque::with_capacity(window_len.get() + 1),
            sum_squared: T::zero(),
            num_drawdowns: 0,
            count: 0,
            evictions: 0,
        }
    }

    /// Push `price` into the monotonic deque and return the max of the last `n` prices.
    fn push_price(&mut self, price: T) -> T {
        let idx = self.count;
        self.count += 1;
        if let Some((front_idx, _)) = self.maxima.front()
            && front_idx + self.window_len.get() <= idx
        {
            self.maxima.pop_front();
        }
        while self.maxima.back().is_some_and(|(_, p)| *p <= price) {
            self.maxima.pop_back();
        }
        self.maxima.push_back((idx, price));
        debug_assert!(
            self.maxima.len() <= self.window_len.get(),
            "deque is bounded"
        );
        let max = self.maxima.front().expect("just pushed").1;
        debug_assert!(max >= price, "the window max is at least the latest price");
        max
    }

    fn push_squared_drawdown(&mut self, sq: T) {
        self.squared_drawdowns.push_back(sq);
        self.sum_squared = self.sum_squared + sq;
        self.num_drawdowns += usize::from(sq > T::zero());
        if self.squared_drawdowns.len() > self.window_len.get() {
            let old = self
                .squared_drawdowns
                .pop_front()
                .expect("window is not empty");
            self.sum_squared = self.sum_squared - old;
            self.num_drawdowns -= usize::from(old > T::zero());
            self.evictions += 1;
            if self.evictions == self.window_len.get() {
                self.evictions = 0;
                self.sum_squared = self
                    .squared_drawdowns
                    .iter()
                    .fold(T::zero(), |acc, v| acc + *v);
            }
        }
        // Subtraction can leave tiny residuals, which `sqrt` would amplify into a visible
        // index after all drawdowns left the window.
        if self.num_drawdowns == 0 {
            self.sum_squared = T::zero();
        }
        self.sum_squared = self.sum_squared.max(T::zero());
    }
}

impl<T, V> View<T> for UlcerIndex<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        debug_assert!(val > T::zero(), "prices must be positive");

        let max = self.push_price(val);
        if self.count < self.window_len.get() {
            return;
        }
        let hundred = T::from(100).expect("can convert");
        let drawdown = hundred * (val - max) / max;
        debug_assert!(drawdown <= T::zero(), "drawdowns are not positive");
        self.push_squared_drawdown(drawdown * drawdown);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.squared_drawdowns.len() < self.window_len.get() {
            return None;
        }
        let n = T::from(self.window_len.get()).expect("can convert");
        let out = (self.sum_squared / n).sqrt();
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(out >= T::zero(), "the Ulcer Index is non-negative");
        Some(out)
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

    fn ui(window_len: usize) -> UlcerIndex<f64, Echo<f64>> {
        UlcerIndex::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Naive O(n^2) Ulcer Index of the last price of `prices`, `None` during warm-up.
    fn naive_ui(prices: &[f64], n: usize) -> Option<f64> {
        if prices.len() < 2 * n - 1 {
            return None;
        }
        let t = prices.len() - 1;
        let sum: f64 = (t + 1 - n..=t)
            .map(|i| {
                let max = prices[i + 1 - n..=i]
                    .iter()
                    .copied()
                    .fold(f64::MIN, f64::max);
                (100.0 * (prices[i] - max) / max).powi(2)
            })
            .sum();
        Some((sum / n as f64).sqrt())
    }

    #[test]
    fn ulcer_index_plot() {
        let mut v = ui(14);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/ulcer_index.png").unwrap();
    }

    #[test]
    fn ulcer_index_warmup_and_hand_computed() {
        let mut v = ui(2);
        assert_eq!(v.window_len().get(), 2);
        v.update(10.0);
        assert_eq!(v.last(), None);
        v.update(8.0); // R = -20
        assert_eq!(v.last(), None);
        v.update(9.0); // R = 0, as the max of [8, 9] is 9
        assert_eq!(v.last(), Some(200.0_f64.sqrt()));
        v.update(12.0); // R = 0
        assert_eq!(v.last(), Some(0.0));
        v.update(6.0); // R = -50
        assert_eq!(v.last(), Some(1250.0_f64.sqrt()));
    }

    #[test]
    fn ulcer_index_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(141);
        for n in [1, 2, 3, 14, 50] {
            let mut v = ui(n);
            let mut prices = Vec::new();
            let mut price = 100.0;
            for i in 0..2_000 {
                // Random walk with flat stretches, so equal prices occur.
                if rng.range_usize(0..4) > 0 {
                    price *= 1.0 + (rng.f64() - 0.5) * 0.04;
                }
                prices.push(price);
                v.update(price);
                match (v.last(), naive_ui(&prices, n)) {
                    (None, None) => {}
                    (Some(got), Some(expected)) => {
                        assert!((got - expected).abs() < 1e-9, "step {i}, n {n}");
                    }
                    (got, expected) => panic!("step {i}: {got:?} vs {expected:?}"),
                }
            }
        }
    }

    #[test]
    fn ulcer_index_no_drawdown() {
        let (mut rising, mut flat) = (ui(10), ui(10));
        for i in 0..100 {
            rising.update(1.0 + i as f64);
            flat.update(5.0);
        }
        assert_eq!(rising.last(), Some(0.0));
        assert_eq!(flat.last(), Some(0.0));
        let mut single = ui(1);
        single.update(3.0);
        assert_eq!(single.last(), Some(0.0));
    }

    #[test]
    #[should_panic(expected = "prices must be positive")]
    fn ulcer_index_non_positive_price() {
        let mut v = ui(3);
        v.update(0.0);
    }
}

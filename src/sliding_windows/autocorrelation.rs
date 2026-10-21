// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sample autocorrelation at a fixed lag over a sliding window.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Sample autocorrelation at lag `k` of the last `n = window_len` values, the standard
/// estimator of G. E. P. Box and G. M. Jenkins, "Time Series Analysis: Forecasting and Control"
/// (1970):
///
/// `r_k = sum_{t=k+1}^{n} (x_t - mean)(x_{t-k} - mean) / sum_{t=1}^{n} (x_t - mean)^2`
///
/// with `mean` the mean of the whole window. Unlike the Pearson correlation of the two
/// overlapping sub-series, this estimator is guaranteed to lie in `[-1, 1]`; it is biased
/// towards 0 by the factor `(n - k) / n`.
///
/// Output range `[-1, 1]`. A window with numerically zero variance has no measurable linear
/// dependence and outputs 0.
///
/// Complexity per update: O(1) amortized via running sums (`sum x`, `sum x^2`,
/// `sum x_t x_{t-k}`, and the sums of the first and last `k` values) over values shifted by a
/// reference value (against cancellation); the sums are rebuilt exactly every `window_len`
/// updates to bound floating point drift.
#[derive(Debug, Clone, CopyGetters)]
pub struct Autocorrelation<T: Float, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The lag k.
    #[getset(get_copy = "pub")]
    lag: NonZeroUsize,
    window: VecDeque<T>,
    sums: Sums<T>,
    updates_since_rebuild: usize,
}

/// Running sums over the shifted values `y = x - shift`.
#[derive(Debug, Clone, Copy)]
struct Sums<T> {
    shift: T,
    s1: T,
    s2: T,
    // sum of y_t * y_{t-k} over all pairs inside the window
    lagged: T,
    // sum of the oldest k values
    head: T,
    // sum of the newest k values
    tail: T,
}

impl<T: Float> Sums<T> {
    fn new(shift: T) -> Self {
        Self {
            shift,
            s1: T::zero(),
            s2: T::zero(),
            lagged: T::zero(),
            head: T::zero(),
            tail: T::zero(),
        }
    }

    /// Exact sums of `window` with a fresh shift, to discard accumulated rounding error.
    fn rebuild(shift: T, window: &VecDeque<T>, k: usize) -> Self {
        let y = |i: usize| window[i] - shift;
        let n = window.len();
        let mut sums = Self::new(shift);
        for i in 0..n {
            sums.s1 = sums.s1 + y(i);
            sums.s2 = sums.s2 + y(i) * y(i);
        }
        for i in k..n {
            sums.lagged = sums.lagged + y(i) * y(i - k);
        }
        sums.head = (0..k.min(n)).fold(T::zero(), |acc, i| acc + y(i));
        sums.tail = (n.saturating_sub(k)..n).fold(T::zero(), |acc, i| acc + y(i));
        sums
    }
}

impl<T, V> Autocorrelation<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `Autocorrelation` with a chained `View`.
    ///
    /// # Panics
    /// If `lag >= window_len`.
    pub fn new(view: V, window_len: NonZeroUsize, lag: NonZeroUsize) -> Self {
        assert!(
            lag.get() < window_len.get(),
            "lag must be smaller than window_len"
        );
        Self {
            view,
            window_len,
            lag,
            window: VecDeque::with_capacity(window_len.get() + 1),
            sums: Sums::new(T::zero()),
            updates_since_rebuild: 0,
        }
    }

    /// Adds the newest value (already pushed to the back of the window) to the sums.
    fn add_newest(&mut self) {
        let (k, len, shift) = (self.lag.get(), self.window.len(), self.sums.shift);
        let y = self.window[len - 1] - shift;
        let s = &mut self.sums;
        s.s1 = s.s1 + y;
        s.s2 = s.s2 + y * y;
        s.tail = s.tail + y;
        if len > k {
            let partner = self.window[len - 1 - k] - shift;
            s.tail = s.tail - partner;
            s.lagged = s.lagged + y * partner;
        } else {
            s.head = s.head + y;
        }
    }

    /// Removes the oldest value (still at the front of the window) from the sums.
    fn remove_oldest(&mut self) {
        let (k, shift) = (self.lag.get(), self.sums.shift);
        debug_assert!(self.window.len() > k + 1, "partner must stay in the window");
        let y = self.window[0] - shift;
        let partner = self.window[k] - shift;
        let s = &mut self.sums;
        s.s1 = s.s1 - y;
        s.s2 = s.s2 - y * y;
        s.lagged = s.lagged - y * partner;
        s.head = s.head - y + partner;
    }
}

/// `r_k` from the running sums, 0 for numerically zero variance.
fn autocorrelation<T: Float>(s: &Sums<T>, n: usize, k: usize) -> T {
    let nt = T::from(n).expect("can convert");
    let mean = s.s1 / nt;
    let den = s.s2 - s.s1 * mean;
    // Cancellation leaves an error of order n * eps * s2, anything below is zero variance.
    if den <= nt * T::epsilon() * s.s2 {
        return T::zero();
    }
    let pairs = T::from(n - k).expect("can convert");
    let num = s.lagged - mean * ((s.s1 - s.head) + (s.s1 - s.tail)) + pairs * mean * mean;
    // Rounding can exceed the Cauchy-Schwarz bound by a few ulps.
    (num / den).max(-T::one()).min(T::one())
}

impl<T, V> View<T> for Autocorrelation<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.window.push_back(val);
        self.add_newest();
        if self.window.len() > self.window_len.get() {
            self.remove_oldest();
            self.window.pop_front();
        }
        self.updates_since_rebuild += 1;
        if self.updates_since_rebuild >= self.window_len.get() {
            self.sums = Sums::rebuild(val, &self.window, self.lag.get());
            self.updates_since_rebuild = 0;
        }
    }

    fn last(&self) -> Option<T> {
        if self.window.len() < self.window_len.get() {
            return None;
        }
        let out = autocorrelation(&self.sums, self.window_len.get(), self.lag.get());
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(out.abs() <= T::one(), "autocorrelation is in [-1, 1]");
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

    fn new_f64(window_len: usize, lag: usize) -> Autocorrelation<f64, Echo<f64>> {
        Autocorrelation::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            NonZeroUsize::new(lag).unwrap(),
        )
    }

    fn naive(x: &[f64], k: usize) -> f64 {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let num: f64 = (k..n).map(|t| (x[t] - mean) * (x[t - k] - mean)).sum();
        let den: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
        num / den
    }

    fn ar1(seed: u64, phi: f64, n: usize) -> Vec<f64> {
        let rng = romu::Rng::from_seed_with_64bit(seed);
        let mut x = 0.0;
        (0..n)
            .map(|_| {
                x = phi * x + rng.f64() - 0.5;
                x
            })
            .collect()
    }

    #[test]
    fn autocorrelation_warmup() {
        let mut ac = new_f64(4, 1);
        for v in [1.0, 3.0, 2.0] {
            ac.update(v);
            assert_eq!(ac.last(), None);
        }
        ac.update(5.0);
        assert!(ac.last().is_some());
    }

    #[test]
    fn autocorrelation_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(47);
        for (window_len, lag) in [(2, 1), (10, 1), (10, 9), (50, 3), (97, 20)] {
            let mut ac = new_f64(window_len, lag);
            let mut vals = Vec::new();
            for _ in 0..1_000 {
                // A large offset stresses the cancellation handling.
                let v = 1_000.0 + rng.f64();
                vals.push(v);
                ac.update(v);
                if vals.len() < window_len {
                    assert!(ac.last().is_none());
                    continue;
                }
                let expected = naive(&vals[vals.len() - window_len..], lag);
                let got = ac.last().unwrap();
                assert!((got - expected).abs() < 1e-7, "{got} vs {expected}");
            }
        }
    }

    #[test]
    fn autocorrelation_ar1() {
        let x = ar1(53, 0.7, 20_000);
        let mut lag1 = new_f64(10_000, 1);
        let mut lag2 = new_f64(10_000, 2);
        for v in &x {
            lag1.update(*v);
            lag2.update(*v);
        }
        let (r1, r2) = (lag1.last().unwrap(), lag2.last().unwrap());
        assert!((r1 - 0.7).abs() < 0.05, "{r1}");
        assert!((r2 - 0.49).abs() < 0.05, "{r2}");
    }

    #[test]
    fn autocorrelation_iid_near_zero() {
        let x = ar1(59, 0.0, 20_000);
        let mut ac = new_f64(10_000, 1);
        x.iter().for_each(|v| ac.update(*v));
        let r = ac.last().unwrap();
        assert!(r.abs() < 0.05, "{r}");
    }

    #[test]
    fn autocorrelation_alternating_is_negative_one() {
        let mut ac = new_f64(100, 1);
        for i in 0..300 {
            ac.update(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        // Biased estimator: -(n - k) / n.
        assert!((ac.last().unwrap() + 0.99).abs() < 1e-9);
    }

    #[test]
    fn autocorrelation_constant_is_zero() {
        let mut ac = new_f64(20, 2);
        for _ in 0..100 {
            ac.update(7.3);
        }
        assert_eq!(ac.last().unwrap(), 0.0);
    }

    #[test]
    #[should_panic(expected = "lag must be smaller than window_len")]
    fn autocorrelation_lag_too_large() {
        let _ = new_f64(5, 5);
    }

    #[test]
    fn autocorrelation_plot() {
        let mut ac = new_f64(32, 1);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for w in TEST_DATA.windows(2) {
            ac.update(w[1] - w[0]);
            if let Some(val) = ac.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/autocorrelation.png").unwrap();
    }
}

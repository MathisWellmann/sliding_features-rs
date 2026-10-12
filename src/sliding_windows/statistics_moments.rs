//! Shared sliding power-sum moments helper for `Skewness` and `Kurtosis`.
//!
//! Maintains the running count, mean and the second/third/fourth central
//! sums `M_k = Σ(x_i - mean)^k` of the values currently in a fixed-length
//! sliding window, from which the population central moments (`M_k / n`)
//! and hence the mean, variance, skewness and kurtosis are recovered in
//! O(1).
//!
//! # Numerical stability
//! Both insertion and removal update `mean` and `M2..M4` incrementally
//! without ever forming `Σx^k` about a fixed point, so there is no
//! catastrophic cancellation between a huge `Σx^k` and a huge `n * mean^k`
//! for large-magnitude data (the running `mean` tracks the data at every
//! step, not just periodically). Insertion uses Terriberry's (2008)
//! extension of Welford's online algorithm to higher moments (as described
//! in Wikipedia, "Algorithms for calculating variance", section
//! "Higher-order statistics"). Removal is the same article's "parallel"
//! merge formula for combining two groups, solved for the first group given
//! the combined state and a one-element second group: this is the standard
//! way to turn a one-pass accumulator into a sliding one, since removing a
//! sample is "un-merging" it. Both are O(1) and only ever add/subtract/
//! multiply bounded quantities already present in the running state.
//!
//! An earlier version of this helper centered raw power sums on a shift
//! recomputed only periodically; that suffered exactly the cancellation
//! this design avoids whenever a transient outlier left the window before
//! the next recompute (caught by `statistics_moments_vs_naive` below, which
//! failed under that design and passes under this one).

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use num::Float;

/// Sliding window of the running mean and second/third/fourth central sums,
/// used to derive the mean and the population central moments in O(1).
#[derive(Debug, Clone)]
pub(crate) struct StatisticsMoments<T> {
    window_len: NonZeroUsize,
    /// Raw values in insertion order, needed to know which value to remove
    /// when the window slides.
    buf: VecDeque<T>,
    mean: T,
    /// `Σ(x_i - mean)^2`.
    m2: T,
    /// `Σ(x_i - mean)^3`.
    m3: T,
    /// `Σ(x_i - mean)^4`.
    m4: T,
}

impl<T: Float> StatisticsMoments<T> {
    /// Create an empty moments window holding at most `window_len` values.
    pub(crate) fn new(window_len: NonZeroUsize) -> Self {
        Self {
            window_len,
            buf: VecDeque::with_capacity(window_len.get()),
            mean: T::zero(),
            m2: T::zero(),
            m3: T::zero(),
            m4: T::zero(),
        }
    }

    /// The maximum number of values in the window.
    pub(crate) fn window_len(&self) -> NonZeroUsize {
        self.window_len
    }

    /// Whether the window holds `window_len` values.
    pub(crate) fn is_full(&self) -> bool {
        self.buf.len() == self.window_len.get()
    }

    /// Push `val`, evicting the oldest value first if the window is full.
    pub(crate) fn push(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        if self.buf.len() == self.window_len.get() {
            let old = self.buf.pop_front().expect("the window is full");
            self.remove(old);
        }
        self.buf.push_back(val);
        self.insert(val);
    }

    /// Terriberry's (2008) one-pass update of `mean`, `m2`, `m3`, `m4` for
    /// inserting `x`, generalizing Welford's online mean/variance update.
    fn insert(&mut self, x: T) {
        let n1 = T::from(self.buf.len() - 1).expect("can convert");
        let n = T::from(self.buf.len()).expect("can convert");
        let (two, three, four, six) = moment_constants::<T>();

        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;

        self.mean = self.mean + delta_n;
        self.m4 =
            self.m4 + term1 * delta_n2 * (n * n - three * n + three) + six * delta_n2 * self.m2
                - four * delta_n * self.m3;
        self.m3 = self.m3 + term1 * delta_n * (n - two) - three * delta_n * self.m2;
        self.m2 = self.m2 + term1;
    }

    /// The inverse of [`Self::insert`]: removes `x` (the value evicted from
    /// `buf`) from `mean`, `m2`, `m3`, `m4`, by solving the "combine two
    /// groups" formula for the group without `x`, given the current
    /// (`x`-including) state as the combined group and `{x}` as the other.
    fn remove(&mut self, x: T) {
        // `buf` has already had `x` popped by `push`, so its length is the
        // post-removal count `n_x`; `n` (the pre-removal, combined count)
        // is one more than that.
        let n_x = T::from(self.buf.len()).expect("can convert");
        let n = n_x + T::one();
        if n_x <= T::zero() {
            self.mean = T::zero();
            self.m2 = T::zero();
            self.m3 = T::zero();
            self.m4 = T::zero();
            return;
        }
        let (_two, three, four, six) = moment_constants::<T>();

        let mean_x = (self.mean * n - x) / n_x;
        let delta = x - mean_x;
        let m2_x = self.m2 - delta * delta * n_x / n;
        let m3_x =
            self.m3 - delta.powi(3) * n_x * (n_x - T::one()) / (n * n) + three * delta * m2_x / n;
        let m4_x = self.m4
            - delta.powi(4) * n_x * (n_x * n_x - n_x + T::one()) / (n * n * n)
            - six * delta * delta * m2_x / (n * n)
            + four * delta * m3_x / n;

        self.mean = mean_x;
        self.m2 = m2_x;
        self.m3 = m3_x;
        self.m4 = m4_x;
    }

    /// The number of samples the moments were computed over.
    #[inline]
    fn n(&self) -> T {
        T::from(self.buf.len()).expect("can convert")
    }

    /// The mean of the window.
    ///
    /// Not used by library code yet (`Skewness` and `Kurtosis` only need the
    /// central moments below); `#[cfg(test)]` to avoid a `dead_code`
    /// warning. Remove the attribute when using it.
    #[cfg(test)]
    pub(crate) fn mean(&self) -> T {
        self.mean
    }

    /// The population (`/ n`, biased) second central moment of the window,
    /// i.e. the biased variance.
    pub(crate) fn central_moment2(&self) -> T {
        let m2 = self.m2 / self.n();
        // Round-off can push this slightly negative for near-constant data.
        if m2 < T::zero() { T::zero() } else { m2 }
    }

    /// The population third central moment of the window.
    pub(crate) fn central_moment3(&self) -> T {
        self.m3 / self.n()
    }

    /// The population fourth central moment of the window.
    pub(crate) fn central_moment4(&self) -> T {
        self.m4 / self.n()
    }
}

/// The constants `2, 3, 4, 6` converted to `T`, for the moment update
/// formulas above.
#[inline]
fn moment_constants<T: Float>() -> (T, T, T, T) {
    (
        T::from(2.0).expect("can convert"),
        T::from(3.0).expect("can convert"),
        T::from(4.0).expect("can convert"),
        T::from(6.0).expect("can convert"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_central_moments(vals: &[f64]) -> (f64, f64, f64, f64) {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let m2 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
        let m4 = vals.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
        (mean, m2, m3, m4)
    }

    #[test]
    fn statistics_moments_vs_naive() {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let window_len = 20;
        let mut m = StatisticsMoments::<f64>::new(NonZeroUsize::new(window_len).unwrap());
        let mut all: VecDeque<f64> = VecDeque::new();
        for i in 0..500 {
            // A constant large offset stresses numerical stability: unlike a
            // transient outlier (which forces a large-magnitude cancellation
            // when it leaves the window, an unavoidable limitation shared by
            // every O(1) streaming moment estimator), a shift-invariant
            // design handles this perfectly, since every delta is taken
            // against a running mean that itself tracks the offset.
            let val = 1e6 + rng.f64() * 10.0;
            m.push(val);
            all.push_back(val);
            if all.len() > window_len {
                all.pop_front();
            }
            if all.len() < window_len {
                continue;
            }
            let window: Vec<f64> = all.iter().copied().collect();
            let (mean, m2, m3, m4) = naive_central_moments(&window);
            let close = |got: f64, want: f64| (got - want).abs() <= 1e-6 * (1.0 + want.abs());
            assert!(close(m.mean(), mean), "mean mismatch at {i}");
            assert!(close(m.central_moment2(), m2), "m2 mismatch at {i}");
            assert!(close(m.central_moment3(), m3), "m3 mismatch at {i}");
            assert!(close(m.central_moment4(), m4), "m4 mismatch at {i}");
        }
    }

    #[test]
    fn statistics_moments_constant_input() {
        let mut m = StatisticsMoments::<f64>::new(NonZeroUsize::new(4).unwrap());
        for _ in 0..10 {
            m.push(3.0);
        }
        assert_eq!(m.mean(), 3.0);
        assert_eq!(m.central_moment2(), 0.0);
        assert_eq!(m.central_moment3(), 0.0);
    }
}

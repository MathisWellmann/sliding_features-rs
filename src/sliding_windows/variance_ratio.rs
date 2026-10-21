// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Lo-MacKinlay variance ratio over a sliding window of returns.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Variance ratio `VR(q)` after A. W. Lo and A. C. MacKinlay, "Stock Market Prices Do Not Follow
/// Random Walks: Evidence from a Simple Specification Test", Rev. Financ. Stud. 1 (1988).
///
/// Input convention: the chained input are 1-period returns `r_t` (e.g. log returns or price
/// differences), NOT price levels. The window holds the last `n = window_len` returns.
///
/// With `mu` the window mean and the overlapping q-period sums `Q_t = r_t + ... + r_{t-q+1}`
/// that lie fully inside the window (`n - q + 1` of them), using the unbiased estimators of the
/// paper (their eq. for `sigma_a^2` and the overlapping `sigma_c^2` with `nq = n`):
/// - `sigma_a^2 = sum (r_t - mu)^2 / (n - 1)`
/// - `sigma_c^2 = sum (Q_t - q mu)^2 / m`, with `m = q (n - q + 1) (1 - q / n)`
/// - `VR(q) = sigma_c^2 / sigma_a^2`
///
/// Output range `[0, inf)`: about 1 for a random walk, above 1 for positively autocorrelated
/// (trending) returns, below 1 for negatively autocorrelated (mean-reverting) returns.
/// `None` after warm-up when the window's returns have numerically zero variance.
///
/// Complexity per update: O(1) amortized via running sums over values shifted by a reference
/// value (against cancellation); the sums are rebuilt exactly every `window_len` updates,
/// which bounds floating point drift.
#[derive(Debug, Clone, CopyGetters)]
pub struct VarianceRatio<T: Float, V> {
    view: V,
    /// The number of returns in the sliding window.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The aggregation period q of the long-horizon returns.
    #[getset(get_copy = "pub")]
    q: usize,
    rets: VecDeque<T>,
    // Overlapping q-period sums that lie fully inside the window, unshifted.
    qsums: VecDeque<T>,
    // Sum of the last q returns.
    qsum: T,
    sums: Sums<T>,
    updates_since_rebuild: usize,
}

/// Running sums of the shifted returns `y = r - shift` and q-sums `Q - q * shift`.
#[derive(Debug, Clone, Copy)]
struct Sums<T> {
    shift: T,
    s1: T,
    s2: T,
    sq1: T,
    sq2: T,
}

impl<T: Float> Sums<T> {
    fn new(shift: T) -> Self {
        Self {
            shift,
            s1: T::zero(),
            s2: T::zero(),
            sq1: T::zero(),
            sq2: T::zero(),
        }
    }

    /// Sums of `rets` and `qsums` with a fresh shift, to discard accumulated rounding error.
    fn rebuild(shift: T, q: T, rets: &VecDeque<T>, qsums: &VecDeque<T>) -> Self {
        let mut sums = Self::new(shift);
        rets.iter().for_each(|&r| sums.add_ret(r, T::one()));
        qsums.iter().for_each(|&s| sums.add_qsum(s, q, T::one()));
        sums
    }

    /// Adds (`sign = 1`) or removes (`sign = -1`) a return.
    #[inline]
    fn add_ret(&mut self, r: T, sign: T) {
        let y = r - self.shift;
        self.s1 = self.s1 + sign * y;
        self.s2 = self.s2 + sign * y * y;
    }

    /// Adds (`sign = 1`) or removes (`sign = -1`) a q-period sum.
    #[inline]
    fn add_qsum(&mut self, s: T, q: T, sign: T) {
        let y = s - q * self.shift;
        self.sq1 = self.sq1 + sign * y;
        self.sq2 = self.sq2 + sign * y * y;
    }
}

impl<T, V> VarianceRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `VarianceRatio` with a chained `View` that yields 1-period returns.
    ///
    /// # Panics
    /// If `q < 2` or `window_len <= q`.
    pub fn new(view: V, window_len: NonZeroUsize, q: usize) -> Self {
        assert!(q >= 2, "q must be at least 2");
        assert!(window_len.get() > q, "window_len must be larger than q");
        Self {
            view,
            window_len,
            q,
            rets: VecDeque::with_capacity(window_len.get() + 1),
            qsums: VecDeque::with_capacity(window_len.get() - q + 2),
            qsum: T::zero(),
            sums: Sums::new(T::zero()),
            updates_since_rebuild: 0,
        }
    }

    fn rebuild(&mut self) {
        let shift = *self.rets.back().expect("rebuild only after an update");
        let q = T::from(self.q).expect("can convert");
        self.sums = Sums::rebuild(shift, q, &self.rets, &self.qsums);
        let start = self.rets.len().saturating_sub(self.q);
        self.qsum = self.rets.range(start..).fold(T::zero(), |acc, &r| acc + r);
        self.updates_since_rebuild = 0;
    }
}

/// `VR(q)` from the running sums, `None` for numerically zero variance.
fn variance_ratio<T: Float>(sums: &Sums<T>, n: usize, q: usize) -> Option<T> {
    let (nt, qt) = (
        T::from(n).expect("can convert"),
        T::from(q).expect("can convert"),
    );
    let count_q = T::from(n - q + 1).expect("can convert");
    let mu = sums.s1 / nt;
    let ss_a = sums.s2 - sums.s1 * mu;
    // Cancellation leaves an error of order n * eps * s2, anything below is zero variance.
    if ss_a <= nt * T::epsilon() * sums.s2 {
        return None;
    }
    let ss_c = (sums.sq2 - (qt + qt) * mu * sums.sq1 + count_q * qt * qt * mu * mu).max(T::zero());
    let m = qt * count_q * (T::one() - qt / nt);
    debug_assert!(m > T::zero(), "n > q gives a positive normalization");
    let out = (ss_c / m) / (ss_a / (nt - T::one()));
    debug_assert!(out >= T::zero(), "variance ratio is non-negative");
    Some(out)
}

impl<T, V> View<T> for VarianceRatio<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let q = T::from(self.q).expect("can convert");
        self.rets.push_back(val);
        self.sums.add_ret(val, T::one());
        self.qsum = self.qsum + val;
        if self.rets.len() > self.q {
            self.qsum = self.qsum - self.rets[self.rets.len() - 1 - self.q];
        }
        if self.rets.len() > self.window_len.get() {
            let old = self.rets.pop_front().expect("not empty");
            self.sums.add_ret(old, -T::one());
        }
        if self.rets.len() >= self.q {
            self.qsums.push_back(self.qsum);
            self.sums.add_qsum(self.qsum, q, T::one());
        }
        if self.qsums.len() > self.window_len.get() - self.q + 1 {
            let old = self.qsums.pop_front().expect("not empty");
            self.sums.add_qsum(old, q, -T::one());
        }
        self.updates_since_rebuild += 1;
        if self.updates_since_rebuild >= self.window_len.get() {
            self.rebuild();
        }
    }

    fn last(&self) -> Option<T> {
        if self.rets.len() < self.window_len.get() {
            return None;
        }
        debug_assert_eq!(self.qsums.len(), self.window_len.get() - self.q + 1);
        let out = variance_ratio(&self.sums, self.window_len.get(), self.q)?;
        debug_assert!(out.is_finite(), "value must be finite");
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

    fn new_f64(window_len: usize, q: usize) -> VarianceRatio<f64, Echo<f64>> {
        VarianceRatio::new(Echo::new(), NonZeroUsize::new(window_len).unwrap(), q)
    }

    /// Direct two-pass computation of the Lo-MacKinlay estimator.
    fn naive(r: &[f64], q: usize) -> f64 {
        let n = r.len() as f64;
        let mu = r.iter().sum::<f64>() / n;
        let var_a = r.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
        let qf = q as f64;
        let m = qf * (n - qf + 1.0) * (1.0 - qf / n);
        let var_c = r
            .windows(q)
            .map(|w| (w.iter().sum::<f64>() - qf * mu).powi(2))
            .sum::<f64>()
            / m;
        var_c / var_a
    }

    fn mean_vr(window_len: usize, q: usize, rets: &[f64]) -> f64 {
        let mut vr = new_f64(window_len, q);
        let mut outs = Vec::new();
        for r in rets {
            vr.update(*r);
            if let Some(o) = vr.last() {
                outs.push(o);
            }
        }
        outs.iter().sum::<f64>() / outs.len() as f64
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
    fn variance_ratio_warmup() {
        let mut vr = new_f64(5, 2);
        for i in 0..4 {
            vr.update(i as f64 * 0.1 + (i % 2) as f64);
            assert_eq!(vr.last(), None);
        }
        vr.update(0.3);
        assert!(vr.last().is_some());
    }

    #[test]
    fn variance_ratio_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(31);
        for (window_len, q) in [(10, 2), (50, 5), (101, 7)] {
            let mut vr = new_f64(window_len, q);
            let mut vals = Vec::new();
            for _ in 0..1_000 {
                // A large offset stresses the cancellation handling.
                let v = 100.0 + rng.f64();
                vals.push(v);
                vr.update(v);
                if vals.len() < window_len {
                    continue;
                }
                let expected = naive(&vals[vals.len() - window_len..], q);
                let got = vr.last().unwrap();
                assert!(
                    (got - expected).abs() < 1e-6 * expected,
                    "{got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn variance_ratio_random_walk_near_one() {
        let vr = mean_vr(2_000, 4, &ar1(37, 0.0, 10_000));
        assert!((vr - 1.0).abs() < 0.1, "{vr}");
    }

    #[test]
    fn variance_ratio_detects_autocorrelation() {
        let trending = mean_vr(1_000, 4, &ar1(41, 0.5, 5_000));
        let reverting = mean_vr(1_000, 4, &ar1(43, -0.5, 5_000));
        assert!(trending > 1.5, "{trending}");
        assert!(reverting < 0.6, "{reverting}");
    }

    #[test]
    fn variance_ratio_constant_is_none() {
        let mut vr = new_f64(20, 3);
        for _ in 0..100 {
            vr.update(0.25);
            assert_eq!(vr.last(), None);
        }
    }

    #[test]
    #[should_panic(expected = "q must be at least 2")]
    fn variance_ratio_q_too_small() {
        let _ = new_f64(10, 1);
    }

    #[test]
    #[should_panic(expected = "window_len must be larger than q")]
    fn variance_ratio_window_too_small() {
        let _ = new_f64(4, 4);
    }

    #[test]
    fn variance_ratio_plot() {
        let mut vr = new_f64(64, 4);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for w in TEST_DATA.windows(2) {
            vr.update((w[1] / w[0]).ln());
            if let Some(val) = vr.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/variance_ratio.png").unwrap();
    }
}

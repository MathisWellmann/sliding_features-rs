// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Ornstein-Uhlenbeck mean-reversion half-life over a sliding window.

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Mean-reversion half-life (in updates) of an Ornstein-Uhlenbeck process fitted to the window,
/// the standard OLS approach for discretely sampled OU processes (e.g. E. Chan,
/// "Algorithmic Trading: Winning Strategies and Their Rationale", 2013, ch. 2).
///
/// Over the last `n = window_len` pairs `(x_{t-1}, dx_t = x_t - x_{t-1})` the OLS fit
/// `dx_t = a + b x_{t-1}` gives the AR(1) coefficient `phi = 1 + b` of `x_t = a + phi x_{t-1}`.
/// The output is the exact discrete-time half-life `-ln(2) / ln(|phi|)`, i.e. the number of
/// updates after which an expected deviation from the mean has halved.
/// The continuous-time approximation `-ln(2) / b` used by some authors overstates the half-life
/// for strong reversion (`b = -0.5`: 1.39 vs. 1.0) and is not used.
///
/// Input convention: the chained input are levels (prices, spreads), not returns.
///
/// Output range `[0, inf)`. `None` after warm-up when:
/// - `b >= 0` (`phi >= 1`): no mean reversion (random walk or trending), half-life infinite,
/// - `b <= -2` (`phi <= -1`): explosive oscillation,
/// - the window's `x_{t-1}` have numerically zero variance (slope undefined).
///
/// For `-2 < b < -1` (`-1 < phi < 0`) the process overshoots the mean; the reported half-life is
/// that of the deviation's magnitude, `|phi|^h = 1/2`. `phi = 0` gives 0.
///
/// Complexity per update: O(1) amortized via running sums over levels shifted by a reference
/// value (against cancellation), rebuilt exactly every `window_len` updates to bound drift.
#[derive(Debug, Clone, CopyGetters)]
pub struct OuHalfLife<T: Float, V> {
    view: V,
    /// The number of `(x_{t-1}, dx_t)` pairs in the sliding window.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    // The last `window_len + 1` levels.
    values: VecDeque<T>,
    sums: Sums<T>,
    updates_since_rebuild: usize,
}

/// Running regression sums over `u = x_{t-1} - shift` and `dx`.
#[derive(Debug, Clone, Copy)]
struct Sums<T> {
    shift: T,
    su: T,
    suu: T,
    sd: T,
    sud: T,
}

impl<T: Float> Sums<T> {
    fn new(shift: T) -> Self {
        Self {
            shift,
            su: T::zero(),
            suu: T::zero(),
            sd: T::zero(),
            sud: T::zero(),
        }
    }

    /// Exact sums of `values` with a fresh shift, to discard accumulated rounding error.
    fn rebuild(shift: T, values: &VecDeque<T>) -> Self {
        let mut sums = Self::new(shift);
        for i in 1..values.len() {
            sums.add_pair(values[i - 1], values[i], T::one());
        }
        sums
    }

    /// Adds (`sign = 1`) or removes (`sign = -1`) the pair of consecutive levels.
    #[inline]
    fn add_pair(&mut self, prev: T, cur: T, sign: T) {
        let u = prev - self.shift;
        let d = cur - prev;
        self.su = self.su + sign * u;
        self.suu = self.suu + sign * u * u;
        self.sd = self.sd + sign * d;
        self.sud = self.sud + sign * u * d;
    }

    /// OLS slope `b` of `dx` on `x_{t-1}`, `None` for numerically zero variance of `x_{t-1}`.
    fn slope(&self, n: usize) -> Option<T> {
        let nt = T::from(n).expect("can convert");
        let den = nt * self.suu - self.su * self.su;
        // Cancellation leaves an error of order n^2 * eps * suu, anything below is zero variance.
        if den <= nt * nt * T::epsilon() * self.suu {
            return None;
        }
        Some((nt * self.sud - self.su * self.sd) / den)
    }
}

impl<T, V> OuHalfLife<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `OuHalfLife` with a chained `View` that yields levels.
    ///
    /// # Panics
    /// If `window_len < 3`, as two pairs always fit a line exactly.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 3, "window_len must be at least 3");
        Self {
            view,
            window_len,
            values: VecDeque::with_capacity(window_len.get() + 2),
            sums: Sums::new(T::zero()),
            updates_since_rebuild: 0,
        }
    }

    /// The fitted AR(1) coefficient `phi = 1 + b` of `x_t = a + phi x_{t-1}`.
    /// `None` during warm-up or when the window's levels have numerically zero variance.
    pub fn ar_coefficient(&self) -> Option<T> {
        if self.values.len() <= self.window_len.get() {
            return None;
        }
        let phi = T::one() + self.sums.slope(self.window_len.get())?;
        debug_assert!(phi.is_finite(), "value must be finite");
        Some(phi)
    }
}

/// Half-life `-ln(2) / ln(|phi|)` for `|phi| < 1`, `None` otherwise.
fn half_life<T: Float>(phi: T) -> Option<T> {
    if phi.abs() >= T::one() {
        return None;
    }
    if phi == T::zero() {
        return Some(T::zero());
    }
    let out = -T::from(std::f64::consts::LN_2).expect("can convert") / phi.abs().ln();
    debug_assert!(out > T::zero(), "half-life of a contraction is positive");
    Some(out)
}

impl<T, V> View<T> for OuHalfLife<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if let Some(&prev) = self.values.back() {
            self.sums.add_pair(prev, val, T::one());
        }
        self.values.push_back(val);
        if self.values.len() > self.window_len.get() + 1 {
            let (old, next) = (self.values[0], self.values[1]);
            self.sums.add_pair(old, next, -T::one());
            self.values.pop_front();
        }
        self.updates_since_rebuild += 1;
        if self.updates_since_rebuild >= self.window_len.get() {
            self.sums = Sums::rebuild(val, &self.values);
            self.updates_since_rebuild = 0;
        }
    }

    fn last(&self) -> Option<T> {
        let out = half_life(self.ar_coefficient()?)?;
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

    fn new_f64(window_len: usize) -> OuHalfLife<f64, Echo<f64>> {
        OuHalfLife::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Two-pass OLS slope of dx on x_{t-1}.
    fn naive_slope(x: &[f64]) -> f64 {
        let prev = &x[..x.len() - 1];
        let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let n = prev.len() as f64;
        let mp = prev.iter().sum::<f64>() / n;
        let md = dx.iter().sum::<f64>() / n;
        let cov: f64 = prev.iter().zip(&dx).map(|(p, d)| (p - mp) * (d - md)).sum();
        let var: f64 = prev.iter().map(|p| (p - mp).powi(2)).sum();
        cov / var
    }

    /// Discretized OU: x += theta (mu - x) + noise.
    fn ou(seed: u64, theta: f64, mu: f64, n: usize) -> Vec<f64> {
        let rng = romu::Rng::from_seed_with_64bit(seed);
        let mut x = mu;
        (0..n)
            .map(|_| {
                x += theta * (mu - x) + rng.f64() - 0.5;
                x
            })
            .collect()
    }

    #[test]
    fn ou_half_life_warmup() {
        let x = ou(1, 0.5, 0.0, 5);
        let mut hl = new_f64(4);
        for v in &x[..4] {
            hl.update(*v);
            assert_eq!(hl.ar_coefficient(), None);
        }
        hl.update(x[4]);
        assert!(hl.ar_coefficient().is_some());
    }

    #[test]
    fn ou_half_life_matches_naive() {
        for window_len in [3, 20, 101] {
            let x = ou(window_len as u64, 0.2, 100.0, 1_000);
            let mut hl = new_f64(window_len);
            for (i, v) in x.iter().enumerate() {
                hl.update(*v);
                if i < window_len {
                    continue;
                }
                let b = naive_slope(&x[i - window_len..=i]);
                let phi = hl.ar_coefficient().unwrap();
                assert!((phi - (1.0 + b)).abs() < 1e-7, "{phi} vs {}", 1.0 + b);
                let expected =
                    (b < 0.0 && b > -2.0).then(|| -(2.0_f64.ln()) / (1.0 + b).abs().ln());
                match (hl.last(), expected) {
                    (Some(got), Some(exp)) => assert!((got - exp).abs() < 1e-6 * exp.max(1.0)),
                    (got, exp) => assert_eq!(got, exp),
                }
            }
        }
    }

    #[test]
    fn ou_half_life_recovers_theta() {
        let theta = 0.05;
        let mut hl = new_f64(10_000);
        for v in ou(61, theta, 50.0, 20_000) {
            hl.update(v);
        }
        let got = hl.last().unwrap();
        let exact = -(2.0_f64.ln()) / (1.0 - theta).ln();
        let continuous = 2.0_f64.ln() / theta;
        assert!((got - exact).abs() < 0.15 * exact, "{got} vs {exact}");
        assert!(
            (got - continuous).abs() < 0.2 * continuous,
            "{got} vs {continuous}"
        );
    }

    #[test]
    fn ou_half_life_no_mean_reversion_is_none() {
        let mut trend = new_f64(50);
        let mut explosive = new_f64(50);
        let mut x = 1.0;
        for i in 0..200 {
            trend.update(i as f64);
            x *= 1.01;
            explosive.update(x);
        }
        assert_eq!(trend.last(), None);
        assert_eq!(explosive.last(), None);
        assert!(explosive.ar_coefficient().unwrap() > 1.0);
    }

    #[test]
    fn ou_half_life_overshooting() {
        // x_t = -0.5 x_{t-1} + noise: |phi| = 0.5 halves the deviation every step.
        let rng = romu::Rng::from_seed_with_64bit(67);
        let mut hl = new_f64(5_000);
        let mut x = 0.0;
        for _ in 0..10_000 {
            x = -0.5 * x + rng.f64() - 0.5;
            hl.update(x);
        }
        let got = hl.last().unwrap();
        assert!((got - 1.0).abs() < 0.1, "{got}");
    }

    #[test]
    fn ou_half_life_constant_is_none() {
        let mut hl = new_f64(20);
        for _ in 0..100 {
            hl.update(3.7);
        }
        assert_eq!(hl.ar_coefficient(), None);
        assert_eq!(hl.last(), None);
    }

    #[test]
    fn ou_half_life_half_life_edges() {
        assert_eq!(half_life(1.0), None);
        assert_eq!(half_life(-1.0), None);
        assert_eq!(half_life(1.5), None);
        assert_eq!(half_life(0.0), Some(0.0));
        assert!((half_life(0.5).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "window_len must be at least 3")]
    fn ou_half_life_window_too_small() {
        let _ = new_f64(2);
    }

    #[test]
    fn ou_half_life_plot() {
        let mut hl = new_f64(64);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            hl.update(*v);
            // Gaps (no mean reversion) are plotted as 0.
            if hl.ar_coefficient().is_some() {
                out.push(hl.last().unwrap_or(0.0));
            }
        }
        plot_values(out, "img/ou_half_life.png").unwrap();
    }
}

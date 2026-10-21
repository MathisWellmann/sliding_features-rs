// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Recursive least squares (RLS) adaptive linear predictor.

use std::num::NonZeroUsize;

use num::Float;

use crate::View;

/// A recursive least squares (RLS) adaptive linear predictor (Plackett,
/// 1950; popularized for adaptive filtering by Haykin, "Adaptive Filter
/// Theory") with `order` taps, a forgetting factor `lambda` and a ridge-like
/// initialization `delta` for the inverse correlation matrix `P = I / delta`.
///
/// Predicts the current value from the `order` values observed before it,
/// `y_hat = w . h`, exactly as [`LmsFilter`](crate::sliding_windows::LmsFilter)
/// and [`NlmsFilter`](crate::sliding_windows::NlmsFilter) do, but instead of
/// a stochastic gradient step it solves the exponentially-weighted least
/// squares problem recursively: `pi = P h`, `k = pi / (lambda + h^T pi)`
/// (the gain), `w += k e`, `P = (P - k pi^T) / lambda`. `lambda` in `(0, 1]`
/// controls how fast past data is forgotten (`1` never forgets); smaller
/// values track non-stationary coefficients faster but are noisier.
///
/// RLS converges in far fewer samples than LMS/NLMS, at the cost of
/// `O(order^2)` time and memory per update for the `P` matrix (stored as a
/// flat, row-major `Vec` of length `order^2`, preallocated once).
///
/// As with `LmsFilter`/`NlmsFilter`, `last()` -- read after `update(x_t)` --
/// is the one-step-ahead prediction of `x_{t+1}`. It is `None` until
/// `order + 1` values were observed.
///
/// Input that does not excite every direction of the tap space (zeros, a
/// constant) makes `P` grow by `1 / lambda` per update along the others
/// ("covariance wind-up"), until rounding makes it indefinite, gains turn
/// to garbage and `P` overflows. So once an entry of `P` exceeds a million
/// times its initial `1 / delta`, or `h^T P h` turns negative, `P` is reset
/// to `I / delta` (Goodwin and Sin's covariance resetting), keeping the
/// weights; should the weights still overflow, they restart from zero.
/// Inputs that excite every direction with a power well above
/// `delta (1 - lambda) / 1e6` never trigger it; with `lambda = 1`, `P`
/// only shrinks.
#[derive(Debug, Clone)]
pub struct RlsFilter<T, V> {
    view: V,
    order: NonZeroUsize,
    lambda: T,
    weights: Vec<T>,
    history: Vec<T>,
    /// The inverse correlation matrix, flattened row-major, `order x order`.
    p: Vec<T>,
    /// Preallocated scratch buffer for `P h`, reused every update.
    pi: Vec<T>,
    /// Preallocated scratch buffer for the gain vector, reused every update.
    gain: Vec<T>,
    /// The initial diagonal of `P`, `1 / delta`.
    p_diag: T,
    prediction: T,
    error: T,
    n_observed: usize,
}

impl<T, V> RlsFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new predictor with a chained `View`, tap count `order`,
    /// forgetting factor `lambda` and initialization `delta` for
    /// `P = I / delta`.
    ///
    /// # Panics
    /// If `lambda` is not in `(0, 1]` or `delta <= 0`.
    pub fn new(view: V, order: NonZeroUsize, lambda: T, delta: T) -> Self {
        assert!(lambda > T::zero(), "lambda must be > 0");
        assert!(lambda <= T::one(), "lambda must be <= 1");
        assert!(delta > T::zero(), "delta must be > 0");
        let n = order.get();
        let p_diag = T::one() / delta;
        let mut p = vec![T::zero(); n * n];
        reset_to_scaled_identity(&mut p, n, p_diag);
        Self {
            view,
            order,
            lambda,
            weights: vec![T::zero(); n],
            history: Vec::with_capacity(n),
            p,
            pi: vec![T::zero(); n],
            gain: vec![T::zero(); n],
            p_diag,
            prediction: T::zero(),
            error: T::zero(),
            n_observed: 0,
        }
    }

    /// The number of taps.
    #[inline]
    pub fn order(&self) -> NonZeroUsize {
        self.order
    }

    /// The forgetting factor in `(0, 1]`.
    #[inline]
    pub fn lambda(&self) -> T {
        self.lambda
    }

    /// The last a-priori prediction error (actual minus predicted).
    #[inline]
    pub fn error(&self) -> T {
        self.error
    }

    /// The current tap weights, oldest-history-first.
    #[inline]
    pub fn weights(&self) -> &[T] {
        &self.weights
    }

    /// The dot product of the weights with the current tap history.
    fn predict(&self) -> T {
        self.weights
            .iter()
            .zip(self.history.iter())
            .fold(T::zero(), |acc, (w, h)| acc + *w * *h)
    }

    /// Push `val` into the fixed-size tap history, evicting the oldest
    /// value once full. Allocation-free: `history` never grows past `order`.
    fn push_history(&mut self, val: T) {
        let n = self.order.get();
        if self.history.len() == n {
            self.history.copy_within(1.., 0);
            *self.history.last_mut().expect("order > 0") = val;
        } else {
            self.history.push(val);
        }
    }

    /// Set `pi = P h` and return the gain denominator `lambda + h^T P h`.
    fn project_history(&mut self) -> T {
        let n = self.order.get();
        for i in 0..n {
            let row = &self.p[i * n..i * n + n];
            self.pi[i] = row
                .iter()
                .zip(self.history.iter())
                .fold(T::zero(), |acc, (p_ij, h_j)| acc + *p_ij * *h_j);
        }
        self.lambda
            + self
                .history
                .iter()
                .zip(self.pi.iter())
                .fold(T::zero(), |acc, (h, pi)| acc + *h * *pi)
    }

    /// Start over from zero weights and `P = I / delta`, keeping the history.
    fn restart(&mut self) {
        self.weights.fill(T::zero());
        reset_to_scaled_identity(&mut self.p, self.order.get(), self.p_diag);
    }

    /// One RLS correction step: updates `pi`, `gain`, `weights`, `p` and
    /// `error` in place from the current `history` and the observed `val`.
    fn correct(&mut self, val: T) {
        let n = self.order.get();
        let mut denom = self.project_history();
        // `h^T P h >= 0` for the positive definite `P` of exact arithmetic;
        // below, rounding broke `P` (after long wind-ups with small lambda).
        let half = T::from(0.5).expect("can convert");
        if !denom.is_finite() || denom < half * self.lambda {
            reset_to_scaled_identity(&mut self.p, n, self.p_diag);
            denom = self.project_history();
        }
        debug_assert!(denom.is_finite(), "value must be finite");
        for i in 0..n {
            self.gain[i] = self.pi[i] / denom;
        }

        self.error = val - self.predict();
        for (w, g) in self.weights.iter_mut().zip(self.gain.iter()) {
            *w = *w + *g * self.error;
        }
        if !self.error.is_finite() || self.weights.iter().any(|w| !w.is_finite()) {
            self.restart();
            self.error = val;
            return;
        }
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                self.p[idx] = (self.p[idx] - self.gain[i] * self.pi[j]) / self.lambda;
            }
        }
        reset_on_wind_up(&mut self.p, n, self.p_diag);
        debug_assert!(self.error.is_finite(), "value must be finite");
    }
}

/// Set the `n x n` row-major matrix `p` to `diag * I`.
fn reset_to_scaled_identity<T: Float>(p: &mut [T], n: usize, diag: T) {
    debug_assert_eq!(p.len(), n * n, "p is n x n");
    p.fill(T::zero());
    for i in 0..n {
        p[i * n + i] = diag;
    }
}

/// How far the entries of `P` may grow beyond its initial diagonal before
/// [`reset_on_wind_up`] resets it: far enough that excited inputs never
/// reach it, close enough that the gains along unexcited directions stay
/// small and that `P` stays far from overflowing.
const MAX_P_GROWTH: f64 = 1e6;

/// Reset `p` to its initial `diag * I` once an entry exceeds
/// [`MAX_P_GROWTH`] times `diag` in magnitude or is not finite. All
/// entries, not only the diagonal: rounding can make a wound-up `p`
/// indefinite, with huge entries of both signs.
fn reset_on_wind_up<T: Float>(p: &mut [T], n: usize, diag: T) {
    let max_entry = T::from(MAX_P_GROWTH).expect("can convert") * diag;
    if p.iter().any(|v| v.is_nan() || v.abs() > max_entry) {
        reset_to_scaled_identity(p, n, diag);
    }
    debug_assert!(p.iter().all(|v| v.is_finite()), "p must stay finite");
}

impl<T, V> View<T> for RlsFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.n_observed += 1;
        if self.history.len() == self.order.get() {
            self.correct(val);
        }

        self.push_history(val);
        if self.history.len() == self.order.get() {
            self.prediction = self.predict();
            debug_assert!(self.prediction.is_finite(), "value must be finite");
        }
    }

    fn last(&self) -> Option<T> {
        if self.n_observed <= self.order.get() {
            return None;
        }
        Some(self.prediction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::LmsFilter,
        test_data::TEST_DATA,
    };

    fn rls(order: usize, lambda: f64, delta: f64) -> RlsFilter<f64, Echo<f64>> {
        RlsFilter::new(
            Echo::new(),
            NonZeroUsize::new(order).unwrap(),
            lambda,
            delta,
        )
    }

    /// Simulate an AR(2) process `x_t = a1 x_{t-1} + a2 x_{t-2} + noise`.
    fn ar2(rng: &romu::Rng, n: usize, a1: f64, a2: f64, noise_sd: f64) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 2..n {
            let u1 = (1.0 - rng.f64()).max(f64::MIN_POSITIVE);
            let u2 = rng.f64();
            let noise =
                noise_sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + noise;
        }
        x
    }

    #[test]
    fn rls_plot() {
        let mut f = rls(8, 0.995, 1.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/rls_filter.png").unwrap();
    }

    #[test]
    fn rls_warmup() {
        let mut f = rls(3, 1.0, 1.0);
        for x in [1.0, 2.0, 3.0] {
            f.update(x);
            assert_eq!(f.last(), None);
        }
        f.update(4.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn rls_learns_ar2_coefficients_and_predicts_near_noise_floor() {
        let rng = romu::Rng::from_seed_with_64bit(12);
        let (a1, a2, noise_sd) = (0.6, -0.3, 0.1);
        let series = ar2(&rng, 20_000, a1, a2, noise_sd);

        let mut f = rls(2, 1.0, 1.0);
        let mut sq_err = Vec::new();
        for x in &series {
            f.update(*x);
            if f.last().is_some() {
                sq_err.push(f.error() * f.error());
            }
        }
        let tail = &sq_err[sq_err.len() / 2..];
        let mse = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!(
            mse < 2.0 * noise_sd * noise_sd,
            "steady-state MSE {mse} should be close to the noise variance {}",
            noise_sd * noise_sd
        );
        assert!(
            (f.weights()[0] - a2).abs() < 0.05,
            "weight[0] {} should be close to a2 {a2}",
            f.weights()[0]
        );
        assert!(
            (f.weights()[1] - a1).abs() < 0.05,
            "weight[1] {} should be close to a1 {a1}",
            f.weights()[1]
        );
    }

    #[test]
    fn rls_converges_faster_than_lms() {
        let rng = romu::Rng::from_seed_with_64bit(13);
        let (a1, a2, noise_sd) = (0.6, -0.3, 0.1);
        let series = ar2(&rng, 300, a1, a2, noise_sd);

        let mut rls_f = rls(2, 0.99, 1.0);
        let mut lms_f = LmsFilter::new(Echo::new(), NonZeroUsize::new(2).unwrap(), 0.05);
        let mut rls_err = Vec::new();
        let mut lms_err = Vec::new();
        for x in &series {
            rls_f.update(*x);
            lms_f.update(*x);
            if rls_f.last().is_some() {
                rls_err.push(rls_f.error() * rls_f.error());
                lms_err.push(lms_f.error() * lms_f.error());
            }
        }
        // Compare the early transient (first 20 corrections): RLS should
        // already be much closer to the noise floor than LMS, which is
        // still adapting its weights.
        let early = 20;
        let rls_early_mse = rls_err[..early].iter().sum::<f64>() / early as f64;
        let lms_early_mse = lms_err[..early].iter().sum::<f64>() / early as f64;
        assert!(
            rls_early_mse < lms_early_mse,
            "RLS early MSE {rls_early_mse} should be lower than LMS early MSE {lms_early_mse}"
        );
    }

    #[test]
    fn rls_constant_input_converges_to_zero_error() {
        // With a constant input every tap equals every other tap, so the
        // two-weight system is underdetermined (any w0 + w1 = 1 fits
        // exactly); a forgetting factor below 1 is needed for the gain
        // along that degenerate direction to keep shrinking the error
        // geometrically rather than only as `O(1/n)`.
        let mut f = rls(2, 0.9, 1.0);
        for _ in 0..150 {
            f.update(5.0);
        }
        assert!(
            f.error().abs() < 1e-5,
            "error {} should vanish on constant input",
            f.error()
        );
    }

    /// Input that does not excite every direction (zeros, a constant)
    /// makes `P` grow by `1 / lambda` per update along the unexcited ones
    /// until it overflows.
    #[test]
    fn rls_survives_covariance_wind_up() {
        for (input, lambda) in [(0.0, 0.5), (5.0, 0.5), (0.0, 1e-9), (1e12, 0.01)] {
            let mut f = rls(2, lambda, 0.01);
            for _ in 0..3000 {
                f.update(input);
                assert!(f.last().is_none_or(f64::is_finite));
            }
            for x in [1.0, -2.0, 3.0, 1e12, 0.0, 4.0] {
                f.update(x);
                assert!(f.last().is_some_and(f64::is_finite), "{input} {lambda}");
            }
            assert!(f.weights().iter().all(|w| w.is_finite()));
        }
        // Too short a memory for 4 taps: rounding makes the wound-up `P`
        // indefinite, with entries near 1e307 of both signs.
        let rng = romu::Rng::from_seed_with_64bit(9);
        let mut price = 100.0;
        let mut f = rls(4, 0.5, 0.01);
        for _ in 0..20_000 {
            price += rng.f64() - 0.5;
            f.update(price);
            assert!(f.last().is_none_or(f64::is_finite));
        }
        assert!(f.weights().iter().all(|w| w.is_finite()));
    }

    #[test]
    #[should_panic(expected = "lambda must be > 0")]
    fn rls_rejects_non_positive_lambda() {
        let _ = rls(4, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "lambda must be <= 1")]
    fn rls_rejects_lambda_above_one() {
        let _ = rls(4, 1.1, 1.0);
    }

    #[test]
    #[should_panic(expected = "delta must be > 0")]
    fn rls_rejects_non_positive_delta() {
        let _ = rls(4, 1.0, 0.0);
    }
}

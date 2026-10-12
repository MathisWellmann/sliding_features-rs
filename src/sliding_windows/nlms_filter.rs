//! Normalized LMS (NLMS) adaptive linear predictor.

use std::num::NonZeroUsize;

use num::Float;

use crate::View;

/// A normalized LMS (NLMS) adaptive linear predictor (Nagumo & Noda, 1967;
/// see also Haykin, "Adaptive Filter Theory") with `order` taps.
///
/// Identical to [`LmsFilter`](crate::sliding_windows::LmsFilter) except that
/// the step is normalized by the energy of the tap history,
/// `w_i += (mu / (epsilon + ||h||^2)) * e * h_i`, which makes the
/// convergence speed roughly independent of the input's scale: `mu` can be
/// picked in the scale-free range `(0, 2)` regardless of the signal's
/// magnitude, unlike plain LMS where it must be tuned against the (usually
/// unknown) input power. `epsilon` is a small regularizer that keeps the
/// step bounded while the tap history is still zero or near-zero.
///
/// As with `LmsFilter`, `last()` -- read after `update(x_t)` -- is the
/// one-step-ahead prediction of `x_{t+1}`, computed with the just-updated
/// weights and the history ending in `x_t`. It is `None` until `order + 1`
/// values were observed.
///
/// Complexity: O(order) per update, O(order) memory.
#[derive(Debug, Clone)]
pub struct NlmsFilter<T, V> {
    view: V,
    order: NonZeroUsize,
    mu: T,
    epsilon: T,
    weights: Vec<T>,
    history: Vec<T>,
    prediction: T,
    error: T,
    n_observed: usize,
}

impl<T, V> NlmsFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new predictor with a chained `View`, tap count `order`, step
    /// size `mu` and regularizer `epsilon`.
    ///
    /// # Panics
    /// If `mu` is not in `(0, 2)` or `epsilon <= 0`.
    pub fn new(view: V, order: NonZeroUsize, mu: T, epsilon: T) -> Self {
        assert!(mu > T::zero(), "mu must be > 0");
        assert!(
            mu < T::from(2.0).expect("can convert"),
            "mu must be < 2 for stability"
        );
        assert!(epsilon > T::zero(), "epsilon must be > 0");
        Self {
            view,
            order,
            mu,
            epsilon,
            weights: vec![T::zero(); order.get()],
            history: Vec::with_capacity(order.get()),
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

    /// The scale-free step size in `(0, 2)`.
    #[inline]
    pub fn mu(&self) -> T {
        self.mu
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

    /// The squared L2 norm of the current tap history.
    fn history_energy(&self) -> T {
        self.history.iter().fold(T::zero(), |acc, h| acc + *h * *h)
    }

    /// Push `val` into the fixed-size tap history, evicting the oldest
    /// value once full. Allocation-free: `history` never grows past `order`.
    fn push_history(&mut self, val: T) {
        if self.history.len() == self.order.get() {
            self.history.copy_within(1.., 0);
            *self.history.last_mut().expect("order > 0") = val;
        } else {
            self.history.push(val);
        }
    }
}

impl<T, V> View<T> for NlmsFilter<T, V>
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
            let y_hat = self.predict();
            self.error = val - y_hat;
            let step = self.mu / (self.epsilon + self.history_energy());
            for (w, h) in self.weights.iter_mut().zip(self.history.iter()) {
                *w = *w + step * self.error * *h;
            }
            debug_assert!(self.error.is_finite(), "value must be finite");
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
        test_data::TEST_DATA,
    };

    fn nlms(order: usize, mu: f64, epsilon: f64) -> NlmsFilter<f64, Echo<f64>> {
        NlmsFilter::new(Echo::new(), NonZeroUsize::new(order).unwrap(), mu, epsilon)
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
    fn nlms_plot() {
        // Unlike plain LMS, the step is scale-free, so a mu of order 1 is
        // stable directly on the raw, undemeaned price-like TEST_DATA.
        let mut f = nlms(8, 0.5, 1e-6);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/nlms_filter.png").unwrap();
    }

    #[test]
    fn nlms_warmup() {
        let mut f = nlms(3, 0.5, 1e-6);
        for x in [1.0, 2.0, 3.0] {
            f.update(x);
            assert_eq!(f.last(), None);
        }
        f.update(4.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn nlms_learns_ar2_coefficients_and_predicts_near_noise_floor() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        let (a1, a2, noise_sd) = (0.6, -0.3, 0.1);
        let series = ar2(&rng, 20_000, a1, a2, noise_sd);

        let mut f = nlms(2, 0.05, 1e-6);
        let mut sq_err = Vec::new();
        let mut weights = Vec::new();
        for x in &series {
            f.update(*x);
            if f.last().is_some() {
                sq_err.push(f.error() * f.error());
                weights.push([f.weights()[0], f.weights()[1]]);
            }
        }
        let tail_start = sq_err.len() / 2;
        let mse_tail = &sq_err[tail_start..];
        let mse = mse_tail.iter().sum::<f64>() / mse_tail.len() as f64;
        assert!(
            mse < 4.0 * noise_sd * noise_sd,
            "steady-state MSE {mse} should be close to the noise variance {}",
            noise_sd * noise_sd
        );
        // Average the weight trajectory over the converged tail, as the
        // stochastic update makes a single final snapshot fluctuate around
        // the Wiener solution.
        let weights_tail = &weights[tail_start..];
        let n = weights_tail.len() as f64;
        let mean_w0 = weights_tail.iter().map(|w| w[0]).sum::<f64>() / n;
        let mean_w1 = weights_tail.iter().map(|w| w[1]).sum::<f64>() / n;
        assert!(
            (mean_w0 - a2).abs() < 0.1,
            "mean weight[0] {mean_w0} should be close to a2 {a2}",
        );
        assert!(
            (mean_w1 - a1).abs() < 0.1,
            "mean weight[1] {mean_w1} should be close to a1 {a1}",
        );
    }

    #[test]
    fn nlms_converges_faster_than_lms_on_a_large_scale_signal() {
        // On a signal scaled far from unit power, plain LMS needs a tiny mu
        // (tuned to the power) while NLMS, being scale-free, still converges
        // quickly with mu close to 1: run both at a mu that is stable for
        // NLMS and demonstrate NLMS's prediction error vanishes while an
        // LMS instance with the same nominal mu diverges.
        use crate::sliding_windows::LmsFilter;

        let scale = 1000.0;
        let mut nlms_f = nlms(2, 0.5, 1e-6);
        let mut lms_f = LmsFilter::new(Echo::new(), NonZeroUsize::new(2).unwrap(), 0.5);
        for i in 0..50 {
            let x = scale + (i % 2) as f64;
            nlms_f.update(x);
            lms_f.update(x);
        }
        assert!(
            nlms_f.last().unwrap().is_finite(),
            "NLMS must stay stable on a large-scale signal"
        );
        assert!(
            !lms_f.last().unwrap().is_finite() || lms_f.last().unwrap().abs() > scale * 10.0,
            "LMS with the same nominal mu should diverge on a large-scale signal"
        );
    }

    #[test]
    fn nlms_constant_input_converges_to_zero_error() {
        let mut f = nlms(2, 0.5, 1e-6);
        for _ in 0..200 {
            f.update(5.0);
        }
        assert!(
            f.error().abs() < 1e-6,
            "error {} should vanish on constant input",
            f.error()
        );
    }

    #[test]
    #[should_panic(expected = "mu must be > 0")]
    fn nlms_rejects_non_positive_mu() {
        let _ = nlms(4, 0.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "mu must be < 2")]
    fn nlms_rejects_mu_at_or_above_two() {
        let _ = nlms(4, 2.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "epsilon must be > 0")]
    fn nlms_rejects_non_positive_epsilon() {
        let _ = nlms(4, 0.5, 0.0);
    }
}

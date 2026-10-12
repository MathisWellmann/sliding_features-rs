//! LMS (Least Mean Squares) adaptive linear predictor.

use std::num::NonZeroUsize;

use num::Float;

use crate::View;

/// An LMS (least mean squares) adaptive linear predictor (Widrow & Hoff,
/// "Adaptive Switching Circuits", 1960) with `order` taps and step size
/// `mu`.
///
/// On every update it first predicts the incoming value from the `order`
/// values observed before it, `y_hat = w . [x_{t-1}, ..., x_{t-order}]`,
/// forms the a-priori prediction error `e = x_t - y_hat`, and adapts the
/// weights with the stochastic gradient step `w_i += mu * e * x_{t-i}`.
/// After that, `x_t` enters the tap history, so `last()` -- read *after*
/// `update(x_t)` -- is the one-step-ahead prediction of `x_{t+1}`, computed
/// with the just-updated weights and the history ending in `x_t`.
///
/// `mu` trades convergence speed against steady-state misadjustment; it
/// must be positive and, for stability, well below `2 / (order * input
/// power)` (not enforced here, as the input power is not known upfront).
///
/// `last()` is `None` until `order + 1` values were observed (the first
/// `order` values only fill the tap history; the first error and weight
/// update happen on the `(order + 1)`-th). `error()` is the last a-priori
/// prediction error and `weights()` the current tap weights.
///
/// Complexity: O(order) per update, O(order) memory.
#[derive(Debug, Clone)]
pub struct LmsFilter<T, V> {
    view: V,
    order: NonZeroUsize,
    mu: T,
    /// Tap weights, applied to `history` (oldest first) to predict the next value.
    weights: Vec<T>,
    /// The last `order` observed values (or fewer, while filling up), oldest first.
    history: Vec<T>,
    prediction: T,
    error: T,
    n_observed: usize,
}

impl<T, V> LmsFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new predictor with a chained `View`, tap count `order` and
    /// step size `mu`.
    ///
    /// # Panics
    /// If `mu <= 0`.
    pub fn new(view: V, order: NonZeroUsize, mu: T) -> Self {
        assert!(mu > T::zero(), "mu must be > 0");
        Self {
            view,
            order,
            mu,
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

    /// The step size controlling the adaptation speed.
    #[inline]
    pub fn mu(&self) -> T {
        self.mu
    }

    /// The last a-priori prediction error (actual minus predicted), `None`
    /// until a first correction was made.
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
        if self.history.len() == self.order.get() {
            self.history.copy_within(1.., 0);
            *self.history.last_mut().expect("order > 0") = val;
        } else {
            self.history.push(val);
        }
    }
}

impl<T, V> View<T> for LmsFilter<T, V>
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
            for (w, h) in self.weights.iter_mut().zip(self.history.iter()) {
                *w = *w + self.mu * self.error * *h;
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

    fn lms(order: usize, mu: f64) -> LmsFilter<f64, Echo<f64>> {
        LmsFilter::new(Echo::new(), NonZeroUsize::new(order).unwrap(), mu)
    }

    /// Simulate an AR(2) process `x_t = a1 x_{t-1} + a2 x_{t-2} + noise`.
    fn ar2(rng: &romu::Rng, n: usize, a1: f64, a2: f64, noise_sd: f64) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 2..n {
            // Box-Muller transform for Gaussian noise. Clamp u1 away from
            // zero so `ln(u1)` stays finite.
            let u1 = (1.0 - rng.f64()).max(f64::MIN_POSITIVE);
            let u2 = rng.f64();
            let noise =
                noise_sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x[i] = a1 * x[i - 1] + a2 * x[i - 2] + noise;
        }
        x
    }

    #[test]
    fn lms_plot() {
        // TEST_DATA is raw price-like data around 90-100, undemeaned, so mu
        // must be tiny for stability (bound ~ 2 / (order * mean(x^2))).
        let mut f = lms(8, 1e-6);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/lms_filter.png").unwrap();
    }

    #[test]
    fn lms_warmup() {
        let mut f = lms(3, 0.1);
        for x in [1.0, 2.0, 3.0] {
            f.update(x);
            assert_eq!(f.last(), None);
        }
        f.update(4.0);
        assert!(f.last().is_some());
    }

    #[test]
    fn lms_learns_ar2_coefficients_and_predicts_near_noise_floor() {
        let rng = romu::Rng::from_seed_with_64bit(10);
        let (a1, a2, noise_sd) = (0.6, -0.3, 0.1);
        let series = ar2(&rng, 20_000, a1, a2, noise_sd);

        let mut f = lms(2, 0.05);
        let mut sq_err = Vec::new();
        for x in &series {
            f.update(*x);
            if f.last().is_some() {
                sq_err.push(f.error() * f.error());
            }
        }
        // Average over the tail, once the weights have converged.
        let tail = &sq_err[sq_err.len() / 2..];
        let mse = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!(
            mse < 4.0 * noise_sd * noise_sd,
            "steady-state MSE {mse} should be close to the noise variance {}",
            noise_sd * noise_sd
        );
        // history is oldest-first, so weights()[0] predicts from x_{t-2}
        // (coefficient a2) and weights()[1] from x_{t-1} (coefficient a1).
        assert!(
            (f.weights()[0] - a2).abs() < 0.1,
            "weight[0] {} should be close to a2 {a2}",
            f.weights()[0]
        );
        assert!(
            (f.weights()[1] - a1).abs() < 0.1,
            "weight[1] {} should be close to a1 {a1}",
            f.weights()[1]
        );
    }

    #[test]
    fn lms_constant_input_converges_to_zero_error() {
        let mut f = lms(2, 0.005);
        for _ in 0..200 {
            f.update(5.0);
        }
        assert!(
            f.error().abs() < 1e-6,
            "error {} should vanish on constant input",
            f.error()
        );
        assert!(
            (f.last().unwrap() - 5.0).abs() < 1e-6,
            "prediction {} should converge to the constant",
            f.last().unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "mu must be > 0")]
    fn lms_rejects_non_positive_mu() {
        let _ = lms(4, 0.0);
    }
}

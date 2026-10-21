// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Scalar Kalman filter for the local-level (random walk) model.

use getset::CopyGetters;
use num::Float;

use crate::View;

/// A scalar Kalman filter for the local-level model: the true state follows
/// a random walk `x_t = x_{t-1} + w_t`, `w_t ~ N(0, q)`, and is observed
/// with noise `z_t = x_t + v_t`, `v_t ~ N(0, r)` (Kalman, "A New Approach to
/// Linear Filtering and Prediction Problems", 1960, specialized to one state
/// and no control input).
///
/// `q` is the process variance (how much the true state is expected to
/// drift per unit time step) and `r` is the measurement variance (how noisy
/// the observations are), both in squared units of the input. Each update
/// performs the predict step (`x_pred = x`, `p_pred = p + q`, since the
/// state transition is the identity) followed by the correct step
/// (`k = p_pred / (p_pred + r)`, `x = x_pred + k (z - x_pred)`,
/// `p = (1 - k) p_pred`).
///
/// `last()` is `Some` from the first update onward: the first measurement
/// seeds the state directly, with the estimate variance set to `r`.
/// `variance()` returns the current estimate variance and `gain()` the last
/// Kalman gain, both useful for monitoring convergence.
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Kalman1d<T: Float, V> {
    view: V,
    /// The process variance: expected squared drift of the state per time step.
    #[getset(get_copy = "pub")]
    process_variance: T,
    /// The measurement variance: expected squared noise of the observations.
    #[getset(get_copy = "pub")]
    measurement_variance: T,
    /// The current estimate variance.
    #[getset(get_copy = "pub")]
    variance: T,
    /// The last computed Kalman gain, in `[0, 1]`.
    #[getset(get_copy = "pub")]
    gain: T,
    state: Option<T>,
}

impl<T, V> Kalman1d<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new filter with a chained `View`, a process variance `q` and
    /// a measurement variance `r`.
    ///
    /// # Panics
    /// If `q <= 0` or `r <= 0`.
    pub fn new(view: V, q: T, r: T) -> Self {
        assert!(q > T::zero(), "process variance q must be > 0");
        assert!(r > T::zero(), "measurement variance r must be > 0");
        Self {
            view,
            process_variance: q,
            measurement_variance: r,
            variance: T::zero(),
            gain: T::zero(),
            state: None,
        }
    }
}

impl<T, V> View<T> for Kalman1d<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(state) = self.state else {
            // Seed the state with the first measurement; its uncertainty is
            // exactly the measurement noise, as there is no prior estimate.
            self.state = Some(val);
            self.variance = self.measurement_variance;
            return;
        };

        let predicted_variance = self.variance + self.process_variance;
        self.gain = predicted_variance / (predicted_variance + self.measurement_variance);
        debug_assert!(self.gain >= T::zero(), "gain must be >= 0");
        debug_assert!(self.gain <= T::one(), "gain must be <= 1");
        self.state = Some(state + self.gain * (val - state));
        self.variance = (T::one() - self.gain) * predicted_variance;
        debug_assert!(self.variance >= T::zero(), "variance must be >= 0");
        debug_assert!(
            self.state.expect("just set").is_finite(),
            "value must be finite"
        );
    }

    fn last(&self) -> Option<T> {
        self.state
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

    fn kf(q: f64, r: f64) -> Kalman1d<f64, Echo<f64>> {
        Kalman1d::new(Echo::new(), q, r)
    }

    #[test]
    fn kalman1d_plot() {
        let mut f = kf(0.01, 1.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/kalman1d.png").unwrap();
    }

    #[test]
    fn kalman1d_ready_from_first_update() {
        let mut f = kf(0.01, 1.0);
        assert_eq!(f.last(), None);
        f.update(3.0);
        assert_eq!(f.last(), Some(3.0));
        assert_eq!(f.variance(), 1.0);
    }

    #[test]
    fn kalman1d_variance_converges_to_closed_form_steady_state() {
        // Steady-state variance of the local-level Kalman filter solves
        // p = (p + q) r / (p + q + r), whose positive root is
        // p* = (-q + sqrt(q^2 + 4 q r)) / 2 (see e.g. Durbin & Koopman,
        // "Time Series Analysis by State Space Methods").
        let (q, r) = (0.05, 2.0);
        let steady_state = (-q + (q * q + 4.0 * q * r).sqrt()) / 2.0;
        // The gain uses the *predicted* variance (posterior variance + q),
        // not the posterior variance itself.
        let steady_state_predicted = steady_state + q;
        let steady_state_gain = steady_state_predicted / (steady_state_predicted + r);

        let rng = romu::Rng::from_seed_with_64bit(3);
        let mut f = kf(q, r);
        let truth = 42.0;
        // Uniform(-a, a) has variance a^2 / 3, so a = sqrt(3 r) gives noise
        // variance exactly r.
        let half_range = (3.0 * r).sqrt();
        for _ in 0..20_000 {
            let noise = (rng.f64() - 0.5) * 2.0 * half_range;
            f.update(truth + noise);
        }
        assert!(
            (f.variance() - steady_state).abs() < 1e-3,
            "variance {} should converge to the steady state {steady_state}",
            f.variance()
        );
        assert!(
            (f.gain() - steady_state_gain).abs() < 1e-3,
            "gain {} should converge to the steady-state gain {steady_state_gain}",
            f.gain()
        );
    }

    #[test]
    fn kalman1d_converges_to_the_true_constant() {
        let rng = romu::Rng::from_seed_with_64bit(4);
        let mut f = kf(1e-6, 1.0);
        let truth = -7.5;
        for _ in 0..5000 {
            f.update(truth + (rng.f64() - 0.5) * 2.0);
        }
        assert!(
            (f.last().unwrap() - truth).abs() < 0.2,
            "estimate {} should converge close to the true constant {truth}",
            f.last().unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "process variance q must be > 0")]
    fn kalman1d_rejects_non_positive_q() {
        let _ = kf(0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "measurement variance r must be > 0")]
    fn kalman1d_rejects_non_positive_r() {
        let _ = kf(0.1, 0.0);
    }
}

//! Alpha-beta (g-h) tracker: a two-state Kalman-like filter with fixed gains.

use getset::CopyGetters;
use num::Float;

use crate::View;

/// The alpha-beta (g-h) tracker, a constant-velocity position/velocity
/// estimator with fixed gains, as covered for trading data in Ehlers,
/// "Kalman and Alpha-Beta Filters" (a simplified, fixed-gain relative of the
/// Kalman filter predating it, see e.g. Benedict & Bordner, 1962).
///
/// Each update is a one unit time step predict-correct cycle:
/// `x_pred = x + v`, `v_pred = v` (constant-velocity model), then
/// `residual = measurement - x_pred`, `x = x_pred + alpha * residual`,
/// `v = v_pred + beta * residual` (`beta` already absorbs the `1 / dt` of
/// the textbook formula, since `dt = 1` here).
///
/// `alpha` weighs how much the position estimate trusts a new measurement;
/// `beta` does the same for the velocity estimate. The pair must lie in the
/// classical stability region `0 < alpha <= 1`, `0 < beta <= 2` and
/// `4 - 2 alpha - beta > 0`.
///
/// `last()` is `Some` from the first update onward (there is no window to
/// fill): the first measurement seeds the position with zero velocity.
/// `velocity()` returns the current velocity estimate.
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct AlphaBetaFilter<T: Float, V> {
    view: V,
    /// The position gain in `(0, 1]`.
    #[getset(get_copy = "pub")]
    alpha: T,
    /// The velocity gain in `(0, 2]`.
    #[getset(get_copy = "pub")]
    beta: T,
    /// The current velocity estimate, in units per time step.
    #[getset(get_copy = "pub")]
    velocity: T,
    position: Option<T>,
}

impl<T, V> AlphaBetaFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new tracker with a chained `View` and gains `alpha`, `beta`.
    ///
    /// # Panics
    /// If the gains fall outside the stability region
    /// `0 < alpha <= 1`, `0 < beta <= 2`, `4 - 2 alpha - beta > 0`.
    pub fn new(view: V, alpha: T, beta: T) -> Self {
        assert!(alpha > T::zero(), "alpha must be > 0");
        assert!(alpha <= T::one(), "alpha must be <= 1");
        assert!(beta > T::zero(), "beta must be > 0");
        assert!(
            beta <= T::from(2.0).expect("can convert"),
            "beta must be <= 2"
        );
        let four = T::from(4.0).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        assert!(
            four - two * alpha - beta > T::zero(),
            "alpha and beta must satisfy 4 - 2 alpha - beta > 0"
        );
        Self {
            view,
            alpha,
            beta,
            velocity: T::zero(),
            position: None,
        }
    }
}

impl<T, V> View<T> for AlphaBetaFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(position) = self.position else {
            // Seed the state with the first measurement and zero velocity,
            // as there is no prior estimate to predict from yet.
            self.position = Some(val);
            return;
        };

        let predicted_position = position + self.velocity;
        let residual = val - predicted_position;
        self.position = Some(predicted_position + self.alpha * residual);
        self.velocity = self.velocity + self.beta * residual;
        debug_assert!(
            self.position.expect("just set").is_finite(),
            "value must be finite"
        );
        debug_assert!(self.velocity.is_finite(), "value must be finite");
    }

    fn last(&self) -> Option<T> {
        self.position
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

    fn ab(alpha: f64, beta: f64) -> AlphaBetaFilter<f64, Echo<f64>> {
        AlphaBetaFilter::new(Echo::new(), alpha, beta)
    }

    #[test]
    fn alpha_beta_plot() {
        let mut f = ab(0.5, 0.2);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/alpha_beta_filter.png").unwrap();
    }

    #[test]
    fn alpha_beta_ready_from_first_update() {
        let mut f = ab(0.5, 0.2);
        assert_eq!(f.last(), None);
        f.update(3.0);
        assert_eq!(f.last(), Some(3.0));
        assert_eq!(f.velocity(), 0.0);
    }

    #[test]
    fn alpha_beta_tracks_ramp_with_zero_steady_state_velocity_error() {
        // A noise-free ramp of slope `v`: once the velocity estimate has
        // settled, the filter must track the true velocity exactly, as it
        // has no model mismatch to correct for.
        let slope = 2.5;
        let mut f = ab(0.5, 0.2);
        for i in 0..200 {
            f.update(10.0 + slope * i as f64);
        }
        assert!(
            (f.velocity() - slope).abs() < 1e-9,
            "velocity {} should equal the ramp's slope {slope}",
            f.velocity()
        );
    }

    #[test]
    fn alpha_beta_smooths_noisy_constant() {
        let rng = romu::Rng::from_seed_with_64bit(2);
        let mut f = ab(0.2, 0.05);
        let mut last = 0.0;
        for _ in 0..1000 {
            let x = 5.0 + (rng.f64() - 0.5) * 4.0;
            f.update(x);
            last = f.last().unwrap();
        }
        assert!(
            (last - 5.0).abs() < 1.0,
            "filtered estimate {last} should be close to the true constant 5.0"
        );
    }

    #[test]
    #[should_panic(expected = "alpha must be > 0")]
    fn alpha_beta_rejects_non_positive_alpha() {
        let _ = ab(0.0, 0.2);
    }

    #[test]
    #[should_panic(expected = "alpha must be <= 1")]
    fn alpha_beta_rejects_alpha_above_one() {
        let _ = ab(1.1, 0.2);
    }

    #[test]
    #[should_panic(expected = "beta must be <= 2")]
    fn alpha_beta_rejects_beta_above_two() {
        let _ = ab(0.5, 2.1);
    }

    #[test]
    #[should_panic(expected = "4 - 2 alpha - beta > 0")]
    fn alpha_beta_rejects_unstable_region() {
        let _ = ab(1.0, 2.0);
    }
}

//! Exponential moving average with Huber-clipped residuals, robust to outliers.

use getset::CopyGetters;
use num::Float;

use crate::View;

/// An exponential moving average whose update is robustified with Huber's
/// (1964, "Robust Estimation of a Location Parameter") clipped score
/// function, following the robust exponential smoothing scheme of Gelper,
/// Fried & Croux (2010, "Robust Forecasting with Exponential and
/// Holt-Winters Smoothing", Journal of Forecasting): a plain EMA update
/// replaces the raw residual `e_t = x_t - L_{t-1}` by a clipped version
/// `psi(e_t) = clip(e_t, -k s_t, k s_t)`, where `s_t` is itself an EMA of
/// `|e_t|` (a robust, slowly-adapting scale estimate). This bounds the
/// influence of any single outlier on the level while leaving normal
/// updates essentially unchanged (`psi(e) = e` whenever `|e| <= k s_t`).
///
/// `level_alpha` smooths the level (`L_t = L_{t-1} + level_alpha * psi(e_t)`)
/// and `scale_alpha` smooths the scale
/// (`s_t = (1 - scale_alpha) s_{t-1} + scale_alpha |e_t|`), both in `(0, 1]`.
/// `k` is the Huber clipping threshold in robust-scale units (typical values
/// are 1 to 3; a larger `k` clips less and approaches a plain EMA).
///
/// `last()` is `Some` from the first update onward: the first observation
/// seeds the level and the scale starts at zero (no clipping until the
/// second update, as there is no scale estimate yet).
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct HuberEma<T: Float, V> {
    view: V,
    /// The smoothing factor for the level, in `(0, 1]`.
    #[getset(get_copy = "pub")]
    level_alpha: T,
    /// The smoothing factor for the robust scale, in `(0, 1]`.
    #[getset(get_copy = "pub")]
    scale_alpha: T,
    /// The Huber clipping threshold, in robust-scale units.
    #[getset(get_copy = "pub")]
    k: T,
    /// The current robust scale estimate (an EMA of the absolute residuals).
    #[getset(get_copy = "pub")]
    scale: T,
    level: Option<T>,
}

impl<T, V> HuberEma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new filter with a chained `View`, level and scale smoothing
    /// factors and a Huber clipping threshold `k`.
    ///
    /// # Panics
    /// If `level_alpha` or `scale_alpha` are not in `(0, 1]`, or `k <= 0`.
    pub fn new(view: V, level_alpha: T, scale_alpha: T, k: T) -> Self {
        assert!(level_alpha > T::zero(), "level_alpha must be > 0");
        assert!(level_alpha <= T::one(), "level_alpha must be <= 1");
        assert!(scale_alpha > T::zero(), "scale_alpha must be > 0");
        assert!(scale_alpha <= T::one(), "scale_alpha must be <= 1");
        assert!(k > T::zero(), "k must be > 0");
        Self {
            view,
            level_alpha,
            scale_alpha,
            k,
            scale: T::zero(),
            level: None,
        }
    }
}

impl<T, V> View<T> for HuberEma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(level) = self.level else {
            self.level = Some(val);
            return;
        };

        let residual = val - level;
        let clipped = if self.scale > T::zero() {
            let bound = self.k * self.scale;
            residual.max(-bound).min(bound)
        } else {
            // No scale estimate yet: nothing to clip against.
            residual
        };
        self.level = Some(level + self.level_alpha * clipped);
        self.scale = (T::one() - self.scale_alpha) * self.scale + self.scale_alpha * residual.abs();
        debug_assert!(
            self.level.expect("just set").is_finite(),
            "value must be finite"
        );
        debug_assert!(self.scale.is_finite(), "value must be finite");
        debug_assert!(self.scale >= T::zero(), "scale must be >= 0");
    }

    fn last(&self) -> Option<T> {
        self.level
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    fn huber_ema(level_alpha: f64, scale_alpha: f64, k: f64) -> HuberEma<f64, Echo<f64>> {
        HuberEma::new(Echo::new(), level_alpha, scale_alpha, k)
    }

    #[test]
    fn huber_ema_plot() {
        let mut f = huber_ema(0.1, 0.1, 3.0);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            f.update(*v);
            if let Some(v) = f.last() {
                out.push(v);
            }
        }
        plot_values(out, "img/huber_ema.png").unwrap();
    }

    #[test]
    fn huber_ema_ready_from_first_update() {
        let mut f = huber_ema(0.2, 0.2, 2.0);
        assert_eq!(f.last(), None);
        f.update(3.0);
        assert_eq!(f.last(), Some(3.0));
        assert_eq!(f.scale(), 0.0);
    }

    #[test]
    fn huber_ema_robust_to_a_single_large_outlier() {
        let rng = romu::Rng::from_seed_with_64bit(20);
        let mut huber = huber_ema(0.1, 0.1, 3.0);
        let mut ema = Ema::with_alpha(Echo::new(), NonZeroUsize::new(1).unwrap(), 0.1);

        for i in 0..200 {
            let x = if i == 100 {
                1000.0 // A single, large outlier injected into an otherwise quiet series.
            } else {
                10.0 + (rng.f64() - 0.5) * 0.2
            };
            huber.update(x);
            ema.update(x);
        }
        let huber_level = huber.last().unwrap();
        let ema_level = ema.last().unwrap();
        assert!(
            (huber_level - 10.0).abs() < (ema_level - 10.0).abs(),
            "HuberEma level {huber_level} should stay closer to 10.0 than plain Ema's {ema_level}"
        );
    }

    #[test]
    fn huber_ema_constant_input() {
        let mut f = huber_ema(0.3, 0.3, 2.0);
        for _ in 0..20 {
            f.update(7.5);
        }
        assert_eq!(f.last(), Some(7.5));
        assert_eq!(f.scale(), 0.0);
    }

    #[test]
    #[should_panic(expected = "level_alpha must be > 0")]
    fn huber_ema_rejects_non_positive_level_alpha() {
        let _ = huber_ema(0.0, 0.1, 2.0);
    }

    #[test]
    #[should_panic(expected = "scale_alpha must be <= 1")]
    fn huber_ema_rejects_scale_alpha_above_one() {
        let _ = huber_ema(0.1, 1.1, 2.0);
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn huber_ema_rejects_non_positive_k() {
        let _ = huber_ema(0.1, 0.1, 0.0);
    }
}

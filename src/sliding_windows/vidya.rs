//! VIDYA - Variable Index Dynamic Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::Cmo,
};

/// VIDYA - Chande's Variable Index Dynamic Average (Tushar Chande, "Adapting
/// Moving Averages To Market Volatility", Technical Analysis of Stocks &
/// Commodities, 1992): an EMA whose smoothing factor scales with the
/// market's volatility index, here `|CMO| / 100` (`CMO` the
/// [`crate::sliding_windows::Cmo`] over the same `window_len`), so it moves
/// close to the input in a strong trend and barely moves in a flat market.
///
/// `VIDYA = alpha * k * x + (1 - alpha * k) * VIDYA[1]`, with `alpha = 2 /
/// (window_len + 1)` and `k = |CMO| / 100 in [0, 1]`, seeded with the first
/// value for which `CMO` is available.
///
/// `last()` is `None` until the inner `CMO` is ready, i.e. after
/// `window_len + 1` input values; output is unbounded, following the input.
///
/// Complexity: O(1) per update (the inner `CMO` is O(1)); O(window_len)
/// memory for its window.
#[derive(Debug, Clone, CopyGetters)]
pub struct Vidya<T, V> {
    view: V,
    /// The non-zero window length, used both for `alpha` and for the inner
    /// `CMO`.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    alpha: T,
    cmo: Cmo<T, Echo<T>>,
    last_vidya: Option<T>,
}

impl<T, V> Vidya<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new VIDYA with a chained `View` and a window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        let two = T::from(2.0).expect("can convert");
        let alpha = two / (T::from(window_len.get()).expect("can convert") + T::one());
        Self {
            view,
            window_len,
            alpha,
            cmo: Cmo::new(Echo::new(), window_len),
            last_vidya: None,
        }
    }
}

impl<T, V> View<T> for Vidya<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.cmo.update(val);
        let Some(cmo) = self.cmo.last() else { return };
        let hundred = T::from(100.0).expect("can convert");
        let k = (cmo / hundred).abs();
        debug_assert!((T::zero()..=T::one()).contains(&k), "k is in [0, 1]");

        let factor = self.alpha * k;
        self.last_vidya = Some(match self.last_vidya {
            None => val,
            Some(prev) => factor * val + (T::one() - factor) * prev,
        });
    }

    fn last(&self) -> Option<T> {
        self.last_vidya.inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    #[test]
    fn vidya_warms_up() {
        let mut vidya = Vidya::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            vidya.update(v);
            assert_eq!(vidya.last(), None);
        }
        vidya.update(6.0);
        assert!(vidya.last().is_some());
    }

    #[test]
    fn vidya_constant_input_is_the_constant() {
        let mut vidya = Vidya::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..20 {
            vidya.update(4.5);
        }
        assert_eq!(vidya.last(), Some(4.5));
    }

    /// Naive reference: the CMO formula plus the adaptive EMA recursion,
    /// applied directly, step by step.
    #[test]
    fn vidya_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(234);
        let window_len = 8usize;
        let alpha = 2.0 / (window_len as f64 + 1.0);
        let mut price = 100.0_f64;
        let vals: Vec<f64> = (0..300)
            .map(|_| {
                price *= 1.0 + 0.01 * (rng.f64() - 0.5);
                price
            })
            .collect();

        let mut vidya = Vidya::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let mut naive: Option<f64> = None;
        for (i, v) in vals.iter().enumerate() {
            vidya.update(*v);
            if i < window_len {
                assert_eq!(vidya.last(), None);
                continue;
            }
            let window = &vals[i - window_len..=i];
            let (mut up, mut down) = (0.0, 0.0);
            for w in window.windows(2) {
                let diff = w[1] - w[0];
                if diff > 0.0 {
                    up += diff;
                } else {
                    down += -diff;
                }
            }
            let cmo = if up + down == 0.0 {
                0.0
            } else {
                100.0 * (up - down) / (up + down)
            };
            let k = (cmo / 100.0).abs();
            let factor = alpha * k;
            naive = Some(match naive {
                None => *v,
                Some(prev) => factor * v + (1.0 - factor) * prev,
            });
            let got = vidya.last().expect("ready");
            assert!((got - naive.expect("set above")).abs() < 1e-9, "step {i}");
        }
    }

    /// With no volatility (a flat window), the smoothing factor is 0 and
    /// VIDYA freezes at its previous value, ignoring an isolated outlier.
    #[test]
    fn vidya_freezes_when_volatility_index_is_zero() {
        let mut vidya = Vidya::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in [5.0, 5.0, 5.0, 5.0, 5.0] {
            vidya.update(v);
        }
        assert_eq!(vidya.last(), Some(5.0));
        // One more flat value keeps CMO at 0, so VIDYA cannot move at all.
        vidya.update(5.0);
        assert_eq!(vidya.last(), Some(5.0));
    }

    #[test]
    fn vidya_plot() {
        let mut vidya = Vidya::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            vidya.update(*v);
            if let Some(val) = vidya.last() {
                out.push(val);
            }
        }
        let filename = "img/vidya.png";
        plot_values(out, filename).unwrap();
    }
}

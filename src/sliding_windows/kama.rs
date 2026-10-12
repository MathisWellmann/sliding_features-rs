//! KAMA - Kaufman's Adaptive Moving Average

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::{
    View,
    pure_functions::Echo,
    sliding_windows::KaufmanEfficiencyRatio,
};

/// KAMA - Kaufman's Adaptive Moving Average (Perry J. Kaufman, "Smarter
/// Trading", 1995): an EMA whose smoothing constant adapts to the
/// [`KaufmanEfficiencyRatio`] (ER) of the input, moving as fast as a short
/// EMA in a clean trend and as slow as a long EMA in noisy, directionless
/// markets.
///
/// `sc = (ER * (fast_sc - slow_sc) + slow_sc)^2`, with `fast_sc = 2 /
/// (fast_period + 1)` and `slow_sc = 2 / (slow_period + 1)` (defaults
/// `fast_period = 2`, `slow_period = 30`, per the original article).
/// `KAMA = KAMA[1] + sc * (x - KAMA[1])`, seeded with the first value for
/// which `ER` is available.
///
/// `last()` is `None` until the inner efficiency ratio is ready, i.e. after
/// `window_len + 1` input values; output is unbounded, following the input.
///
/// Complexity: O(1) per update (the inner efficiency ratio is O(1)); O(n)
/// memory for its window.
#[derive(Debug, Clone, CopyGetters)]
pub struct Kama<T, V> {
    view: V,
    /// The non-zero efficiency-ratio window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The fast period behind `fast_sc`.
    #[getset(get_copy = "pub")]
    fast_period: NonZeroUsize,
    /// The slow period behind `slow_sc`.
    #[getset(get_copy = "pub")]
    slow_period: NonZeroUsize,
    fast_sc: T,
    slow_sc: T,
    er: KaufmanEfficiencyRatio<T, Echo<T>>,
    last_kama: Option<T>,
}

impl<T, V> Kama<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new KAMA with a chained `View`, an efficiency-ratio window
    /// length and the default periods `fast_period = 2`, `slow_period = 30`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self::with_periods(
            view,
            window_len,
            NonZeroUsize::new(2).expect("2 != 0"),
            NonZeroUsize::new(30).expect("30 != 0"),
        )
    }

    /// Create a new KAMA with a chained `View`, an efficiency-ratio window
    /// length and custom fast/slow periods.
    ///
    /// # Panics
    /// If `fast_period >= slow_period`.
    pub fn with_periods(
        view: V,
        window_len: NonZeroUsize,
        fast_period: NonZeroUsize,
        slow_period: NonZeroUsize,
    ) -> Self {
        assert!(
            fast_period.get() < slow_period.get(),
            "fast_period must be < slow_period"
        );
        let two = T::from(2.0).expect("can convert");
        let fast_sc = two / (T::from(fast_period.get()).expect("can convert") + T::one());
        let slow_sc = two / (T::from(slow_period.get()).expect("can convert") + T::one());
        Self {
            view,
            window_len,
            fast_period,
            slow_period,
            fast_sc,
            slow_sc,
            er: KaufmanEfficiencyRatio::new(Echo::new(), window_len),
            last_kama: None,
        }
    }
}

impl<T, V> View<T> for Kama<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.er.update(val);
        let Some(er) = self.er.last() else { return };
        debug_assert!((T::zero()..=T::one()).contains(&er), "ER is in [0, 1]");

        let sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc).powi(2);
        self.last_kama = Some(match self.last_kama {
            None => val,
            Some(prev) => prev + sc * (val - prev),
        });
    }

    fn last(&self) -> Option<T> {
        self.last_kama.inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        sliding_windows::Ema,
        test_data::TEST_DATA,
    };

    #[test]
    fn kama_warms_up() {
        let mut kama = Kama::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            kama.update(v);
            assert_eq!(kama.last(), None);
        }
        kama.update(6.0);
        assert!(kama.last().is_some());
    }

    #[test]
    fn kama_constant_input_is_the_constant() {
        let mut kama = Kama::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..20 {
            kama.update(8.0);
        }
        assert_eq!(kama.last(), Some(8.0));
    }

    /// Naive reference: the ER formula plus the adaptive EMA recursion,
    /// applied directly, step by step.
    #[test]
    fn kama_matches_naive_reference() {
        let rng = romu::Rng::from_seed_with_64bit(123);
        let window_len = 10usize;
        let (fast, slow) = (2.0, 30.0);
        let fast_sc = 2.0 / (fast + 1.0);
        let slow_sc = 2.0 / (slow + 1.0);
        let mut price = 100.0_f64;
        let vals: Vec<f64> = (0..300)
            .map(|_| {
                price *= 1.0 + 0.01 * (rng.f64() - 0.5);
                price
            })
            .collect();

        let mut kama = Kama::new(Echo::new(), NonZeroUsize::new(window_len).unwrap());
        let mut naive: Option<f64> = None;
        for (i, v) in vals.iter().enumerate() {
            kama.update(*v);
            if i < window_len {
                assert_eq!(kama.last(), None);
                continue;
            }
            let window = &vals[i - window_len..=i];
            let numerator = (window[window_len] - window[0]).abs();
            let denom: f64 = window.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
            let er = if denom == 0.0 { 0.0 } else { numerator / denom };
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);
            naive = Some(match naive {
                None => *v,
                Some(prev) => prev + sc * (v - prev),
            });
            let got = kama.last().expect("ready");
            assert!((got - naive.expect("set above")).abs() < 1e-9, "step {i}");
        }
    }

    /// In a clean trend (ER close to 1), KAMA moves close to the fast EMA,
    /// clearly faster than a slow EMA of the `slow_period` length.
    #[test]
    fn kama_tracks_a_clean_ramp_faster_than_a_slow_ema() {
        let window_len = NonZeroUsize::new(10).unwrap();
        let mut kama = Kama::new(Echo::new(), window_len);
        let mut slow_ema = Ema::new(Echo::new(), NonZeroUsize::new(30).unwrap());
        let ramp: Vec<f64> = (0..200).map(|i| i as f64).collect();
        for v in &ramp {
            kama.update(*v);
            slow_ema.update(*v);
        }
        let i = ramp.len() - 1;
        let kama_lag = i as f64 - kama.last().expect("ready");
        let slow_ema_lag = i as f64 - slow_ema.last().expect("ready");
        assert!(
            kama_lag < slow_ema_lag,
            "kama {kama_lag} slow {slow_ema_lag}"
        );
    }

    #[test]
    #[should_panic(expected = "fast_period must be < slow_period")]
    fn fast_period_not_below_slow_period_is_rejected() {
        let _ = Kama::<f64, _>::with_periods(
            Echo::new(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(30).unwrap(),
            NonZeroUsize::new(30).unwrap(),
        );
    }

    #[test]
    fn kama_plot() {
        let mut kama = Kama::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            kama.update(*v);
            if let Some(val) = kama.last() {
                out.push(val);
            }
        }
        let filename = "img/kama.png";
        plot_values(out, filename).unwrap();
    }
}

//! APO - Absolute Price Oscillator

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::Ema;
use crate::{
    View,
    pure_functions::Echo,
};

/// APO - Absolute Price Oscillator (TA-Lib `APO`).
///
/// `APO = EMA(fast_period) - EMA(slow_period)`, the un-normalized difference
/// of two EMAs; unlike `Macd` there is no signal line.
///
/// `last()` is `None` until the slow EMA is ready (`slow_period` values
/// observed).
///
/// Output range: unbounded (same units as the input).
///
/// Complexity: O(1) per update (two chained EMAs), O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Apo<T, V> {
    view: V,
    /// The fast EMA period.
    #[getset(get_copy = "pub")]
    fast_period: NonZeroUsize,
    /// The slow EMA period.
    #[getset(get_copy = "pub")]
    slow_period: NonZeroUsize,
    ema_fast: Ema<T, Echo<T>>,
    ema_slow: Ema<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Apo<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new APO with a chained `View` and custom fast and slow
    /// periods.
    ///
    /// # Panics
    /// If `fast_period >= slow_period`.
    pub fn with_periods(view: V, fast_period: NonZeroUsize, slow_period: NonZeroUsize) -> Self {
        assert!(
            fast_period.get() < slow_period.get(),
            "the fast period must be shorter than the slow period"
        );
        Self {
            view,
            fast_period,
            slow_period,
            ema_fast: Ema::new(Echo::new(), fast_period),
            ema_slow: Ema::new(Echo::new(), slow_period),
            out: None,
        }
    }

    /// Create a new APO with a chained `View` and the classic default
    /// periods: 12 (fast) and 26 (slow).
    pub fn new(view: V) -> Self {
        Self::with_periods(
            view,
            NonZeroUsize::new(12).expect("12 is non-zero"),
            NonZeroUsize::new(26).expect("26 is non-zero"),
        )
    }
}

impl<T, V> View<T> for Apo<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.ema_fast.update(val);
        self.ema_slow.update(val);
        let (Some(fast), Some(slow)) = (self.ema_fast.last(), self.ema_slow.last()) else {
            return;
        };
        let apo = fast - slow;
        debug_assert!(apo.is_finite(), "`apo` must be finite");
        self.out = Some(apo);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    /// Naive reference: plain EMA recurrence, alpha = 2 / (n + 1).
    struct NaiveEma {
        period: usize,
        val: Option<f64>,
    }
    impl NaiveEma {
        fn new(period: usize) -> Self {
            Self { period, val: None }
        }
        fn update(&mut self, x: f64) -> f64 {
            let alpha = 2.0 / (self.period as f64 + 1.0);
            let out = match self.val {
                None => x,
                Some(prev) => alpha * x + (1.0 - alpha) * prev,
            };
            self.val = Some(out);
            out
        }
    }

    #[test]
    fn apo_plot() {
        let mut apo = Apo::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            apo.update(*v);
            if let Some(val) = apo.last() {
                out.push(val);
            }
        }
        let filename = "img/apo.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn apo_warms_up() {
        let mut apo = Apo::with_periods(
            Echo::new(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(4).unwrap(),
        );
        for (i, v) in [1.0, 2.0, 3.0, 4.0, 5.0].into_iter().enumerate() {
            apo.update(v);
            if i < 3 {
                assert_eq!(apo.last(), None, "step {i}");
            } else {
                assert!(apo.last().is_some());
            }
        }
    }

    #[test]
    fn apo_vs_naive_random() {
        let rng = romu::Rng::from_seed_with_64bit(13);
        let (fast, slow) = (5, 13);
        let mut apo = Apo::with_periods(
            Echo::new(),
            NonZeroUsize::new(fast).unwrap(),
            NonZeroUsize::new(slow).unwrap(),
        );
        let mut naive_fast = NaiveEma::new(fast);
        let mut naive_slow = NaiveEma::new(slow);
        for _ in 0..2_000 {
            let x = rng.f64() * 100.0;
            apo.update(x);
            let naive_apo = naive_fast.update(x) - naive_slow.update(x);
            if let Some(got) = apo.last() {
                assert!((got - naive_apo).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn apo_constant_input_converges_to_zero() {
        let mut apo = Apo::new(Echo::new());
        for _ in 0..500 {
            apo.update(42.0);
        }
        assert!(apo.last().unwrap().abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "fast period must be shorter")]
    fn apo_fast_not_shorter_than_slow_panics() {
        let _ = Apo::with_periods(
            Echo::<f64>::new(),
            NonZeroUsize::new(26).unwrap(),
            NonZeroUsize::new(26).unwrap(),
        );
    }
}

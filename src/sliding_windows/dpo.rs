//! DPO - Detrended Price Oscillator

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Lag,
    Sma,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// DPO - Detrended Price Oscillator.
///
/// Removes the trend from a price series by subtracting an `period`-bar
/// simple moving average from the price `shift` bars ago, where
/// `shift = period / 2 + 1` (integer division): `DPO = x[t - shift] -
/// SMA(period)[t]`.
///
/// This shift convention aligns the average with the middle of the
/// look-back window it was computed over (rather than with "now"), which is
/// what removes the trend component from the price rather than comparing it
/// against an average skewed by recent history.
///
/// `last()` is `None` until both legs are ready: the `period`-bar average
/// and the `shift`-bar-old price (`shift + 1` observations, via the
/// crate's [`Lag`]).
///
/// Output range: unbounded, oscillates around `0`.
///
/// Complexity: O(1) per update, O(`period`) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Dpo<T, V> {
    view: V,
    /// The look-back period of the simple moving average.
    #[getset(get_copy = "pub")]
    period: NonZeroUsize,
    /// How many bars in the past the price is taken from
    /// (`period / 2 + 1`).
    #[getset(get_copy = "pub")]
    shift: NonZeroUsize,
    sma: Sma<T, Echo<T>>,
    lag: Lag<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Dpo<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new DPO with a chained `View` and the moving average
    /// `period`. The price shift is derived as `period / 2 + 1`.
    pub fn new(view: V, period: NonZeroUsize) -> Self {
        let shift = NonZeroUsize::new(period.get() / 2 + 1).expect("period / 2 + 1 is non-zero");
        let lag_window = NonZeroUsize::new(shift.get() + 1).expect("shift + 1 is non-zero");
        Self {
            view,
            period,
            shift,
            sma: Sma::new(Echo::new(), period),
            lag: Lag::new(Echo::new(), lag_window),
            out: None,
        }
    }

    /// Create a new DPO with a chained `View` and the classic default
    /// period of `20`.
    pub fn new_default(view: V) -> Self {
        Self::new(view, NonZeroUsize::new(20).expect("20 is non-zero"))
    }
}

impl<T, V> View<T> for Dpo<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value from `View` must be finite");

        self.sma.update(val);
        self.lag.update(val);

        let Some((shifted, sma)) = self.lag.last().zip(self.sma.last()) else {
            return;
        };
        let out = shifted - sma;
        debug_assert!(out.is_finite(), "`out` must be finite");
        self.out = Some(out);
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

    #[test]
    fn dpo_plot() {
        let mut dpo = Dpo::new_default(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            dpo.update(*v);
            if let Some(val) = dpo.last() {
                out.push(val);
            }
        }
        let filename = "img/dpo.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn dpo_warms_up() {
        let mut dpo = Dpo::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        assert_eq!(dpo.shift().get(), 6);
        let mut seen = false;
        for (i, v) in TEST_DATA.iter().enumerate() {
            dpo.update(*v);
            if i < 9 {
                assert_eq!(dpo.last(), None, "must warm up for `period` values");
            }
            if dpo.last().is_some() {
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn dpo_constant_input_is_zero() {
        let mut dpo = Dpo::new_default(Echo::new());
        for _ in 0..50 {
            dpo.update(42.0);
        }
        assert_eq!(dpo.last(), Some(0.0));
    }

    #[test]
    fn dpo_finite() {
        let mut dpo = Dpo::new_default(Echo::new());
        for v in &TEST_DATA {
            dpo.update(*v);
            if let Some(val) = dpo.last() {
                assert!(val.is_finite(), "val = {val}");
            }
        }
    }

    /// Naive reference, mirroring the SMA and shift of the real
    /// implementation.
    #[test]
    fn dpo_vs_naive_random() {
        use std::collections::VecDeque;

        let rng = romu::Rng::from_seed_with_64bit(31);
        let period = 14_usize;
        let shift = period / 2 + 1;
        let lag_window = shift + 1;
        let mut dpo = Dpo::new(Echo::new(), NonZeroUsize::new(period).unwrap());

        let mut window: VecDeque<f64> = VecDeque::new();
        let mut lag_buffer: VecDeque<f64> = VecDeque::new();
        let mut sum = 0.0_f64;

        for _ in 0..3_000 {
            let x = rng.f64() * 100.0;
            dpo.update(x);

            // Mirrors `Lag::update` exactly: push, then pop the oldest once
            // the buffer reaches `lag_window`.
            lag_buffer.push_back(x);
            let shifted = if lag_buffer.len() >= lag_window {
                lag_buffer.pop_front()
            } else {
                None
            };

            if window.len() == period {
                sum -= window.pop_front().expect("full");
            }
            window.push_back(x);
            sum += x;

            let naive = if window.len() == period {
                shifted.map(|s| s - sum / period as f64)
            } else {
                None
            };

            if let (Some(got), Some(expected)) = (dpo.last(), naive) {
                assert!(
                    (got - expected).abs() < 1e-9,
                    "got = {got}, expected = {expected}"
                );
            } else {
                assert_eq!(dpo.last(), naive);
            }
        }
    }
}

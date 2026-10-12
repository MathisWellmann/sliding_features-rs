//! TSI - True Strength Index

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::Ema;
use crate::{
    View,
    pure_functions::Echo,
};

/// TSI - True Strength Index (William Blau, 1991, "Momentum, Direction and
/// Divergence").
///
/// With momentum `m = x - x_prev`:
/// `TSI = 100 * EMA(EMA(m, r), s) / EMA(EMA(|m|, r), s)`, a double-smoothed
/// momentum normalized by its double-smoothed absolute value. Classic
/// defaults: `r = 25` (long smoothing), `s = 13` (short smoothing).
///
/// Deviation: the original Blau definition has no signal line; this
/// implementation additionally tracks an EMA of the TSI line over
/// `signal_period` (a common extension used by most charting platforms),
/// exposed through [`Tsi::signal`].
///
/// If the double-smoothed absolute momentum is `0` (e.g. constant input),
/// `TSI` is defined as `0` rather than `NaN`.
///
/// `last()` is `None` until both double EMA cascades are ready.
///
/// Output range: `[-100, 100]`.
///
/// Complexity: O(1) per update (four chained EMAs plus the signal EMA), O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Tsi<T, V> {
    view: V,
    /// The long (first-stage) smoothing period.
    #[getset(get_copy = "pub")]
    long_period: NonZeroUsize,
    /// The short (second-stage) smoothing period.
    #[getset(get_copy = "pub")]
    short_period: NonZeroUsize,
    /// The signal line smoothing period.
    #[getset(get_copy = "pub")]
    signal_period: NonZeroUsize,
    prev_val: Option<T>,
    ema_m_long: Ema<T, Echo<T>>,
    ema_m_short: Ema<T, Echo<T>>,
    ema_am_long: Ema<T, Echo<T>>,
    ema_am_short: Ema<T, Echo<T>>,
    ema_signal: Ema<T, Echo<T>>,
    out: Option<T>,
}

impl<T, V> Tsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new TSI with a chained `View` and custom long, short and
    /// signal periods.
    pub fn with_periods(
        view: V,
        long_period: NonZeroUsize,
        short_period: NonZeroUsize,
        signal_period: NonZeroUsize,
    ) -> Self {
        Self {
            view,
            long_period,
            short_period,
            signal_period,
            prev_val: None,
            ema_m_long: Ema::new(Echo::new(), long_period),
            ema_m_short: Ema::new(Echo::new(), short_period),
            ema_am_long: Ema::new(Echo::new(), long_period),
            ema_am_short: Ema::new(Echo::new(), short_period),
            ema_signal: Ema::new(Echo::new(), signal_period),
            out: None,
        }
    }

    /// Create a new TSI with a chained `View` and the classic default
    /// periods: `r = 25`, `s = 13`, and a 7-period signal line.
    pub fn new(view: V) -> Self {
        Self::with_periods(
            view,
            NonZeroUsize::new(25).expect("25 is non-zero"),
            NonZeroUsize::new(13).expect("13 is non-zero"),
            NonZeroUsize::new(7).expect("7 is non-zero"),
        )
    }

    /// The signal line: an EMA of the TSI line over `signal_period`.
    /// `None` until the signal EMA is ready.
    #[inline]
    pub fn signal(&self) -> Option<T> {
        self.ema_signal.last()
    }
}

impl<T, V> View<T> for Tsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let Some(prev) = self.prev_val else {
            self.prev_val = Some(val);
            return;
        };
        self.prev_val = Some(val);
        let m = val - prev;
        debug_assert!(m.is_finite(), "`m` must be finite");

        self.ema_m_long.update(m);
        self.ema_am_long.update(m.abs());
        let (Some(m_long), Some(am_long)) = (self.ema_m_long.last(), self.ema_am_long.last())
        else {
            return;
        };
        self.ema_m_short.update(m_long);
        self.ema_am_short.update(am_long);
        let (Some(m_short), Some(am_short)) = (self.ema_m_short.last(), self.ema_am_short.last())
        else {
            return;
        };

        let tsi = if am_short == T::zero() {
            T::zero()
        } else {
            T::from(100.0).expect("can convert") * m_short / am_short
        };
        debug_assert!(tsi.is_finite(), "`tsi` must be finite");
        self.ema_signal.update(tsi);
        self.out = Some(tsi);
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
    fn tsi_plot() {
        let mut tsi = Tsi::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            tsi.update(*v);
            if let Some(val) = tsi.last() {
                out.push(val);
            }
        }
        let filename = "img/tsi.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn tsi_warms_up() {
        let mut tsi = Tsi::with_periods(
            Echo::new(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        let mut seen = false;
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            tsi.update(v);
            if tsi.last().is_some() {
                seen = true;
            }
        }
        assert!(seen);
        assert!(tsi.signal().is_some());
    }

    #[test]
    fn tsi_range() {
        let mut tsi = Tsi::new(Echo::new());
        for v in &TEST_DATA {
            tsi.update(*v);
            if let Some(val) = tsi.last() {
                assert!((-100.0..=100.0).contains(&val), "val = {val}");
            }
        }
    }

    #[test]
    fn tsi_constant_input_is_zero() {
        let mut tsi = Tsi::with_periods(
            Echo::new(),
            NonZeroUsize::new(5).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );
        for _ in 0..200 {
            tsi.update(42.0);
        }
        // Momentum is 0 everywhere, so the 0/0 case is hit and defined as 0.
        assert_eq!(tsi.last(), Some(0.0));
    }

    /// Naive reference, mirroring the gating of the real implementation.
    #[test]
    fn tsi_vs_naive_random() {
        struct NaiveEma {
            period: usize,
            val: Option<f64>,
            n: usize,
        }
        impl NaiveEma {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    val: None,
                    n: 0,
                }
            }
            fn update(&mut self, x: f64) -> Option<f64> {
                let alpha = 2.0 / (self.period as f64 + 1.0);
                let out = match self.val {
                    None => x,
                    Some(prev) => alpha * x + (1.0 - alpha) * prev,
                };
                self.val = Some(out);
                self.n += 1;
                (self.n >= self.period).then_some(out)
            }
        }

        let rng = romu::Rng::from_seed_with_64bit(23);
        let (long, short) = (7, 4);
        let mut tsi = Tsi::with_periods(
            Echo::new(),
            NonZeroUsize::new(long).unwrap(),
            NonZeroUsize::new(short).unwrap(),
            NonZeroUsize::new(3).unwrap(),
        );
        let mut m_long = NaiveEma::new(long);
        let mut m_short = NaiveEma::new(short);
        let mut am_long = NaiveEma::new(long);
        let mut am_short = NaiveEma::new(short);
        let mut prev = None;
        for _ in 0..2_000 {
            let x = rng.f64() * 100.0;
            tsi.update(x);
            let Some(p) = prev else {
                prev = Some(x);
                continue;
            };
            prev = Some(x);
            let m = x - p;
            let naive = m_long
                .update(m)
                .and_then(|ml| m_short.update(ml))
                .zip(am_long.update(m.abs()).and_then(|al| am_short.update(al)))
                .map(|(ms, ams)| if ams == 0.0 { 0.0 } else { 100.0 * ms / ams });
            if let (Some(got), Some(expected)) = (tsi.last(), naive) {
                assert!((got - expected).abs() < 1e-9);
            }
        }
    }
}

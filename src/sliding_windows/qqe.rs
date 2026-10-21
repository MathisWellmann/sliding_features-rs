// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! QQE - Quantitative Qualitative Estimation

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Ema,
    Rsi,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// QQE - Quantitative Qualitative Estimation (Yuri Reshetov).
///
/// Smooths the RSI with an EMA (`SF`, default 5 -> "RsiMa"), then builds a
/// volatility band around it from an "ATR of RSI": the Wilder-smoothed
/// (recurrence `avg += (x - avg) / n`, over `2 * rsi_period - 1` bars, the
/// classic Wilder period) average of `|RsiMa - RsiMa_prev|`, itself
/// Wilder-smoothed a second time and scaled by `factor` (default `4.236`).
/// A trailing stop-like level then follows `RsiMa` from below while in an
/// uptrend and from above while in a downtrend, flipping when `RsiMa`
/// crosses the opposite band (see [`Qqe::trailing_level`]).
///
/// Deviation: the original indicator's trend/band update compares `RsiMa`
/// against a one-bar-shifted view of the *band* series itself (a "crossover
/// of a crossover"), which is fragile to transcribe outside of Pine Script.
/// This implementation instead detects the cross using the previous bar's
/// `RsiMa` against the previous bar's band value directly
/// (`prev_rsi_ma <= prev_short_band && rsi_ma > prev_short_band`, and the
/// symmetric case for the long band), which matches the original closely
/// while being well-defined bar-by-bar.
///
/// `last()` is the smoothed RSI (`RsiMa`), `None` until the RSI and its EMA
/// are ready. [`Qqe::trailing_level`] additionally needs both Wilder-smoothed
/// ATR stages to warm up (`2 * (2 * rsi_period - 1)` further values).
///
/// Output range: `RsiMa` is in `[0, 100]`; `trailing_level` oscillates
/// around it and is unbounded in principle but stays close to `[0, 100]`
/// in practice.
///
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct Qqe<T: Float, V> {
    view: V,
    /// The RSI period.
    #[getset(get_copy = "pub")]
    rsi_period: NonZeroUsize,
    /// The EMA period used to smooth the RSI into `RsiMa`.
    #[getset(get_copy = "pub")]
    smoothing_period: NonZeroUsize,
    /// The Wilder smoothing period for both ATR-of-RSI stages
    /// (`2 * rsi_period - 1`).
    #[getset(get_copy = "pub")]
    wilder_period: NonZeroUsize,
    /// The multiplier applied to the twice Wilder-smoothed ATR of RSI.
    #[getset(get_copy = "pub")]
    factor: T,
    rsi: Rsi<T, Echo<T>>,
    rsi_ema: Ema<T, Echo<T>>,
    prev_rsi_ma: Option<T>,
    atr_stage1: T,
    n1: usize,
    atr_stage2: T,
    n2: usize,
    long_band: T,
    short_band: T,
    /// `1` for an uptrend (follow the long band), `-1` for a downtrend.
    trend: i8,
    bands_initialized: bool,
    out: Option<T>,
    trailing_level: Option<T>,
}

impl<T, V> Qqe<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new QQE with a chained `View`, an RSI period, a smoothing
    /// (`SF`) period and the ATR multiplier `factor`. The Wilder period for
    /// the ATR-of-RSI stages is derived as `2 * rsi_period - 1`.
    pub fn with_params(
        view: V,
        rsi_period: NonZeroUsize,
        smoothing_period: NonZeroUsize,
        factor: T,
    ) -> Self {
        let wilder_period = NonZeroUsize::new(2 * rsi_period.get() - 1)
            .expect("2 * rsi_period - 1 is non-zero for rsi_period >= 1");
        Self {
            view,
            rsi_period,
            smoothing_period,
            wilder_period,
            factor,
            rsi: Rsi::new(Echo::new(), rsi_period),
            rsi_ema: Ema::new(Echo::new(), smoothing_period),
            prev_rsi_ma: None,
            atr_stage1: T::zero(),
            n1: 0,
            atr_stage2: T::zero(),
            n2: 0,
            long_band: T::zero(),
            short_band: T::zero(),
            trend: 1,
            bands_initialized: false,
            out: None,
            trailing_level: None,
        }
    }

    /// Create a new QQE with a chained `View` and the classic default
    /// parameters: a 14-period RSI, a 5-period smoothing EMA and a `4.236`
    /// ATR factor.
    pub fn new(view: V) -> Self {
        Self::with_params(
            view,
            NonZeroUsize::new(14).expect("14 is non-zero"),
            NonZeroUsize::new(5).expect("5 is non-zero"),
            T::from(4.236).expect("can convert"),
        )
    }

    /// The trailing stop-like level that follows the smoothed RSI.
    /// `None` until both Wilder-smoothed ATR-of-RSI stages are ready.
    #[inline]
    pub fn trailing_level(&self) -> Option<T> {
        self.trailing_level
    }

    /// One Wilder smoothing step (`avg += (x - avg) / n`), seeded with the
    /// first observation. Returns `None` until `period` values were fed.
    fn wilder_step(avg: &mut T, n: &mut usize, period: usize, x: T) -> Option<T> {
        *n += 1;
        if *n == 1 {
            *avg = x;
        } else {
            *avg = *avg + (x - *avg) / T::from(period).expect("can convert");
        }
        (*n >= period).then_some(*avg)
    }
}

impl<T, V> View<T> for Qqe<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.rsi.update(val);
        let Some(rsi) = self.rsi.last() else { return };
        self.rsi_ema.update(rsi);
        let Some(rsi_ma) = self.rsi_ema.last() else {
            return;
        };
        debug_assert!(rsi_ma.is_finite(), "`rsi_ma` must be finite");
        self.out = Some(rsi_ma);

        let Some(prev_rsi_ma) = self.prev_rsi_ma else {
            self.prev_rsi_ma = Some(rsi_ma);
            return;
        };
        self.prev_rsi_ma = Some(rsi_ma);

        let atr = (rsi_ma - prev_rsi_ma).abs();
        let wilder_period = self.wilder_period.get();
        let Some(stage1) =
            Self::wilder_step(&mut self.atr_stage1, &mut self.n1, wilder_period, atr)
        else {
            return;
        };
        let Some(stage2) =
            Self::wilder_step(&mut self.atr_stage2, &mut self.n2, wilder_period, stage1)
        else {
            return;
        };
        let delta = stage2 * self.factor;
        debug_assert!(delta.is_finite(), "`delta` must be finite");

        let new_long_band = rsi_ma - delta;
        let new_short_band = rsi_ma + delta;
        if !self.bands_initialized {
            self.long_band = new_long_band;
            self.short_band = new_short_band;
            self.bands_initialized = true;
        } else {
            let (old_long, old_short) = (self.long_band, self.short_band);
            self.long_band = if prev_rsi_ma > old_long && rsi_ma > old_long {
                old_long.max(new_long_band)
            } else {
                new_long_band
            };
            self.short_band = if prev_rsi_ma < old_short && rsi_ma < old_short {
                old_short.min(new_short_band)
            } else {
                new_short_band
            };

            if prev_rsi_ma <= old_short && rsi_ma > old_short {
                self.trend = 1;
            } else if prev_rsi_ma >= old_long && rsi_ma < old_long {
                self.trend = -1;
            }
        }

        let trailing = if self.trend == 1 {
            self.long_band
        } else {
            self.short_band
        };
        debug_assert!(trailing.is_finite(), "`trailing` must be finite");
        self.trailing_level = Some(trailing);
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
    fn qqe_plot() {
        let mut qqe = Qqe::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            qqe.update(*v);
            if let Some(val) = qqe.last() {
                out.push(val);
            }
        }
        let filename = "img/qqe.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn qqe_warms_up() {
        let mut qqe = Qqe::with_params(
            Echo::new(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            4.236,
        );
        let mut seen_rsi_ma = false;
        let mut seen_trailing = false;
        for v in (0..100).map(|i| (i as f64 * 0.21).sin() * 10.0 + 50.0) {
            qqe.update(v);
            if qqe.last().is_some() {
                seen_rsi_ma = true;
            }
            if qqe.trailing_level().is_some() {
                seen_trailing = true;
            }
        }
        assert!(seen_rsi_ma);
        assert!(seen_trailing);
    }

    #[test]
    fn qqe_rsi_ma_range() {
        let mut qqe = Qqe::new(Echo::new());
        for v in &TEST_DATA {
            qqe.update(*v);
            if let Some(val) = qqe.last() {
                assert!((0.0..=100.0).contains(&val), "val = {val}");
            }
        }
    }

    #[test]
    fn qqe_trailing_level_finite() {
        let mut qqe = Qqe::new(Echo::new());
        for v in &TEST_DATA {
            qqe.update(*v);
            if let Some(level) = qqe.trailing_level() {
                assert!(level.is_finite());
            }
        }
    }

    #[test]
    fn qqe_constant_input_rsi_ma_is_hundred() {
        // Constant, strictly increasing input is a classic RSI edge case:
        // with no losses at all, RSI (and so RsiMa) saturates at 100.
        let mut qqe = Qqe::new(Echo::new());
        for i in 0..200 {
            qqe.update(i as f64);
        }
        assert_eq!(qqe.last(), Some(100.0));
        // No momentum change in RsiMa once saturated, so the band collapses
        // onto RsiMa itself.
        assert!((qqe.trailing_level().unwrap() - 100.0).abs() < 1e-6);
    }

    /// Naive reference, independently reimplementing the sliding-window
    /// RSI, EMA, Wilder smoothing and band/trend logic, to validate the
    /// wiring of the real implementation.
    #[test]
    fn qqe_vs_naive_random() {
        use std::collections::VecDeque;

        struct NaiveRsi {
            period: usize,
            avg_gain: f64,
            avg_loss: f64,
            old_ref: f64,
            last_val: f64,
            q: VecDeque<f64>,
        }
        impl NaiveRsi {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    avg_gain: 0.0,
                    avg_loss: 0.0,
                    old_ref: 0.0,
                    last_val: 0.0,
                    q: VecDeque::new(),
                }
            }
            fn update(&mut self, val: f64) -> Option<f64> {
                if self.q.is_empty() {
                    self.old_ref = val;
                    self.last_val = val;
                }
                let n = self.period as f64;
                if self.q.len() >= self.period {
                    let old_val = *self.q.front().expect("non-empty");
                    let change = old_val - self.old_ref;
                    self.old_ref = old_val;
                    self.q.pop_front();
                    if change > 0.0 {
                        self.avg_gain -= change / n;
                    } else {
                        self.avg_loss -= change.abs() / n;
                    }
                }
                self.q.push_back(val);
                let change = val - self.last_val;
                self.last_val = val;
                if change > 0.0 {
                    self.avg_gain += change / n;
                } else {
                    self.avg_loss += change.abs() / n;
                }
                if self.q.len() < self.period {
                    return None;
                }
                Some(if self.avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
                })
            }
        }
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
        fn wilder(avg: &mut f64, n: &mut usize, period: usize, x: f64) -> Option<f64> {
            *n += 1;
            if *n == 1 {
                *avg = x;
            } else {
                *avg += (x - *avg) / period as f64;
            }
            (*n >= period).then_some(*avg)
        }

        let rng = romu::Rng::from_seed_with_64bit(7);
        let (rsi_period, smoothing_period, factor) = (14_usize, 5_usize, 4.236);
        let wilder_period = 2 * rsi_period - 1;

        let mut qqe = Qqe::new(Echo::new());
        let mut naive_rsi = NaiveRsi::new(rsi_period);
        let mut naive_ema = NaiveEma::new(smoothing_period);
        let (mut a1, mut n1) = (0.0, 0_usize);
        let (mut a2, mut n2) = (0.0, 0_usize);
        let mut prev_rsi_ma: Option<f64> = None;
        let (mut long_band, mut short_band) = (0.0, 0.0);
        let mut bands_initialized = false;
        let mut trend = 1_i8;

        for _ in 0..5_000 {
            let x = rng.f64() * 100.0;
            qqe.update(x);

            let Some(rsi) = naive_rsi.update(x) else {
                continue;
            };
            let Some(rsi_ma) = naive_ema.update(rsi) else {
                continue;
            };
            assert!((qqe.last().expect("ready") - rsi_ma).abs() < 1e-9);

            let Some(prev) = prev_rsi_ma else {
                prev_rsi_ma = Some(rsi_ma);
                continue;
            };
            prev_rsi_ma = Some(rsi_ma);

            let atr = (rsi_ma - prev).abs();
            let Some(stage1) = wilder(&mut a1, &mut n1, wilder_period, atr) else {
                continue;
            };
            let Some(stage2) = wilder(&mut a2, &mut n2, wilder_period, stage1) else {
                continue;
            };
            let delta = stage2 * factor;
            let (new_long, new_short) = (rsi_ma - delta, rsi_ma + delta);
            if !bands_initialized {
                long_band = new_long;
                short_band = new_short;
                bands_initialized = true;
            } else {
                let (old_long, old_short) = (long_band, short_band);
                long_band = if prev > old_long && rsi_ma > old_long {
                    old_long.max(new_long)
                } else {
                    new_long
                };
                short_band = if prev < old_short && rsi_ma < old_short {
                    old_short.min(new_short)
                } else {
                    new_short
                };
                if prev <= old_short && rsi_ma > old_short {
                    trend = 1;
                } else if prev >= old_long && rsi_ma < old_long {
                    trend = -1;
                }
            }
            let expected_trailing = if trend == 1 { long_band } else { short_band };
            assert!((qqe.trailing_level().expect("ready") - expected_trailing).abs() < 1e-9);
        }
    }
}

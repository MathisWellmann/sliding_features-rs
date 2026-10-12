//! KST - Pring's "Know Sure Thing" momentum oscillator.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    Roc,
    Sma,
};
use crate::{
    View,
    pure_functions::Echo,
};

/// KST - Pring's "Know Sure Thing" (Martin J. Pring): a weighted sum of
/// four smoothed rates of change, plus a signal line:
///
/// `KST = 1 * SMA(ROC(roc1), sma1) + 2 * SMA(ROC(roc2), sma2)
///      + 3 * SMA(ROC(roc3), sma3) + 4 * SMA(ROC(roc4), sma4)`
///
/// `signal = SMA(KST, signal_period)`.
///
/// Classic defaults: `roc` periods `10, 15, 20, 30`, matching `sma`
/// smoothing periods `10, 10, 10, 15`, and `signal_period = 9`. The
/// integer weights `1, 2, 3, 4` (increasing with the `roc`/`sma` period)
/// are a fixed part of the formula, not configurable.
///
/// `last()` (the `KST` line) is `None` until the slowest of the four
/// `SMA(ROC(..), ..)` legs is ready, i.e. the largest of the four `sma`
/// periods (`15` by default): the crate's own [`Roc`] is not gated on its
/// own `window_len` (see its docs: it starts comparing against the
/// first-ever observed value), so only the `Sma` stage that follows it
/// gates each leg. [`Kst::signal`] additionally needs `signal_period - 1`
/// further `KST` values.
///
/// Output range: unbounded, oscillates around `0`.
///
/// Complexity: O(1) per update, O(`roc4 + sma4 + signal_period`) memory
/// (the slowest leg dominates).
#[derive(Debug, Clone, CopyGetters)]
pub struct Kst<T, V> {
    view: V,
    /// The fastest rate-of-change period (weight `1`).
    #[getset(get_copy = "pub")]
    roc1_period: NonZeroUsize,
    /// The second rate-of-change period (weight `2`).
    #[getset(get_copy = "pub")]
    roc2_period: NonZeroUsize,
    /// The third rate-of-change period (weight `3`).
    #[getset(get_copy = "pub")]
    roc3_period: NonZeroUsize,
    /// The slowest rate-of-change period (weight `4`).
    #[getset(get_copy = "pub")]
    roc4_period: NonZeroUsize,
    /// The smoothing period applied to `roc1`.
    #[getset(get_copy = "pub")]
    sma1_period: NonZeroUsize,
    /// The smoothing period applied to `roc2`.
    #[getset(get_copy = "pub")]
    sma2_period: NonZeroUsize,
    /// The smoothing period applied to `roc3`.
    #[getset(get_copy = "pub")]
    sma3_period: NonZeroUsize,
    /// The smoothing period applied to `roc4`.
    #[getset(get_copy = "pub")]
    sma4_period: NonZeroUsize,
    /// The smoothing period of the signal line.
    #[getset(get_copy = "pub")]
    signal_period: NonZeroUsize,
    roc1: Roc<T, Echo<T>>,
    roc2: Roc<T, Echo<T>>,
    roc3: Roc<T, Echo<T>>,
    roc4: Roc<T, Echo<T>>,
    sma1: Sma<T, Echo<T>>,
    sma2: Sma<T, Echo<T>>,
    sma3: Sma<T, Echo<T>>,
    sma4: Sma<T, Echo<T>>,
    signal_sma: Sma<T, Echo<T>>,
    out: Option<T>,
    signal: Option<T>,
}

impl<T, V> Kst<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new KST with a chained `View` and explicit periods for the
    /// four rate-of-change/smoothing pairs and the signal line.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors KST's nine periods one for one; grouping them \
                  into tuples or arrays loses the direct correspondence \
                  with the classic formula's named parameters"
    )]
    pub fn with_params(
        view: V,
        roc1_period: NonZeroUsize,
        roc2_period: NonZeroUsize,
        roc3_period: NonZeroUsize,
        roc4_period: NonZeroUsize,
        sma1_period: NonZeroUsize,
        sma2_period: NonZeroUsize,
        sma3_period: NonZeroUsize,
        sma4_period: NonZeroUsize,
        signal_period: NonZeroUsize,
    ) -> Self {
        Self {
            view,
            roc1_period,
            roc2_period,
            roc3_period,
            roc4_period,
            sma1_period,
            sma2_period,
            sma3_period,
            sma4_period,
            signal_period,
            roc1: Roc::new(Echo::new(), roc1_period),
            roc2: Roc::new(Echo::new(), roc2_period),
            roc3: Roc::new(Echo::new(), roc3_period),
            roc4: Roc::new(Echo::new(), roc4_period),
            sma1: Sma::new(Echo::new(), sma1_period),
            sma2: Sma::new(Echo::new(), sma2_period),
            sma3: Sma::new(Echo::new(), sma3_period),
            sma4: Sma::new(Echo::new(), sma4_period),
            signal_sma: Sma::new(Echo::new(), signal_period),
            out: None,
            signal: None,
        }
    }

    /// Create a new KST with a chained `View` and the classic default
    /// periods: `ROC(10, 15, 20, 30)` smoothed by `SMA(10, 10, 10, 15)`,
    /// with a `9`-period signal line.
    pub fn new(view: V) -> Self {
        let p = |n| NonZeroUsize::new(n).expect("n is non-zero");
        Self::with_params(
            view,
            p(10),
            p(15),
            p(20),
            p(30),
            p(10),
            p(10),
            p(10),
            p(15),
            p(9),
        )
    }

    /// The signal line, the `signal_period`-period `SMA` of the `KST`
    /// line. `None` until it has warmed up.
    #[inline]
    pub fn signal(&self) -> Option<T> {
        self.signal
    }
}

impl<T, V> View<T> for Kst<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value from `View` must be finite");

        self.roc1.update(val);
        self.roc2.update(val);
        self.roc3.update(val);
        self.roc4.update(val);
        if let Some(r) = self.roc1.last() {
            self.sma1.update(r);
        }
        if let Some(r) = self.roc2.last() {
            self.sma2.update(r);
        }
        if let Some(r) = self.roc3.last() {
            self.sma3.update(r);
        }
        if let Some(r) = self.roc4.last() {
            self.sma4.update(r);
        }

        let Some((((s1, s2), s3), s4)) = self
            .sma1
            .last()
            .zip(self.sma2.last())
            .zip(self.sma3.last())
            .zip(self.sma4.last())
        else {
            return;
        };
        let (w1, w2, w3, w4) = (
            T::from(1.0).expect("can convert"),
            T::from(2.0).expect("can convert"),
            T::from(3.0).expect("can convert"),
            T::from(4.0).expect("can convert"),
        );
        let kst = s1 * w1 + s2 * w2 + s3 * w3 + s4 * w4;
        debug_assert!(kst.is_finite(), "`kst` must be finite");
        self.out = Some(kst);

        self.signal_sma.update(kst);
        self.signal = self.signal_sma.last();
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
    fn kst_plot() {
        let mut kst = Kst::new(Echo::new());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            kst.update(*v);
            if let Some(val) = kst.last() {
                out.push(val);
            }
        }
        let filename = "img/kst.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn kst_warms_up() {
        let p = |n| NonZeroUsize::new(n).unwrap();
        let mut kst = Kst::with_params(
            Echo::new(),
            p(3),
            p(4),
            p(5),
            p(6),
            p(2),
            p(2),
            p(2),
            p(3),
            p(2),
        );
        // Bottleneck is the slowest `sma` leg (3).
        let mut seen_kst = false;
        let mut seen_signal = false;
        for (i, v) in TEST_DATA.iter().enumerate() {
            kst.update(*v);
            if i + 1 < 3 {
                assert_eq!(kst.last(), None);
            }
            if kst.last().is_some() {
                seen_kst = true;
            }
            if kst.signal().is_some() {
                seen_signal = true;
            }
        }
        assert!(seen_kst);
        assert!(seen_signal);
    }

    #[test]
    fn kst_constant_input_is_zero() {
        let mut kst = Kst::new(Echo::new());
        for _ in 0..60 {
            kst.update(42.0);
        }
        assert_eq!(kst.last(), Some(0.0));
        assert_eq!(kst.signal(), Some(0.0));
    }

    #[test]
    fn kst_finite() {
        let mut kst = Kst::new(Echo::new());
        for v in &TEST_DATA {
            kst.update(*v);
            if let Some(val) = kst.last() {
                assert!(val.is_finite(), "val = {val}");
            }
            if let Some(val) = kst.signal() {
                assert!(val.is_finite(), "val = {val}");
            }
        }
    }

    /// Naive reference, independently reimplementing the crate's own `Roc`
    /// (including its not-gated-on-`window_len` warm-up) and the plain
    /// average underlying `Sma`, to validate the wiring of the real
    /// implementation.
    #[test]
    fn kst_vs_naive_random() {
        use std::collections::VecDeque;

        /// Independent reimplementation of the crate's own `Roc`, matching
        /// its exact (not-gated) warm-up bookkeeping.
        struct NaiveRoc {
            period: usize,
            oldest: Option<f64>,
            q: VecDeque<f64>,
        }
        impl NaiveRoc {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    oldest: None,
                    q: VecDeque::new(),
                }
            }
            fn update(&mut self, val: f64) -> Option<f64> {
                if self.q.is_empty() {
                    self.oldest = Some(val);
                }
                if self.q.len() >= self.period {
                    let old = *self.q.front().expect("non-empty");
                    self.oldest = Some(old);
                    self.q.pop_front();
                }
                self.q.push_back(val);
                let oldest = self.oldest?;
                if oldest == 0.0 {
                    return None;
                }
                Some((val - oldest) / oldest * 100.0)
            }
        }
        struct NaiveSma {
            period: usize,
            q: VecDeque<f64>,
            sum: f64,
        }
        impl NaiveSma {
            fn new(period: usize) -> Self {
                Self {
                    period,
                    q: VecDeque::new(),
                    sum: 0.0,
                }
            }
            fn update(&mut self, val: f64) -> Option<f64> {
                if self.q.len() >= self.period {
                    self.sum -= self.q.pop_front().expect("full");
                }
                self.q.push_back(val);
                self.sum += val;
                (self.q.len() >= self.period).then(|| self.sum / self.period as f64)
            }
        }

        let rng = romu::Rng::from_seed_with_64bit(53);
        let roc_periods = [10_usize, 15, 20, 30];
        let sma_periods = [10_usize, 10, 10, 15];
        let signal_period = 9_usize;
        let mut kst = Kst::new(Echo::new());
        let mut naive_roc: Vec<NaiveRoc> = roc_periods.iter().map(|p| NaiveRoc::new(*p)).collect();
        let mut naive_sma: Vec<NaiveSma> = sma_periods.iter().map(|p| NaiveSma::new(*p)).collect();
        let mut naive_signal = NaiveSma::new(signal_period);

        for _ in 0..3_000 {
            let x = rng.f64() * 100.0 + 1.0;
            kst.update(x);

            let mut smoothed = [None; 4];
            for i in 0..4 {
                if let Some(r) = naive_roc[i].update(x) {
                    smoothed[i] = naive_sma[i].update(r);
                }
            }
            let Some(expected_kst) = smoothed[0]
                .zip(smoothed[1])
                .zip(smoothed[2])
                .zip(smoothed[3])
                .map(|(((s1, s2), s3), s4)| s1 * 1.0 + s2 * 2.0 + s3 * 3.0 + s4 * 4.0)
            else {
                assert_eq!(kst.last(), None);
                continue;
            };
            let got_kst = kst.last().expect("ready");
            assert!(
                (got_kst - expected_kst).abs() < 1e-9,
                "got = {got_kst}, expected = {expected_kst}"
            );

            let expected_signal = naive_signal.update(expected_kst);
            assert_eq!(kst.signal().is_some(), expected_signal.is_some());
            if let (Some(got), Some(expected)) = (kst.signal(), expected_signal) {
                assert!(
                    (got - expected).abs() < 1e-9,
                    "got = {got}, expected = {expected}"
                );
            }
        }
    }
}

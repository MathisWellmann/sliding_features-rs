use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::{
    HighPass,
    HighPassPoles,
    SuperSmoother,
};
use crate::View;

/// Length of the SuperSmoother after the high-pass filter, fixed to 10 in the book.
const SMOOTHER_LEN: usize = 10;

/// John Ehlers Even Better Sinewave (EBSW) indicator.
///
/// Reference: John Ehlers, "Cycle Analytics for Traders" (2013), chapter 12,
/// "Even Better Sinewave Indicator":
///
/// ```text
/// HP   = one-pole high-pass of the input with cutoff `duration`
/// Filt = SuperSmoother(HP, 10)
/// Wave = (Filt + Filt[1] + Filt[2]) / 3
/// Pwr  = (Filt^2 + Filt[1]^2 + Filt[2]^2) / 3
/// EBSW = Wave / sqrt(Pwr)
/// ```
///
/// The normalization acts as an automatic gain control, so the output swings
/// between -1 and +1 in cycling markets and saturates near +1 (-1) in up (down) trends.
/// `duration` is the longest period kept by the high-pass filter, 40 in the book.
///
/// Deviations: the high-pass filter starts at rest (see [`HighPass`]) and the
/// SuperSmoother only starts once the high-pass filter is warmed up.
/// If the power is zero (e.g. constant input) the output is 0 instead of a division by zero.
/// The output is clamped to `[-1, 1]` to absorb rounding errors; mathematically
/// `|Wave| <= sqrt(Pwr)` holds by the Cauchy-Schwarz inequality.
///
/// Output range: `[-1, 1]`.
/// Complexity: O(1) per update, O(1) memory.
#[derive(Debug, Clone, CopyGetters)]
pub struct EvenBetterSinewave<T, V> {
    /// High-pass filter followed by the SuperSmoother, with the inner view chained in.
    filter: SuperSmoother<T, HighPass<T, V>>,
    /// The cutoff period of the high-pass filter.
    #[getset(get_copy = "pub")]
    duration: NonZeroUsize,
    /// The last three filtered values, newest first.
    filt: [T; 3],
    /// Number of filtered values received, saturating at 3.
    n: usize,
    wave: T,
}

impl<T, V> EvenBetterSinewave<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with the high-pass cutoff period `duration` (40 in the book).
    ///
    /// # Panics
    /// If `duration < 5`, as the one-pole high-pass filter is unstable below that.
    pub fn new(view: V, duration: NonZeroUsize) -> Self {
        assert!(duration.get() >= 5, "duration must be >= 5");
        let filter = SuperSmoother::new(
            HighPass::new(view, duration, HighPassPoles::One),
            NonZeroUsize::new(SMOOTHER_LEN).expect("is non zero"),
        );
        Self {
            filter,
            duration,
            filt: [T::zero(); 3],
            n: 0,
            wave: T::zero(),
        }
    }
}

impl<T, V> View<T> for EvenBetterSinewave<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.filter.update(val);
        let Some(filt) = self.filter.last() else {
            return;
        };
        debug_assert!(filt.is_finite(), "value must be finite");

        self.filt = [filt, self.filt[0], self.filt[1]];
        self.n = (self.n + 1).min(3);
        if self.n < 3 {
            return;
        }
        let three = T::from(3.0).expect("can convert");
        let wave = self.filt.iter().fold(T::zero(), |s, &f| s + f) / three;
        let pwr = self.filt.iter().fold(T::zero(), |s, &f| s + f * f) / three;
        debug_assert!(pwr >= T::zero(), "power must be >= 0");
        self.wave = if pwr > T::zero() {
            (wave / pwr.sqrt()).max(-T::one()).min(T::one())
        } else {
            T::zero()
        };
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.n < 3 {
            return None;
        }
        debug_assert!(self.wave.is_finite(), "value must be finite");
        debug_assert!(self.wave.abs() <= T::one(), "value must be in [-1, 1]");
        Some(self.wave)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn ebsw(duration: usize) -> EvenBetterSinewave<f64, Echo<f64>> {
        EvenBetterSinewave::new(Echo::new(), NonZeroUsize::new(duration).unwrap())
    }

    #[test]
    fn even_better_sinewave_plot() {
        let mut e = ebsw(40);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            e.update(*v);
            if let Some(val) = e.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/even_better_sinewave.png").unwrap();
    }

    #[test]
    fn even_better_sinewave_warmup() {
        let mut e = ebsw(40);
        assert_eq!(e.duration().get(), 40);
        // High-pass needs 40 values, the SuperSmoother 10 and the wave 3.
        let warmup = 40 + 10 + 3 - 2;
        for i in 0..warmup - 1 {
            e.update(i as f64);
            assert_eq!(e.last(), None);
        }
        e.update(warmup as f64);
        assert!(e.last().is_some());
    }

    #[test]
    fn even_better_sinewave_sine_swings_full_range() {
        for period in [15.0, 20.0, 30.0] {
            let mut e = ebsw(40);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for i in 0..1_000 {
                e.update(100.0 + 5.0 * (2.0 * PI * i as f64 / period).sin());
                let w = e.last().unwrap_or(0.0);
                assert!((-1.0..=1.0).contains(&w), "wave {w}");
                if i > 500 {
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            assert!(hi > 0.95, "period {period}: max {hi}");
            assert!(lo < -0.95, "period {period}: min {lo}");
        }
    }

    #[test]
    fn even_better_sinewave_trend_saturates() {
        let mut up = ebsw(40);
        let mut down = ebsw(40);
        for i in 0..300 {
            up.update(100.0 + i as f64);
            down.update(100.0 - 0.1 * i as f64);
        }
        assert!(up.last().unwrap() > 0.99);
        assert!(down.last().unwrap() < -0.99);
    }

    #[test]
    fn even_better_sinewave_constant_input_is_zero() {
        let mut e = ebsw(40);
        for _ in 0..200 {
            e.update(42.0);
        }
        assert_eq!(e.last(), Some(0.0));
    }

    #[test]
    fn even_better_sinewave_random_in_range() {
        let rng = romu::Rng::from_seed_with_64bit(3);
        let mut e = EvenBetterSinewave::<f32, _>::new(Echo::new(), NonZeroUsize::new(5).unwrap());
        for _ in 0..10_000 {
            e.update(rng.f32() * 100.0);
            if let Some(w) = e.last() {
                assert!((-1.0..=1.0).contains(&w), "wave {w}");
            }
        }
    }

    /// Direct transcription of the book code, with the same start-up as the crate filters.
    #[test]
    fn even_better_sinewave_matches_book_transcription() {
        let d = 40.0;
        let alpha1 = (1.0 - (2.0 * PI / d).sin()) / (2.0 * PI / d).cos();
        let a1 = (-1.414 * PI / 10.0).exp();
        // The crate SuperSmoother rounds the book's angle 1.414 * pi to 4.4422.
        let c2 = 2.0 * a1 * (4.4422_f64 / 10.0).cos();
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;
        let mut e = ebsw(40);
        let rng = romu::Rng::from_seed_with_64bit(9);
        let (mut close_1, mut hp, mut hp_1) = (0.0, 0.0, 0.0);
        let mut filt = [0.0; 3];
        for bar in 0..500 {
            let close = 100.0 + 10.0 * rng.f64();
            if bar == 0 {
                close_1 = close;
            }
            hp = 0.5 * (1.0 + alpha1) * (close - close_1) + alpha1 * hp;
            close_1 = close;
            e.update(close);
            // The crate SuperSmoother starts after the high-pass warm-up, with HP[1] = 0.
            if bar < 39 {
                continue;
            }
            let f = c1 * (hp + hp_1) / 2.0 + c2 * filt[0] + c3 * filt[1];
            hp_1 = hp;
            filt = [f, filt[0], filt[1]];
            if bar < 39 + 9 + 2 {
                assert_eq!(e.last(), None);
                continue;
            }
            let wave = (filt[0] + filt[1] + filt[2]) / 3.0;
            let pwr = (filt[0] * filt[0] + filt[1] * filt[1] + filt[2] * filt[2]) / 3.0;
            let expected = wave / pwr.sqrt();
            let got = e.last().unwrap();
            assert!(
                (got - expected).abs() < 1e-9,
                "bar {bar}: {got} != {expected}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "duration must be >= 5")]
    fn even_better_sinewave_duration_too_small() {
        let _ = ebsw(4);
    }
}

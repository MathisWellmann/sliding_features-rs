//! Rolling sample excess kurtosis over a sliding window.

use std::num::NonZeroUsize;

use num::Float;

use super::statistics_moments::StatisticsMoments;
use crate::View;

/// Rolling sample excess kurtosis over a sliding window.
///
/// Computes the bias-adjusted excess kurtosis
/// `G2 = ((n - 1) / ((n - 2)(n - 3))) * ((n + 1) g2 + 6)`, with the
/// population (biased) excess kurtosis `g2 = m4 / m2^2 - 3` (`m2`, `m4` the
/// second and fourth central moments of the window). `G2` is the
/// convention used by Excel's `KURT`, SciPy's `kurtosis(..., bias=False)`
/// and pandas' `Series.kurt` (Joanes & Gill, 1998, "Comparing measures of
/// sample skewness and kurtosis", The Statistician 47(1), 183-189);
/// chosen, like [`super::Skewness`]'s `G1`, because it corrects the
/// small-sample bias of the plain population estimator and matches what
/// users of those tools expect.
///
/// `last()` is `None` until `window_len` values were observed, and
/// `Some(0)` if the window has zero variance (no peakedness is defined
/// then). A normal distribution gives values near 0; a uniform
/// distribution gives values near -1.2 (lighter tails than normal); a
/// positive value indicates heavier tails than normal.
///
/// # Complexity
/// O(1) amortized per update (see the crate-private `StatisticsMoments`
/// helper); O(1) for `last()`.
#[derive(Debug, Clone)]
pub struct Kurtosis<T, V> {
    view: V,
    moments: StatisticsMoments<T>,
}

impl<T, V> Kurtosis<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling excess kurtosis with a chained `View` and a
    /// sliding window length.
    ///
    /// # Panics
    /// If `window_len` is below 4: the bias adjustment divides by
    /// `(n - 2)(n - 3)`.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        assert!(window_len.get() >= 4, "window_len must be >= 4");
        Self {
            view,
            moments: StatisticsMoments::new(window_len),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.moments.window_len()
    }
}

impl<T, V> View<T> for Kurtosis<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");
        self.moments.push(val);
    }

    fn last(&self) -> Option<T> {
        if !self.moments.is_full() {
            return None;
        }
        let m2 = self.moments.central_moment2();
        if m2 <= T::zero() {
            return Some(T::zero());
        }
        let m4 = self.moments.central_moment4();
        let three = T::from(3.0).expect("can convert");
        let g2 = m4 / (m2 * m2) - three;

        let n = T::from(self.window_len().get()).expect("can convert");
        let one = T::one();
        let two = one + one;
        let six = three * two;
        let adjust = (n - one) / ((n - two) * (n - three));
        let out = adjust * ((n + one) * g2 + six);
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
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

    fn kurt(window_len: usize) -> Kurtosis<f64, Echo<f64>> {
        Kurtosis::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    /// Naive bias-adjusted excess kurtosis computed directly from a slice.
    fn naive_kurtosis(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let m2 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let m4 = vals.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
        if m2 <= 0.0 {
            return 0.0;
        }
        let g2 = m4 / (m2 * m2) - 3.0;
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
    }

    #[test]
    fn kurtosis_plot() {
        let mut v = kurt(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/kurtosis.png").unwrap();
    }

    #[test]
    fn kurtosis_warmup() {
        let mut v = kurt(5);
        for x in [1.0, 2.0, 3.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(5.0);
        assert!(v.last().is_some());
    }

    #[test]
    fn kurtosis_constant_is_zero() {
        let mut v = kurt(6);
        for _ in 0..20 {
            v.update(7.0);
        }
        assert_eq!(v.last(), Some(0.0));
    }

    #[test]
    fn kurtosis_of_normal_is_near_zero() {
        // Box-Muller normal samples; loose tolerance for sampling noise.
        let rng = romu::Rng::from_seed_with_64bit(3);
        let mut v = kurt(20_000);
        for _ in 0..20_000 {
            let u1 = rng.f64().max(1e-12);
            let u2 = rng.f64();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            v.update(z);
        }
        let got = v.last().expect("warm");
        assert!(got.abs() < 0.2, "expected near 0, got {got}");
    }

    #[test]
    fn kurtosis_of_uniform_is_near_minus_1_2() {
        let rng = romu::Rng::from_seed_with_64bit(4);
        let mut v = kurt(20_000);
        for _ in 0..20_000 {
            v.update(rng.f64());
        }
        let got = v.last().expect("warm");
        assert!((got - (-1.2)).abs() < 0.1, "expected near -1.2, got {got}");
    }

    #[test]
    fn kurtosis_vs_naive_with_offset() {
        let rng = romu::Rng::from_seed_with_64bit(12);
        // `window_len = 4` is excluded: it divides `g2` by `(n - 2)(n - 3)
        // = 2` and `m2^2` where `m2` can by chance be tiny for only 4
        // samples, so adjusted kurtosis is inherently ill-conditioned there
        // (any two correct implementations can disagree sharply); the
        // dedicated `kurtosis_warmup` test covers `window_len = 4` itself.
        for window_len in [5, 10, 50] {
            let mut v = kurt(window_len);
            let vals: Vec<f64> = (0..1_000).map(|_| 1e6 + rng.f64() * 5.0).collect();
            for (i, x) in vals.iter().enumerate() {
                v.update(*x);
                if i + 1 < window_len {
                    assert_eq!(v.last(), None);
                    continue;
                }
                let expected = naive_kurtosis(&vals[i + 1 - window_len..=i]);
                let got = v.last().expect("warm");
                assert!(
                    (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "window_len {window_len} step {i}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "window_len must be >= 4")]
    fn kurtosis_window_len_too_small() {
        let _ = kurt(3);
    }
}

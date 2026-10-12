//! Rolling rank-based normal scores transform ("Gaussian rank" normalizer).

use std::num::NonZeroUsize;

use num::Float;

use super::sorted_window::SortedWindow;
use crate::View;

/// Rolling van der Waerden normal scores transform (van der Waerden, 1952,
/// "Order tests for the two-sample problem and their power") of the most
/// recent value within the last `window_len` values (including itself),
/// also known as a "Gaussian rank" normalizer.
///
/// The latest value's mid-rank `r` (average of the smallest and largest
/// 1-indexed rank among ties, so ties share a rank) is mapped to a uniform
/// plotting position `u = r / (n + 1)` in the open interval `(0, 1)` (n is
/// `window_len`, so `u` never reaches the boundary), which is then mapped to
/// a standard normal quantile `z = Phi^-1(u)` with Acklam's (2003) rational
/// approximation of the inverse normal CDF (absolute error below `1.15e-9`).
/// This reshapes an arbitrarily distributed window into an approximately
/// standard-normal output, which is useful as a robust alternative to
/// z-score standardization ahead of models that assume normality.
///
/// `last()` is `None` until `window_len` values were observed.
/// Output range: approximately `(-infinity, infinity)`, in practice bounded
/// by `Phi^-1(1 / (window_len + 1))` and `Phi^-1(window_len / (window_len + 1))`.
///
/// Complexity per update: O(log n) binary search plus an O(n) `memmove` in a
/// sorted `Vec` (fast for typical window sizes); O(log n) for `last()` (two
/// binary searches plus the O(1) inverse normal CDF approximation); O(n) memory.
#[derive(Debug, Clone)]
pub struct QuantileNormalizer<T, V> {
    view: V,
    window: SortedWindow<T>,
    latest: T,
}

impl<T, V> QuantileNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new rolling quantile normalizer with a chained `View` and a
    /// sliding window length.
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window: SortedWindow::new(window_len),
            latest: T::zero(),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.window.window_len()
    }
}

impl<T, V> View<T> for QuantileNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.latest = val;
        let _evicted = self.window.push(val);
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if !self.window.is_full() {
            return None;
        }
        let n = self.window.window_len().get();
        let less = self.window.count_less(self.latest);
        let less_equal = self.window.count_less_equal(self.latest);
        let one = T::one();
        let two = T::from(2.0).expect("can convert");
        let mid_rank =
            (T::from(less).expect("can convert") + T::from(less_equal).expect("can convert") + one)
                / two;
        let u = mid_rank / (T::from(n).expect("can convert") + one);
        debug_assert!(
            u > T::zero(),
            "plotting position is in the open interval (0, 1)"
        );
        debug_assert!(u < one, "plotting position is in the open interval (0, 1)");
        let out = inverse_normal_cdf(u);
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

/// Acklam's (2003) rational approximation of the inverse standard normal CDF
/// `Phi^-1(p)`, with an absolute error below `1.15e-9` over `p` in `(0, 1)`.
///
/// # Panics
/// If `p` is not in `(0, 1)`.
fn inverse_normal_cdf<T: Float>(p: T) -> T {
    assert!(p > T::zero(), "p must be > 0");
    assert!(p < T::one(), "p must be < 1");

    // Rational approximation coefficients for the lower, central and upper
    // regions, from P. J. Acklam, "An algorithm for computing the inverse
    // normal cumulative distribution function", 2003.
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let f = |x: f64| T::from(x).expect("can convert");
    let p_low = f(0.024_25);
    let p_high = T::one() - p_low;

    if p < p_low {
        let q = (-(p.ln()) * f(2.0)).sqrt();
        (((((f(c[0]) * q + f(c[1])) * q + f(c[2])) * q + f(c[3])) * q + f(c[4])) * q + f(c[5]))
            / ((((f(d[0]) * q + f(d[1])) * q + f(d[2])) * q + f(d[3])) * q + T::one())
    } else if p <= p_high {
        let q = p - f(0.5);
        let r = q * q;
        (((((f(a[0]) * r + f(a[1])) * r + f(a[2])) * r + f(a[3])) * r + f(a[4])) * r + f(a[5])) * q
            / (((((f(b[0]) * r + f(b[1])) * r + f(b[2])) * r + f(b[3])) * r + f(b[4])) * r
                + T::one())
    } else {
        let q = (-((T::one() - p).ln()) * f(2.0)).sqrt();
        -(((((f(c[0]) * q + f(c[1])) * q + f(c[2])) * q + f(c[3])) * q + f(c[4])) * q + f(c[5]))
            / ((((f(d[0]) * q + f(d[1])) * q + f(d[2])) * q + f(d[3])) * q + T::one())
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

    fn qn(window_len: usize) -> QuantileNormalizer<f64, Echo<f64>> {
        QuantileNormalizer::new(Echo::new(), NonZeroUsize::new(window_len).unwrap())
    }

    #[test]
    fn inverse_normal_cdf_known_values() {
        // Standard normal quantiles, e.g. Abramowitz & Stegun Table 26.1 /
        // scipy.stats.norm.ppf.
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-9);
        assert!((inverse_normal_cdf(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((inverse_normal_cdf(0.025) - (-1.959_963_984_540_054)).abs() < 1e-8);
        assert!((inverse_normal_cdf(0.841_344_746_068_543) - 1.0).abs() < 1e-8);
    }

    #[test]
    #[should_panic(expected = "p must be > 0")]
    fn inverse_normal_cdf_p_zero() {
        let _ = inverse_normal_cdf(0.0);
    }

    #[test]
    #[should_panic(expected = "p must be < 1")]
    fn inverse_normal_cdf_p_one() {
        let _ = inverse_normal_cdf(1.0);
    }

    #[test]
    fn quantile_normalizer_plot() {
        let mut v = qn(32);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            v.update(*x);
            if let Some(val) = v.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/quantile_normalizer.png").unwrap();
    }

    #[test]
    fn quantile_normalizer_warmup_and_hand_computed() {
        let mut v = qn(4);
        assert_eq!(v.window_len().get(), 4);
        for x in [3.0, 1.0, 4.0] {
            v.update(x);
            assert_eq!(v.last(), None);
        }
        v.update(2.0);
        // window [1, 2, 3, 4], latest = 2: less = 1, less_equal = 2
        // mid_rank = (1 + 2 + 1) / 2 = 2, u = 2 / 5 = 0.4
        let expected = inverse_normal_cdf(0.4);
        assert!((v.last().expect("warm") - expected).abs() < 1e-12);
    }

    #[test]
    fn quantile_normalizer_constant_input_is_zero() {
        let mut v = qn(5);
        for _ in 0..20 {
            v.update(7.5);
        }
        assert!((v.last().expect("warm") - 0.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_normalizer_is_monotone_in_rank() {
        // Fix the first `window_len - 1` values and vary only the latest
        // one: the normalized score must increase strictly as the latest
        // raw value increases, as long as each `last` falls strictly between
        // (or below/above) distinct base elements, so it is guaranteed a
        // distinct rank within the window (two raw values that both exceed
        // every base element tie for the maximum rank, and so would not be
        // distinguished by a rank-based transform).
        let base = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0];
        let mut prev_out = f64::NEG_INFINITY;
        for last in [0.0, 1.5, 4.0, 6.0, 8.5, 10.0] {
            let mut v = qn(8);
            for x in base {
                v.update(x);
            }
            v.update(last);
            let out = v.last().expect("warm");
            assert!(out > prev_out, "last {last}: out {out}, prev {prev_out}");
            prev_out = out;
        }
    }
}

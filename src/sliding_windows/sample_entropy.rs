// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sample entropy (SampEn) of a sliding window, after Richman & Moorman (2000).

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Sample entropy `SampEn(m, r)` of the last `window_len` values after
/// J. S. Richman and J. R. Moorman, "Physiological time-series analysis using approximate
/// entropy and sample entropy", Am. J. Physiol. Heart Circ. Physiol. 278 (2000).
///
/// With `N = window_len`, the `N - m` templates `x[i..i+m]` (`i < N - m`) are compared pairwise
/// (no self-matches) with the Chebyshev distance. `B` counts pairs within tolerance `r'` for length
/// `m`, `A` counts pairs that also match at length `m + 1`. `SampEn = -ln(A / B)`.
///
/// The tolerance is `r' = r * sd`, where `r` is the constructor argument (a fraction, commonly
/// 0.1..0.25) and `sd` is the sample standard deviation (`N - 1` denominator) of the window,
/// so the measure is scale-free. A pair matches when the distance is `<= r'`, as in the paper.
///
/// Output range `[0, ln((N - m) * (N - m - 1) / 2)]`: low for regular series, high for irregular.
/// - When `A == 0` or `B == 0` SampEn is undefined and `last` returns `None` even after warm-up.
/// - A constant window has `sd = 0`, so `r' = 0` and every pair matches: output 0.
///
/// Complexity per update: O(N^2 * m) worst case (all pairwise template comparisons, early exit
/// on the first mismatching element), no allocations.
#[derive(Debug, Clone, CopyGetters)]
pub struct SampleEntropy<T: Float, V> {
    view: V,
    /// The sliding window length N.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The template length m.
    #[getset(get_copy = "pub")]
    m: NonZeroUsize,
    /// The tolerance as a fraction of the window's standard deviation.
    #[getset(get_copy = "pub")]
    r: T,
    window: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> SampleEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `SampleEntropy` with a chained `View`.
    ///
    /// # Panics
    /// If `window_len < m + 2` (fewer than two templates of length `m + 1`),
    /// or if `r` is not finite and positive.
    pub fn new(view: V, window_len: NonZeroUsize, m: NonZeroUsize, r: T) -> Self {
        assert!(
            window_len.get() >= m.get() + 2,
            "window_len must be at least m + 2"
        );
        assert!(r.is_finite(), "r must be finite");
        assert!(r > T::zero(), "r must be positive");
        Self {
            view,
            window_len,
            m,
            r,
            window: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

/// Sample standard deviation with the `n - 1` denominator, two-pass for accuracy.
fn sample_std<T: Float>(x: &[T]) -> T {
    debug_assert!(x.len() >= 2, "need at least two values");
    let n = T::from(x.len()).expect("can convert");
    let mean = x.iter().fold(T::zero(), |acc, &v| acc + v) / n;
    let ss = x
        .iter()
        .fold(T::zero(), |acc, &v| acc + (v - mean) * (v - mean));
    (ss / (n - T::one())).sqrt()
}

/// Returns the match counts `(A, B)` for template lengths `m + 1` and `m` within `tol`.
fn count_matches<T: Float>(x: &[T], m: usize, tol: T) -> (u64, u64) {
    let num_templates = x.len() - m;
    let (mut a, mut b) = (0_u64, 0_u64);
    for i in 0..num_templates {
        for j in i + 1..num_templates {
            let matches_m = (0..m).all(|k| (x[i + k] - x[j + k]).abs() <= tol);
            if !matches_m {
                continue;
            }
            b += 1;
            if (x[i + m] - x[j + m]).abs() <= tol {
                a += 1;
            }
        }
    }
    debug_assert!(a <= b, "every length m + 1 match is a length m match");
    (a, b)
}

/// `SampEn = -ln(A / B)` of the slice, `None` if undefined.
fn sample_entropy<T: Float>(x: &[T], m: usize, r: T) -> Option<T> {
    let tol = r * sample_std(x);
    let (a, b) = count_matches(x, m, tol);
    if a == 0 {
        return None;
    }
    let a = T::from(a).expect("can convert");
    let b = T::from(b).expect("can convert");
    let out = -(a / b).ln();
    debug_assert!(out >= T::zero(), "SampEn is non-negative");
    Some(out)
}

impl<T, V> View<T> for SampleEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.window.len() == self.window_len.get() {
            self.window.pop_front();
        }
        self.window.push_back(val);
        if self.window.len() < self.window_len.get() {
            return;
        }
        self.out = sample_entropy(self.window.make_contiguous(), self.m.get(), self.r);
        if let Some(out) = self.out {
            debug_assert!(out.is_finite(), "value must be finite");
        }
    }

    #[inline]
    fn last(&self) -> Option<T> {
        self.out
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

    fn new_f64(window_len: usize, m: usize, r: f64) -> SampleEntropy<f64, Echo<f64>> {
        SampleEntropy::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            NonZeroUsize::new(m).unwrap(),
            r,
        )
    }

    /// Reference following the paper literally: B_i and A_i per template, excluding self-matches.
    fn naive(x: &[f64], m: usize, r: f64) -> Option<f64> {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        let tol = r * sd;
        let dist = |i: usize, j: usize, len: usize| {
            (0..len)
                .map(|k| (x[i + k] - x[j + k]).abs())
                .fold(0.0, f64::max)
        };
        let (mut a, mut b) = (0, 0);
        for i in 0..n - m {
            for j in 0..n - m {
                if i == j {
                    continue;
                }
                b += (dist(i, j, m) <= tol) as usize;
                a += (dist(i, j, m + 1) <= tol) as usize;
            }
        }
        (a > 0).then(|| -(a as f64 / b as f64).ln())
    }

    #[test]
    fn sample_entropy_warmup() {
        let mut se = new_f64(10, 2, 0.2);
        for i in 0..9 {
            se.update((i % 3) as f64);
            assert_eq!(se.last(), None);
        }
        se.update(0.0);
        assert!(se.last().is_some());
    }

    #[test]
    fn sample_entropy_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(17);
        for (window_len, m, r) in [(40, 2, 0.2), (25, 1, 0.5), (60, 3, 0.3)] {
            let mut se = new_f64(window_len, m, r);
            let mut vals = Vec::new();
            for _ in 0..400 {
                let v = rng.f64();
                vals.push(v);
                se.update(v);
                if vals.len() < window_len {
                    continue;
                }
                let expected = naive(&vals[vals.len() - window_len..], m, r);
                match (se.last(), expected) {
                    (Some(got), Some(exp)) => assert!((got - exp).abs() < 1e-9, "{got} vs {exp}"),
                    (got, exp) => assert_eq!(got, exp),
                }
            }
        }
    }

    #[test]
    fn sample_entropy_uniform_noise_matches_theory() {
        // For iid data, A / B -> P(|x - y| <= r'), for U(0, 1): 2 r' - r'^2 with sd = 1 / sqrt(12).
        let rng = romu::Rng::from_seed_with_64bit(23);
        let mut se = new_f64(2_000, 2, 0.2);
        for _ in 0..2_000 {
            se.update(rng.f64());
        }
        let tol = 0.2 / 12.0_f64.sqrt();
        let expected = -(2.0 * tol - tol * tol).ln();
        let got = se.last().unwrap();
        assert!((got - expected).abs() < 0.1, "{got} vs {expected}");
    }

    #[test]
    fn sample_entropy_regular_below_irregular() {
        let rng = romu::Rng::from_seed_with_64bit(29);
        let mut sine = new_f64(200, 2, 0.2);
        let mut noise = new_f64(200, 2, 0.2);
        for i in 0..400 {
            sine.update((i as f64 * 0.3).sin());
            noise.update(rng.f64());
        }
        let (s, n) = (sine.last().unwrap(), noise.last().unwrap());
        assert!(s + 1.0 < n, "sine {s} vs noise {n}");
    }

    #[test]
    fn sample_entropy_constant_is_zero() {
        let mut se = new_f64(20, 2, 0.2);
        for _ in 0..30 {
            se.update(4.2);
        }
        assert_eq!(se.last().unwrap(), 0.0);
    }

    #[test]
    fn sample_entropy_no_matches_is_none() {
        // Quadratically growing gaps never match within a small tolerance.
        let mut se = new_f64(20, 2, 0.001);
        for i in 0..30 {
            se.update((i * i) as f64);
        }
        assert_eq!(se.last(), None);
    }

    #[test]
    #[should_panic(expected = "window_len must be at least m + 2")]
    fn sample_entropy_window_too_small() {
        let _ = new_f64(3, 2, 0.2);
    }

    #[test]
    #[should_panic(expected = "r must be positive")]
    fn sample_entropy_zero_r() {
        let _ = new_f64(10, 2, 0.0);
    }

    #[test]
    #[should_panic(expected = "r must be finite")]
    fn sample_entropy_nan_r() {
        let _ = new_f64(10, 2, f64::NAN);
    }

    #[test]
    fn sample_entropy_plot() {
        let mut se = new_f64(64, 2, 0.2);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            se.update(*v);
            if let Some(val) = se.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/sample_entropy.png").unwrap();
    }
}

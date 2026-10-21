// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! ADWIN2 adaptive windowing change detector.

use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// The maximum number of exponential histogram levels. Level `i` holds buckets of
/// `2^i` values, so the window could hold more than `2^63` values before running out
/// of levels, which is never reached in practice.
const MAX_LEVELS: usize = 64;

/// Both sub-windows of a cut must hold at least this many values (as in MOA),
/// so a few outliers at the window edges do not trigger a cut.
const MIN_SUB_WINDOW: usize = 5;

/// ADWIN2 (ADaptive WINdowing): a window of the most recent values that shrinks
/// automatically when the mean of the data changes.
///
/// Reference: A. Bifet, R. Gavalda (2007), "Learning from Time-Changing Data with
/// Adaptive Windowing", SIAM International Conference on Data Mining (SDM);
/// implementation details (bucket merging, cut-check loop, `clock`) follow `ADWIN` of MOA.
///
/// The window `W` grows with every value. Every `clock` updates, all split points of
/// `W` into an older part `W0` (`n0` values, mean `mu0`) and a newer part `W1` (`n1`
/// values, mean `mu1`) at bucket boundaries are checked, and the oldest bucket is dropped
/// while some split has `|mu0 - mu1| > eps_cut`, with the variance-based bound of the paper:
///
/// `eps_cut = sqrt(2 / m * var_W * ln(2 / delta')) + 2 / (3 m) * ln(2 / delta')`,
///
/// where `m = 1 / (1 / n0 + 1 / n1)`, `var_W` is the (population) variance of `W` and
/// `delta' = delta / ln(n)` (only `O(log n)` split points are checked, as in ADWIN2 and MOA).
/// Both sub-windows must hold at least 5 values.
///
/// `W` is stored compressed as an exponential histogram: level `i` holds at most
/// `max_buckets` (`M`) buckets of `2^i` values each, keeping their sum and variance.
/// When a level gets `M + 1` buckets, its two oldest are merged into the next level.
/// The window boundaries are therefore bucket boundaries, so a cut can only drop whole
/// buckets, while mean and variance of the retained window are exact.
///
/// Parameters:
/// - `delta`: confidence in `(0, 1)`; smaller values give fewer false alarms but slower
///   detection (MOA default: 0.002).
/// - `max_buckets`: `M`, the buckets per level; larger values give finer cut points at the
///   cost of memory and time (MOA default: 5).
/// - `clock`: check for cuts every `clock` updates (MOA default: 32); `1` checks every
///   update for the shortest detection delay.
///
/// Output: the mean of the current window. Secondary outputs: [`Adwin::width`],
/// [`Adwin::variance`] and [`Adwin::change_detected`] for the latest update.
///
/// Warm-up: `None` until the chained `View` yields its first value.
/// Complexity: insertion is amortized O(1) (worst case O(log W) merges), a cut check
/// is O(M log W) per dropped bucket. Memory is bounded by `64 (M + 1)` buckets,
/// preallocated in `new`, so updates never allocate.
///
/// Placed in `sliding_windows`, because it maintains a (compressed, adaptive) window
/// of history, unlike the constant-state detectors in `rolling`.
#[derive(Debug, Clone, CopyGetters)]
pub struct Adwin<T, V> {
    view: V,
    delta: T,
    /// The maximum number of buckets per level (`M`).
    #[getset(get_copy = "pub")]
    max_buckets: NonZeroUsize,
    /// The number of updates between cut checks.
    #[getset(get_copy = "pub")]
    clock: NonZeroUsize,
    /// Bucket sums. Level `i` occupies `[i * (M + 1), (i + 1) * (M + 1))`, oldest first.
    totals: Vec<T>,
    /// Bucket sums of squared deviations from the bucket mean, laid out like `totals`.
    variances: Vec<T>,
    /// Number of buckets at each level.
    level_len: Vec<usize>,
    /// Number of non-empty levels; the oldest values live in the highest level.
    num_levels: usize,
    /// The number of values in the window.
    #[getset(get_copy = "pub")]
    width: usize,
    total: T,
    /// Sum of squared deviations from the window mean.
    m2: T,
    time: usize,
    /// Whether the latest update dropped values from the window because of a change.
    #[getset(get_copy = "pub")]
    change_detected: bool,
}

impl<T, V> Adwin<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new ADWIN2 detector with a chained `View`, the confidence `delta`,
    /// the buckets per level `max_buckets` and the cut-check period `clock`.
    ///
    /// # Panics
    /// If `delta` is not in `(0, 1)`.
    pub fn new(view: V, delta: T, max_buckets: NonZeroUsize, clock: NonZeroUsize) -> Self {
        assert!(delta > T::zero(), "delta must be > 0");
        assert!(delta < T::one(), "delta must be < 1");
        let capacity = MAX_LEVELS * (max_buckets.get() + 1);
        Self {
            view,
            delta,
            max_buckets,
            clock,
            totals: vec![T::zero(); capacity],
            variances: vec![T::zero(); capacity],
            level_len: vec![0; MAX_LEVELS],
            num_levels: 0,
            width: 0,
            total: T::zero(),
            m2: T::zero(),
            time: 0,
            change_detected: false,
        }
    }

    /// The confidence parameter.
    #[inline]
    pub fn delta(&self) -> T {
        self.delta
    }

    /// The (population) variance of the values in the window, zero if it is empty.
    #[inline]
    pub fn variance(&self) -> T {
        if self.width == 0 {
            return T::zero();
        }
        self.m2 / T::from(self.width).expect("can convert")
    }

    #[inline]
    fn slot(&self, level: usize, k: usize) -> usize {
        debug_assert!(level < MAX_LEVELS, "level out of bounds");
        debug_assert!(k <= self.max_buckets.get(), "bucket out of bounds");
        level * (self.max_buckets.get() + 1) + k
    }

    #[inline]
    fn bucket_size(level: usize) -> T {
        T::from(1_usize << level).expect("can convert")
    }

    fn push_bucket(&mut self, level: usize, total: T, m2: T) {
        assert!(level < MAX_LEVELS, "ADWIN ran out of histogram levels");
        let len = self.level_len[level];
        assert!(len <= self.max_buckets.get(), "level overflow");
        let idx = self.slot(level, len);
        self.totals[idx] = total;
        self.variances[idx] = m2;
        self.level_len[level] = len + 1;
        self.num_levels = self.num_levels.max(level + 1);
    }

    fn remove_oldest_buckets(&mut self, level: usize, count: usize) {
        let len = self.level_len[level];
        assert!(count <= len, "can not remove more buckets than present");
        let start = self.slot(level, 0);
        self.totals.copy_within(start + count..start + len, start);
        self.variances
            .copy_within(start + count..start + len, start);
        self.level_len[level] = len - count;
        // Levels below the top can be empty (e.g. with `M = 1`), so skip all of them.
        while self.num_levels > 0 && self.level_len[self.num_levels - 1] == 0 {
            self.num_levels -= 1;
        }
    }

    fn insert(&mut self, val: T) {
        self.width += 1;
        if self.width > 1 {
            // Merge a single value into the window (parallel variance formula).
            let n_old = T::from(self.width - 1).expect("can convert");
            let n = T::from(self.width).expect("can convert");
            let diff = val - self.total / n_old;
            self.m2 = self.m2 + n_old * diff * diff / n;
        }
        self.total = self.total + val;
        self.push_bucket(0, val, T::zero());
        self.compress();
    }

    /// Merge the two oldest buckets of every full level into the next level.
    fn compress(&mut self) {
        let max_buckets = self.max_buckets.get();
        let mut level = 0;
        while self.level_len[level] > max_buckets {
            let (a, b) = (self.slot(level, 0), self.slot(level, 1));
            let n = Self::bucket_size(level);
            let diff = self.totals[a] / n - self.totals[b] / n;
            let two = T::from(2).expect("can convert");
            let m2 = self.variances[a] + self.variances[b] + n * diff * diff / two;
            let total = self.totals[a] + self.totals[b];
            self.remove_oldest_buckets(level, 2);
            self.push_bucket(level + 1, total, m2);
            level += 1;
        }
        debug_assert!(
            self.level_len[..self.num_levels]
                .iter()
                .all(|len| *len <= max_buckets),
            "every level holds at most M buckets after compression"
        );
    }

    /// Drop the oldest bucket of the window.
    fn delete_oldest(&mut self) {
        assert!(self.num_levels > 0, "the window is not empty");
        let level = self.num_levels - 1;
        let idx = self.slot(level, 0);
        let (bucket_total, bucket_m2) = (self.totals[idx], self.variances[idx]);
        let size = 1_usize << level;
        assert!(size < self.width, "the newest values are never dropped");
        self.width -= size;
        self.total = self.total - bucket_total;
        let n1 = Self::bucket_size(level);
        let n = T::from(self.width).expect("can convert");
        let diff = bucket_total / n1 - self.total / n;
        let removed = bucket_m2 + n1 * n * diff * diff / (n1 + n);
        self.m2 = (self.m2 - removed).max(T::zero());
        self.remove_oldest_buckets(level, 1);
    }

    /// Whether some split of the window into an older and a newer part at a bucket
    /// boundary has significantly different means.
    fn has_cut(&self) -> bool {
        let n = T::from(self.width).expect("can convert");
        let two = T::from(2).expect("can convert");
        let ln_term = (two * n.ln() / self.delta).ln();
        let var_w = self.variance();
        let mut n0 = 0_usize;
        let mut sum0 = T::zero();
        for level in (0..self.num_levels).rev() {
            for k in 0..self.level_len[level] {
                n0 += 1 << level;
                sum0 = sum0 + self.totals[self.slot(level, k)];
                let n1 = self.width - n0;
                // This also keeps the newest bucket in the newer part.
                if n0 < MIN_SUB_WINDOW || n1 < MIN_SUB_WINDOW {
                    continue;
                }
                let n0_t = T::from(n0).expect("can convert");
                let n1_t = T::from(n1).expect("can convert");
                let m_inv = n0_t.recip() + n1_t.recip();
                let eps = (two * m_inv * var_w * ln_term).sqrt()
                    + two / T::from(3).expect("can convert") * m_inv * ln_term;
                let mean_diff = sum0 / n0_t - (self.total - sum0) / n1_t;
                if mean_diff.abs() > eps {
                    return true;
                }
            }
        }
        false
    }
}

impl<T, V> View<T> for Adwin<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.insert(val);
        self.time += 1;
        self.change_detected = false;
        if self.time.is_multiple_of(self.clock.get()) && self.width >= 2 * MIN_SUB_WINDOW {
            while self.has_cut() {
                self.delete_oldest();
                self.change_detected = true;
            }
        }
        debug_assert!(self.width > 0, "the latest value is always in the window");
        debug_assert!(self.m2 >= T::zero(), "variance is non-negative");
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.width == 0 {
            return None;
        }
        let out = self.total / T::from(self.width).expect("can convert");
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

    /// Standard normal sample via the Box-Muller transform on romu uniforms.
    fn gaussian(rng: &romu::Rng) -> f64 {
        let u1 = 1.0 - rng.f64(); // in (0, 1], so `ln` is finite
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn adwin(delta: f64, max_buckets: usize, clock: usize) -> Adwin<f64, Echo<f64>> {
        Adwin::new(
            Echo::new(),
            delta,
            NonZeroUsize::new(max_buckets).unwrap(),
            NonZeroUsize::new(clock).unwrap(),
        )
    }

    /// Check the bucket invariants and compare mean and variance of the window with
    /// the naive computation over the last `width` inputs.
    fn check_vs_naive(a: &Adwin<f64, Echo<f64>>, inputs: &[f64]) {
        let m = a.max_buckets().get();
        let mut covered = 0;
        for level in 0..a.num_levels {
            assert!(a.level_len[level] <= m, "level {level} over capacity");
            covered += a.level_len[level] << level;
        }
        assert_eq!(covered, a.width());
        assert!(a.level_len[a.num_levels - 1] > 0, "top level is not empty");
        // The top bucket alone holds 2^(levels - 1) values.
        assert!(a.num_levels <= (a.width() as f64).log2() as usize + 1);

        let w = &inputs[inputs.len() - a.width()..];
        let n = w.len() as f64;
        let mean = w.iter().sum::<f64>() / n;
        let var = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((a.last().unwrap() - mean).abs() < 1e-9);
        assert!((a.variance() - var).abs() < 1e-9 * var.max(1.0));
    }

    #[test]
    fn adwin_plot() {
        let mut a = adwin(0.002, 5, 32);
        let mut out = Vec::with_capacity(TEST_DATA.len());
        for x in &TEST_DATA {
            a.update(*x);
            out.push(a.last().unwrap());
        }
        plot_values(out, "img/adwin.png").unwrap();
    }

    #[test]
    fn adwin_warmup() {
        let mut a = adwin(0.002, 5, 32);
        assert_eq!(a.last(), None);
        assert_eq!(a.width(), 0);
        assert_eq!(a.variance(), 0.0);
        a.update(3.0);
        assert_eq!(a.last(), Some(3.0));
        assert_eq!(a.width(), 1);
        a.update(5.0);
        assert_eq!(a.last(), Some(4.0));
        assert_eq!(a.variance(), 1.0);
        assert!(!a.change_detected());
    }

    #[test]
    fn adwin_window_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(101);
        for (m, clock) in [(1, 1), (2, 1), (5, 32), (3, 7)] {
            let mut a = adwin(0.01, m, clock);
            let mut inputs = Vec::new();
            for i in 0..6_000 {
                // Several level shifts, so the window is cut repeatedly.
                let level = [0.0, 3.0, -1.0, 2.0][(i / 1_500) % 4];
                let x = level + gaussian(&rng);
                inputs.push(x);
                a.update(x);
                check_vs_naive(&a, &inputs);
            }
            assert!(a.width() < 3_000, "the window was cut");
        }
    }

    #[test]
    fn adwin_no_change_on_stationary_noise() {
        let rng = romu::Rng::from_seed_with_64bit(102);
        for clock in [1, 32] {
            let mut a = adwin(0.002, 5, clock);
            for i in 0..20_000 {
                a.update(5.0 + gaussian(&rng));
                assert!(!a.change_detected(), "false alarm at {i}");
            }
            assert_eq!(a.width(), 20_000);
            assert!((a.last().unwrap() - 5.0).abs() < 0.05);
            assert!((a.variance() - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn adwin_detects_mean_shift_and_shrinks() {
        for (seed, clock, shift) in [(103, 32, 1.0), (104, 1, 1.0), (105, 32, -2.0)] {
            let rng = romu::Rng::from_seed_with_64bit(seed);
            let mut a = adwin(0.002, 5, clock);
            for _ in 0..5_000 {
                a.update(gaussian(&rng));
                assert!(!a.change_detected());
            }
            assert_eq!(a.width(), 5_000);
            let delay = (1..=200)
                .find(|_| {
                    a.update(shift + gaussian(&rng));
                    a.change_detected()
                })
                .expect("the shift must be detected");
            assert!(delay <= 100, "delay {delay}, clock {clock}");
            assert!(a.width() < 1_000, "width {} after the cut", a.width());

            for _ in 0..2_000 {
                a.update(shift + gaussian(&rng));
            }
            // Only post-change values remain, plus a few pre-change values that share
            // buckets with post-change values, as cuts are only at bucket boundaries.
            assert!(a.width() <= 2_000 + delay + 256, "width {}", a.width());
            assert!(a.width() > 1_000, "width {}", a.width());
            let mean = a.last().unwrap();
            assert!((mean - shift).abs() < 0.1, "mean {mean} for shift {shift}");
        }
    }

    #[test]
    fn adwin_getters() {
        let a = adwin(0.01, 4, 16);
        assert_eq!(a.delta(), 0.01);
        assert_eq!(a.max_buckets().get(), 4);
        assert_eq!(a.clock().get(), 16);
    }

    #[test]
    #[should_panic(expected = "delta must be > 0")]
    fn adwin_zero_delta() {
        let _ = adwin(0.0, 5, 32);
    }

    #[test]
    #[should_panic(expected = "delta must be < 1")]
    fn adwin_delta_one() {
        let _ = adwin(1.0, 5, 32);
    }

    #[test]
    #[should_panic(expected = "delta must be > 0")]
    fn adwin_nan_delta() {
        let _ = adwin(f64::NAN, 5, 32);
    }
}

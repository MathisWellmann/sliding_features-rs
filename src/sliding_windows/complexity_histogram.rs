//! Shared histogram with an incrementally maintained Shannon entropy,
//! used by the entropy indicators of the complexity group.

use num::Float;

/// Minimum number of updates between exact recomputations of `sum_c_ln_c`,
/// so tiny histograms do not refresh on almost every update.
const MIN_REFRESH_INTERVAL: usize = 64;

/// Counts per category with `S = sum_i c_i * ln(c_i)` maintained incrementally.
///
/// The Shannon entropy of the empirical distribution is `H = ln(n) - S / n`,
/// so both `add`, `remove` and `entropy` are O(1) (amortized, see `tick`).
#[derive(Debug, Clone)]
pub(super) struct ComplexityHistogram<T> {
    counts: Vec<usize>,
    total: usize,
    // Number of non-empty categories, so a single occupied category gives exactly 0.
    occupied: usize,
    sum_c_ln_c: T,
    updates_since_refresh: usize,
}

impl<T: Float> ComplexityHistogram<T> {
    /// Create an empty histogram with `num_categories` categories.
    pub(super) fn new(num_categories: usize) -> Self {
        assert!(
            num_categories >= 2,
            "a histogram needs at least two categories"
        );
        Self {
            counts: vec![0; num_categories],
            total: 0,
            occupied: 0,
            sum_c_ln_c: T::zero(),
            updates_since_refresh: 0,
        }
    }

    /// Number of observations currently counted.
    #[inline]
    pub(super) fn total(&self) -> usize {
        self.total
    }

    /// Count one observation of category `idx`.
    #[inline]
    pub(super) fn add(&mut self, idx: usize) {
        assert!(idx < self.counts.len(), "category index out of range");
        let c = self.counts[idx];
        self.sum_c_ln_c = self.sum_c_ln_c + c_ln_c::<T>(c + 1) - c_ln_c::<T>(c);
        self.counts[idx] = c + 1;
        self.total += 1;
        if c == 0 {
            self.occupied += 1;
        }
        self.tick();
    }

    /// Remove one previously added observation of category `idx`.
    #[inline]
    pub(super) fn remove(&mut self, idx: usize) {
        assert!(idx < self.counts.len(), "category index out of range");
        let c = self.counts[idx];
        assert!(c > 0, "cannot remove from an empty category");
        assert!(self.total > 0, "cannot remove from an empty histogram");
        self.sum_c_ln_c = self.sum_c_ln_c + c_ln_c::<T>(c - 1) - c_ln_c::<T>(c);
        self.counts[idx] = c - 1;
        self.total -= 1;
        if c == 1 {
            self.occupied -= 1;
        }
        self.tick();
    }

    /// Floating point error of the incremental sum grows with every update,
    /// so it is recomputed exactly once every `max(categories, 64)` updates,
    /// which keeps the amortized cost O(1) and the drift bounded (matters for `f32`).
    #[inline]
    fn tick(&mut self) {
        self.updates_since_refresh += 1;
        if self.updates_since_refresh < self.counts.len().max(MIN_REFRESH_INTERVAL) {
            return;
        }
        self.updates_since_refresh = 0;
        self.sum_c_ln_c = self
            .counts
            .iter()
            .fold(T::zero(), |acc, &c| acc + c_ln_c::<T>(c));
        debug_assert_eq!(
            self.counts.iter().sum::<usize>(),
            self.total,
            "counts must add up to the total"
        );
    }

    /// Shannon entropy in nats of the counted observations, 0 when empty.
    #[inline]
    pub(super) fn entropy(&self) -> T {
        debug_assert!(
            self.occupied <= self.total,
            "occupied categories need counts"
        );
        // Exact 0 for a degenerate distribution, which the rounded sum cannot guarantee.
        if self.occupied <= 1 {
            return T::zero();
        }
        let n = T::from(self.total).expect("usize converts to float");
        let h = n.ln() - self.sum_c_ln_c / n;
        debug_assert!(h.is_finite(), "entropy must be finite");
        // Rounding can push an exact zero entropy slightly negative.
        h.max(T::zero())
    }
}

/// `c * ln(c)` with the limit value 0 for `c = 0`.
#[inline]
fn c_ln_c<T: Float>(c: usize) -> T {
    if c == 0 {
        return T::zero();
    }
    let c = T::from(c).expect("usize converts to float");
    c * c.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_entropy(counts: &[usize]) -> f64 {
        let n: usize = counts.iter().sum();
        counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n as f64;
                -p * p.ln()
            })
            .sum()
    }

    #[test]
    fn complexity_histogram_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(1);
        let mut h = ComplexityHistogram::<f64>::new(5);
        let mut counts = [0_usize; 5];
        for _ in 0..10_000 {
            let idx = rng.range_usize(0..5);
            if rng.bool() && counts[idx] > 0 {
                h.remove(idx);
                counts[idx] -= 1;
            } else {
                h.add(idx);
                counts[idx] += 1;
            }
            assert_eq!(h.total(), counts.iter().sum::<usize>());
            assert!((h.entropy() - naive_entropy(&counts)).abs() < 1e-9);
        }
    }

    #[test]
    #[should_panic(expected = "cannot remove from an empty category")]
    fn complexity_histogram_remove_empty() {
        let mut h = ComplexityHistogram::<f64>::new(3);
        h.add(0);
        h.remove(1);
    }

    #[test]
    #[should_panic(expected = "at least two categories")]
    fn complexity_histogram_one_category() {
        let _ = ComplexityHistogram::<f64>::new(1);
    }
}

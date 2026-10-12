//! Normalized permutation entropy of a sliding window, after Bandt & Pompe (2002).

use std::{
    collections::VecDeque,
    num::NonZeroUsize,
};

use getset::CopyGetters;
use num::Float;

use super::complexity_histogram::ComplexityHistogram;
use crate::View;

/// Smallest supported embedding order.
const MIN_ORDER: usize = 2;
/// Largest supported embedding order, `7! = 5040` patterns keeps the count table small.
const MAX_ORDER: usize = 7;

/// Normalized permutation entropy after
/// C. Bandt and B. Pompe, "Permutation Entropy: A Natural Complexity Measure for Time Series",
/// Phys. Rev. Lett. 88, 174102 (2002).
///
/// Each new value forms the embedding vector `(x_{t-(m-1)tau}, ..., x_{t-tau}, x_t)`
/// of `order` m and `delay` tau, which is mapped to its ordinal pattern (one of `m!` permutations).
/// The output is the Shannon entropy of the pattern frequencies over the last `window_len`
/// patterns, divided by `ln(m!)`.
///
/// Output range `[0, 1]`: 0 for a monotone series, near 1 for iid noise.
/// Reliable estimates need `window_len` much larger than `m!`.
///
/// Ties: equal values are ranked by their time of occurrence (the earlier one is smaller),
/// the convention of Bandt & Pompe. So a constant run looks like an increasing one and has
/// entropy 0. Inputs with many ties (heavily quantized prices) therefore bias the entropy down.
///
/// Warm-up: `(m - 1) * tau + window_len` values.
///
/// Complexity per update: O(m^2) to find the pattern plus O(1) (amortized) for the
/// incremental pattern counts and entropy.
#[derive(Debug, Clone, CopyGetters)]
pub struct PermutationEntropy<T: Float, V> {
    view: V,
    /// The number of ordinal patterns in the sliding window.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    /// The embedding order m (length of each ordinal pattern).
    #[getset(get_copy = "pub")]
    order: usize,
    /// The embedding delay tau (distance between pattern elements).
    #[getset(get_copy = "pub")]
    delay: NonZeroUsize,
    // The last `(m - 1) * tau + 1` values, oldest first.
    values: VecDeque<T>,
    // The pattern index of each value in the window, oldest first.
    patterns: VecDeque<usize>,
    histogram: ComplexityHistogram<T>,
    ln_num_patterns: T,
}

impl<T, V> PermutationEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new `PermutationEntropy` with a chained `View`.
    ///
    /// # Panics
    /// If `order` is not in `2..=7`.
    pub fn new(view: V, window_len: NonZeroUsize, order: usize, delay: NonZeroUsize) -> Self {
        assert!(order >= MIN_ORDER, "order must be at least 2");
        assert!(order <= MAX_ORDER, "order must be at most 7");
        let num_patterns: usize = (1..=order).product();
        let span = (order - 1) * delay.get() + 1;
        Self {
            view,
            window_len,
            order,
            delay,
            values: VecDeque::with_capacity(span),
            patterns: VecDeque::with_capacity(window_len.get()),
            histogram: ComplexityHistogram::new(num_patterns),
            ln_num_patterns: T::from(num_patterns).expect("can convert").ln(),
        }
    }
}

/// Index in `0..m!` of the ordinal pattern of `values[0], values[delay], ..., values[(m-1)*delay]`.
///
/// Uses the Lehmer code `L_i = #{j > i : x_j < x_i}`, which ranks ties by time of occurrence,
/// and folds it in the factorial number system: `sum_i L_i * (m-1-i)!`.
#[inline]
fn pattern_index<T: Float>(values: &VecDeque<T>, order: usize, delay: usize) -> usize {
    debug_assert_eq!(values.len(), (order - 1) * delay + 1);
    let mut idx = 0;
    for i in 0..order {
        let xi = values[i * delay];
        let lehmer = (i + 1..order).filter(|&j| values[j * delay] < xi).count();
        debug_assert!(lehmer < order - i, "Lehmer digit out of range");
        idx = idx * (order - i) + lehmer;
    }
    idx
}

impl<T, V> View<T> for PermutationEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let span = (self.order - 1) * self.delay.get() + 1;
        if self.values.len() == span {
            self.values.pop_front();
        }
        self.values.push_back(val);
        if self.values.len() < span {
            return;
        }

        if self.patterns.len() == self.window_len.get() {
            let old = self
                .patterns
                .pop_front()
                .expect("window is full, so not empty");
            self.histogram.remove(old);
        }
        let idx = pattern_index(&self.values, self.order, self.delay.get());
        self.patterns.push_back(idx);
        self.histogram.add(idx);
        debug_assert_eq!(self.histogram.total(), self.patterns.len());
    }

    fn last(&self) -> Option<T> {
        if self.patterns.len() < self.window_len.get() {
            return None;
        }
        let out = (self.histogram.entropy() / self.ln_num_patterns).min(T::one());
        debug_assert!(out.is_finite(), "value must be finite");
        debug_assert!(out >= T::zero(), "entropy must be non-negative");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    fn new_f64(
        window_len: usize,
        order: usize,
        delay: usize,
    ) -> PermutationEntropy<f64, Echo<f64>> {
        PermutationEntropy::new(
            Echo::new(),
            NonZeroUsize::new(window_len).unwrap(),
            order,
            NonZeroUsize::new(delay).unwrap(),
        )
    }

    /// Reference: stable argsort of each embedding vector, counted in a hash map.
    fn naive(vals: &[f64], window_len: usize, order: usize, delay: usize) -> f64 {
        let span = (order - 1) * delay + 1;
        let start = vals.len() + 1 - span - window_len;
        let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
        for t in start..start + window_len {
            let emb: Vec<f64> = (0..order).map(|i| vals[t + i * delay]).collect();
            let mut perm: Vec<usize> = (0..order).collect();
            perm.sort_by(|&a, &b| emb[a].partial_cmp(&emb[b]).unwrap());
            *counts.entry(perm).or_default() += 1;
        }
        let n = window_len as f64;
        let h: f64 = counts
            .values()
            .map(|&c| -(c as f64 / n) * (c as f64 / n).ln())
            .sum();
        let fact: usize = (1..=order).product();
        h / (fact as f64).ln()
    }

    #[test]
    fn permutation_entropy_warmup() {
        // span = 2 * 2 + 1 = 5, so the first output needs 5 - 1 + 4 = 8 values.
        let mut pe = new_f64(4, 3, 2);
        for i in 0..7 {
            pe.update(i as f64);
            assert_eq!(pe.last(), None);
        }
        pe.update(7.0);
        assert!(pe.last().is_some());
    }

    #[test]
    fn permutation_entropy_matches_naive() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        for (window_len, order, delay) in [(40, 3, 2), (100, 4, 1), (30, 2, 3), (300, 5, 1)] {
            let mut pe = new_f64(window_len, order, delay);
            let span = (order - 1) * delay + 1;
            let mut vals = Vec::new();
            for _ in 0..1_500 {
                // Quantized values produce ties, which must follow the same convention.
                let v = (rng.f64() * 8.0).floor();
                vals.push(v);
                pe.update(v);
                if vals.len() < span + window_len - 1 {
                    assert!(pe.last().is_none());
                    continue;
                }
                let expected = naive(&vals, window_len, order, delay);
                let got = pe.last().unwrap();
                assert!((got - expected).abs() < 1e-9, "{got} vs {expected}");
            }
        }
    }

    #[test]
    fn permutation_entropy_monotone_is_zero() {
        let mut up = new_f64(50, 4, 2);
        let mut down = new_f64(50, 4, 2);
        let mut flat = new_f64(50, 4, 2);
        for i in 0..200 {
            up.update(i as f64);
            down.update(-(i as f64));
            flat.update(1.0);
        }
        assert_eq!(up.last().unwrap(), 0.0);
        assert_eq!(down.last().unwrap(), 0.0);
        assert_eq!(flat.last().unwrap(), 0.0);
    }

    #[test]
    fn permutation_entropy_noise_near_max() {
        let rng = romu::Rng::from_seed_with_64bit(5);
        let mut pe = new_f64(20_000, 3, 1);
        for _ in 0..30_000 {
            pe.update(rng.f64());
        }
        let h = pe.last().unwrap();
        assert!(h > 0.99, "{h}");
    }

    #[test]
    fn permutation_entropy_pattern_index_ties() {
        let idx = |v: &[f64]| pattern_index(&VecDeque::from(v.to_vec()), v.len(), 1);
        assert_eq!(idx(&[0.0, 1.0, 2.0]), 0);
        assert_eq!(idx(&[2.0, 1.0, 0.0]), 5);
        // Ties rank by time, so equal values look increasing.
        assert_eq!(idx(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(idx(&[1.0, 1.0, 0.0]), idx(&[1.0, 2.0, 0.0]));
        // All 3! patterns map to distinct indices.
        let perms = [
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ];
        let mut seen: Vec<usize> = perms.iter().map(|p| idx(p)).collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "order must be at least 2")]
    fn permutation_entropy_order_too_small() {
        let _ = new_f64(10, 1, 1);
    }

    #[test]
    #[should_panic(expected = "order must be at most 7")]
    fn permutation_entropy_order_too_large() {
        let _ = new_f64(10, 8, 1);
    }

    #[test]
    fn permutation_entropy_plot() {
        let mut pe = new_f64(32, 3, 1);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            pe.update(*v);
            if let Some(val) = pe.last() {
                out.push(val);
            }
        }
        plot_values(out, "img/permutation_entropy.png").unwrap();
    }
}

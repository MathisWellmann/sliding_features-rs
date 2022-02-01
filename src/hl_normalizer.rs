//! A sliding High - Low Normalizer

use std::collections::VecDeque;

use super::View;

/// A sliding High - Low Normalizer
#[derive(Clone)]
pub struct HLNormalizer<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    min: f64,
    max: f64,
    last: f64,
    init: bool,
}

impl<V> std::fmt::Debug for HLNormalizer<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "HLNormalizer(window_len: {}, q_vals: {:?}, min: {}, max: {}, last: {}, init: {})",
            self.window_len, self.q_vals, self.min, self.max, self.last, self.init
        )
    }
}

impl<V> HLNormalizer<V>
where
    V: View,
{
    /// Create a new HLNormalizer with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        HLNormalizer {
            view,
            window_len,
            q_vals: VecDeque::new(),
            min: 0.0,
            max: 0.0,
            last: 0.0,
            init: true,
        }
    }
}

fn extent_queue(q: &VecDeque<f64>) -> (f64, f64) {
    let mut min: &f64 = q.front().unwrap();
    let mut max: &f64 = q.front().unwrap();

    for i in 0..q.len() {
        let val = q.get(i).unwrap();
        if val > max {
            max = val;
        }
        if val < min {
            min = val;
        }
    }

    (*min, *max)
}

impl<V> View for HLNormalizer<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let view_last = self.view.last();

        if self.init {
            self.init = false;
            self.min = view_last;
            self.max = view_last;
            self.last = view_last;
        }
        if self.q_vals.len() >= self.window_len {
            let old = *self.q_vals.front().unwrap();
            if old <= self.min || old >= self.max {
                let (min, max) = extent_queue(&self.q_vals);
                self.min = min;
                self.max = max;
            }
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(view_last);
        if view_last > self.max {
            self.max = view_last;
        }
        if view_last < self.min {
            self.min = view_last;
        }
        self.last = view_last;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        if self.last == self.min && self.last == self.max {
            0.0
        } else {
            -1.0 + (((self.last - self.min) * 2.0) / (self.max - self.min))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_data::TEST_DATA, Echo};

    #[test]
    fn normalizer() {
        let mut n = HLNormalizer::new(Echo::new(), 16);
        for v in &TEST_DATA {
            n.update(*v);
            let last = n.last();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }
}

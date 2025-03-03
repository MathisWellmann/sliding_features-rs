//! A sliding High - Low Normalizer

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// A sliding High - Low Normalizer
#[derive(Clone, Debug, CopyGetters)]
pub struct HLNormalizer<T, V> {
    view: V,
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: usize,
    q_vals: VecDeque<T>,
    min: T,
    max: T,
    last: T,
    init: bool,
}

impl<T, V> HLNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new HLNormalizer with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        HLNormalizer {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len),
            min: T::zero(),
            max: T::zero(),
            last: T::zero(),
            init: true,
        }
    }
}

fn extent_queue<T: Float>(q: &VecDeque<T>) -> (T, T) {
    let mut min = *q.front().unwrap();
    let mut max = *q.front().unwrap();

    for i in 0..q.len() {
        let val = *q.get(i).unwrap();
        if val > max {
            max = val;
        }
        if val < min {
            min = val;
        }
    }

    (min, max)
}

impl<T, V> View<T> for HLNormalizer<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(view_last) = self.view.last() else {
            return;
        };

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

    fn last(&self) -> Option<T> {
        if self.last == self.min && self.last == self.max {
            Some(T::zero())
        } else {
            Some(
                -T::one()
                    + (((self.last - self.min) * T::from(2.0).expect("can convert"))
                        / (self.max - self.min)),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pure_functions::Echo, test_data::TEST_DATA};

    #[test]
    fn normalizer() {
        let mut n = HLNormalizer::new(Echo::new(), 16);
        for v in &TEST_DATA {
            n.update(*v);
            let last = n.last().unwrap();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }
}

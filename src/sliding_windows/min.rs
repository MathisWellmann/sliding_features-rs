use std::collections::VecDeque;

use num::Float;

use crate::View;

/// Keep track of the minimum value observed over the sliding window.
#[derive(Clone, Debug)]
pub struct Min<T, V> {
    view: V,
    opt_min: Option<T>,
    q_vals: VecDeque<T>,
}

impl<T, V> Min<T, V>
where
    T: Float,
    V: View<T>,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    pub fn new(view: V, window_len: usize) -> Self {
        assert!(window_len > 0, "Window length must be greater than zero");
        Self {
            view,
            opt_min: None,
            q_vals: VecDeque::with_capacity(window_len),
        }
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> usize {
        self.q_vals.capacity()
    }
}

impl<T, V> View<T> for Min<T, V>
where
    T: Float,
    V: View<T>,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if let Some(min) = self.opt_min.as_mut() {
            if val < *min {
                *min = val;
            }
        } else {
            self.opt_min = Some(val);
        }
        if self.q_vals.len() >= self.q_vals.capacity() {
            let popped = self.q_vals.pop_front().expect("There is a value");
            if popped == self.opt_min.expect("Has a minimum value") {
                // re-compute the min value.
                self.opt_min = self
                    .q_vals
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).expect("Can compare elements"));
            }
        }
        self.q_vals.push_back(val);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.opt_min
    }
}

#[cfg(test)]
mod test {
    use crate::pure_functions::Echo;

    use super::*;

    #[test]
    fn min() {
        let mut v = Min::new(Echo::new(), 3);
        assert_eq!(v.last(), None);
        v.update(1.0);
        assert_eq!(v.last(), Some(1.0));
        v.update(2.0);
        assert_eq!(v.last(), Some(1.0));
        v.update(0.5);
        assert_eq!(v.last(), Some(0.5));
        v.update(1.1);
        assert_eq!(v.last(), Some(0.5));
        v.update(1.2);
        assert_eq!(v.last(), Some(0.5));
        v.update(1.3);
        assert_eq!(v.last(), Some(1.1));
        v.update(1.4);
        assert_eq!(v.last(), Some(1.2));
    }
}

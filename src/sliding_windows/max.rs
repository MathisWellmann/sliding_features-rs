use std::collections::VecDeque;

use getset::CopyGetters;
use num::Float;

use crate::View;

/// Keep track of the maximum value observed over the sliding window.
#[derive(Clone, Debug, CopyGetters)]
pub struct Max<T, V> {
    view: V,
    opt_max: Option<T>,
    q_vals: VecDeque<T>,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
}

impl<T, V> Max<T, V>
where
    T: Float,
    V: View<T>,
{
    /// Create a new instance with a chained `View` and a sliding window length.
    pub fn new(view: V, window_len: usize) -> Self {
        assert!(window_len > 0, "Window length must be greater than zero");
        Self {
            view,
            opt_max: None,
            q_vals: VecDeque::with_capacity(window_len),
            window_len,
        }
    }
}

impl<T, V> View<T> for Max<T, V>
where
    T: Float,
    V: View<T>,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len {
            let popped = self.q_vals.pop_front().expect("There is a value");
            if popped == self.opt_max.expect("Has a minimum value") {
                // re-compute the max value.
                self.opt_max = self
                    .q_vals
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).expect("Can compare elements"));
            }
        }
        self.q_vals.push_back(val);
        if let Some(max) = self.opt_max.as_mut() {
            if val > *max {
                *max = val;
            }
        } else {
            self.opt_max = Some(val);
        }
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.opt_max
    }
}

#[cfg(test)]
mod test {
    use crate::pure_functions::Echo;

    use super::*;

    #[test]
    fn max() {
        const WINDOW_LEN: usize = 3;
        let mut v = Max::new(Echo::new(), 3);
        assert_eq!(v.last(), None);
        v.update(1.0);
        assert_eq!(v.window_len(), WINDOW_LEN);
        assert_eq!(v.last(), Some(1.0));
        v.update(2.0);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), WINDOW_LEN);
        v.update(0.5);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), WINDOW_LEN);
        v.update(1.1);
        assert_eq!(v.last(), Some(2.0));
        assert_eq!(v.window_len(), WINDOW_LEN);
        v.update(1.2);
        assert_eq!(v.last(), Some(1.2));
        assert_eq!(v.window_len(), WINDOW_LEN);
        v.update(1.3);
        assert_eq!(v.last(), Some(1.3));
        assert_eq!(v.window_len(), WINDOW_LEN);
        v.update(1.4);
        assert_eq!(v.last(), Some(1.4));
        assert_eq!(v.window_len(), WINDOW_LEN);
    }
}

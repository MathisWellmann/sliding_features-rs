//! Shannon entropy sliding window over values,
//! where a positive / negative values are interpreted as true / false

use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

use crate::View;

#[derive(Debug, Clone)]
/// Shannon entropy sliding window over values,
/// where a positive / negative values are interpreted as true / false
pub struct BinaryEntropy<T, V> {
    view: V,
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    // number of positive values
    p: usize,
}

impl<T, V> BinaryEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Entropy Sliding Window
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            p: 0,
        }
    }
}

impl<T, V> View<T> for BinaryEntropy<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Update the Entropy calculation with a new boolean value
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len.get() {
            let old_val = self.q_vals.pop_back().unwrap();
            if old_val >= T::zero() {
                self.p -= 1;
            }
        }
        if val >= T::zero() {
            self.p += 1;
        }
        self.q_vals.push_front(val);
    }

    /// Get the latest entropy value of the sliding window
    fn last(&self) -> Option<T> {
        if self.q_vals.is_empty() {
            return None;
        }
        let pt = T::from(self.p).expect("can convert")
            / T::from(self.q_vals.len()).expect("can convert"); // probability of positive value
        let pn = T::one() - pt; // probability of negative value

        let mut value = pt * pt.log2() + pn * pn.log2();
        if value.is_nan() {
            value = T::zero()
        }
        Some(-value)
    }
}

#[cfg(test)]
mod tests {
    use crate::pure_functions::Echo;

    use super::*;

    #[test]
    fn binary_entropy() {
        let vals: Vec<f64> = vec![1.0; 10];
        let mut e = BinaryEntropy::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for v in &vals {
            e.update(*v);
            let last = e.last().unwrap();
            assert_eq!(last, 0.0);
        }
        let vals: Vec<f64> = vec![1.0; 10];
        let mut e = BinaryEntropy::new(Echo::new(), NonZeroUsize::new(10).unwrap());
        for v in &vals {
            e.update(*v);
            let last = e.last().unwrap();
            assert_eq!(last, 0.0);
        }

        let vals: Vec<f64> = vec![-1.0, 1.0, -1.0, 1.0];
        let mut e = BinaryEntropy::new(Echo::new(), NonZeroUsize::new(4).unwrap());
        for v in &vals {
            e.update(*v);
            let last = e.last().unwrap();
            println!("last: {}", last);
            assert!(last >= 0.0);
        }
        let last = e.last().unwrap();
        assert_eq!(last, 1.0);
    }
}

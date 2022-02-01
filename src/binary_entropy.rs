//! Shannon entropy sliding window over values,
//! where a positive / negative values are interpreted as true / false

use std::collections::VecDeque;

use crate::View;

#[derive(Debug, Clone)]
/// Shannon entropy sliding window over values,
/// where a positive / negative values are interpreted as true / false
pub struct BinaryEntropy<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    // number of positive values
    p: usize,
}

impl<V> BinaryEntropy<V>
where
    V: View,
{
    /// Create a new Entropy Sliding Window
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            p: 0,
        }
    }
}

impl<V> View for BinaryEntropy<V>
where
    V: View,
{
    /// Update the Entropy calculation with a new boolean value
    #[inline]
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() >= self.window_len {
            let old_val = self.q_vals.pop_back().unwrap();
            if old_val >= 0.0 {
                self.p -= 1;
            }
        }
        if val >= 0.0 {
            self.p += 1;
        }
        self.q_vals.push_front(val);
    }

    /// Get the latest entropy value of the sliding window
    fn last(&self) -> f64 {
        let pt: f64 = self.p as f64 / self.q_vals.len() as f64; // probability of positive value
        let pn: f64 = 1.0 - pt; // probability of negative value

        let mut value = pt * pt.log2() + pn * pn.log2();
        if value.is_nan() {
            value = 0.0
        }
        -value
    }
}

#[cfg(test)]
mod tests {
    use crate::{plot::plot_values, test_data::TEST_DATA, Echo};

    use super::*;

    #[test]
    fn binary_entropy() {
        let vals: Vec<f64> = vec![1.0; 10];
        let mut e = BinaryEntropy::new(Echo::new(), 10);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            assert_eq!(last, 0.0);
        }
        let vals: Vec<f64> = vec![1.0; 10];
        let mut e = BinaryEntropy::new(Echo::new(), 10);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            assert_eq!(last, 0.0);
        }

        let vals: Vec<f64> = vec![-1.0, 1.0, -1.0, 1.0];
        let mut e = BinaryEntropy::new(Echo::new(), 4);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            println!("last: {}", last);
            assert!(last >= 0.0);
        }
        let last = e.last();
        assert_eq!(last, 1.0);
    }
}

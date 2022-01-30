//! Shannon Entropy over bool values over a sliding window

use std::collections::VecDeque;

#[derive(Debug, Clone)]
/// Shannon Entropy over bool values over a sliding window
pub struct Entropy {
    value: f64,
    window_len: usize,
    q_vals: VecDeque<bool>,
}

impl Entropy {
    /// Create a new Entropy Sliding Window
    #[inline(always)]
    pub fn new(window_len: usize) -> Self {
        Self {
            value: 0.0,
            window_len,
            q_vals: VecDeque::new(),
        }
    }

    /// Update the Entropy calculation with a new boolean value
    #[inline]
    pub fn update(&mut self, val: bool) {
        if self.q_vals.len() >= self.window_len {
            let _ = self.q_vals.pop_back().unwrap();
        }
        self.q_vals.push_front(val);
    }

    /// Get the latest Entropy
    pub fn last(&self) -> f64 {
        // count of all values
        let c: f64 = self.q_vals.len() as f64;
        // probability of true value
        let s: f64 = self
            .q_vals
            .iter()
            .map(|v| if *v { 1.0 } else { 0.0 })
            .sum::<f64>();
        let pt: f64 = s / c; // probability of true value
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
    use super::*;

    #[test]
    fn entropy() {
        let vals: Vec<bool> = vec![false; 10];
        let mut e = Entropy::new(10);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            assert_eq!(last, 0.0);
        }
        let vals: Vec<bool> = vec![true; 10];
        let mut e = Entropy::new(10);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            assert_eq!(last, 0.0);
        }

        let vals: Vec<bool> = vec![false, true, false, true];
        let mut e = Entropy::new(4);
        for v in &vals {
            e.update(*v);
            let last = e.last();
            println!("last: {}", last);
        }
        let last = e.last();
        assert_eq!(last, 1.0);
    }
}

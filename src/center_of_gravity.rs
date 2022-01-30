//! John Ehlers Center of Gravity Indicator
//! from: https://mesasoftware.com/papers/TheCGOscillator.pdf

use std::collections::VecDeque;

use super::View;
use crate::Echo;

/// John Ehlers Center of Gravity Indicator
/// from: https://mesasoftware.com/papers/TheCGOscillator.pdf
#[derive(Clone)]
pub struct CenterOfGravity<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V> std::fmt::Debug for CenterOfGravity<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "CenterOfGravity(window_len: {}, q_vals: {:?}, out: {})",
            self.window_len, self.q_vals, self.out
        )
    }
}

/// Create a Center of Gravity Indicator with a given window length
#[inline(always)]
pub fn new_final(window_len: usize) -> CenterOfGravity<Echo> {
    CenterOfGravity::new(Echo::new(), window_len)
}

impl<V> CenterOfGravity<V>
where
    V: View,
{
    /// Create a Center of Gravity Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl<V> View for CenterOfGravity<V>
where
    V: View,
{
    // update receives a new value and updates its internal state
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let mut denom: f64 = 0.0;
        let mut num: f64 = 0.0;
        let q_len = self.q_vals.len();
        for i in 0..q_len {
            let weight = q_len - i;
            let val_i = self.q_vals.get(i).unwrap();
            num += weight as f64 * val_i;
            denom += *val_i;
        }
        if denom != 0.0 {
            self.out = -num / denom + (self.q_vals.len() as f64 + 1.0) / 2.0
        } else {
            self.out = 0.0;
        }
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn center_of_gravity_plot() {
        let mut cgo = new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cgo.update(*v);
            out.push(cgo.last());
        }
        let filename = "img/center_of_gravity.png";
        plot_values(out, filename).unwrap();
    }
}

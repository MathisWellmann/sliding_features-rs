//! John Ehlers Center of Gravity Indicator
//! from: <https://mesasoftware.com/papers/TheCGOscillator.pdf>

use std::collections::VecDeque;

use crate::View;

/// John Ehlers Center of Gravity Indicator
/// from: <https://mesasoftware.com/papers/TheCGOscillator.pdf>
#[derive(Clone)]
pub struct CenterOfGravity<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: Option<f64>,
}

impl<V> std::fmt::Debug for CenterOfGravity<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "CenterOfGravity(window_len: {}, q_vals: {:?}, out: {:?})",
            self.window_len, self.q_vals, self.out
        )
    }
}

impl<V> CenterOfGravity<V>
where
    V: View,
{
    /// Create a Center of Gravity Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
            out: None,
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
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let mut denom: f64 = 0.0;
        let mut num: f64 = 0.0;
        let q_len = self.q_vals.len();
        for (i, val) in self.q_vals.iter().enumerate() {
            let weight = q_len - i;
            num += weight as f64 * val;
            denom += *val;
        }
        if denom != 0.0 {
            self.out = Some(-num / denom + (q_len as f64 + 1.0) / 2.0)
        } else {
            self.out = Some(0.0);
        }
    }

    fn last(&self) -> Option<f64> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn center_of_gravity_plot() {
        let mut cgo = CenterOfGravity::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cgo.update(*v);
            out.push(cgo.last().unwrap());
        }
        let filename = "img/center_of_gravity.png";
        plot_values(out, filename).unwrap();
    }
}

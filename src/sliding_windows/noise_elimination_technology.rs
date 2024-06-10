//! John Ehlers Noise elimination technology using kendall correlation
//! from <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>

use crate::View;
use std::collections::VecDeque;

/// John Ehlers Noise elimination technology using kendall correlation
/// from <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>
#[derive(Debug, Clone)]
pub struct NET<V> {
    view: V,
    window_len: usize,
    out: Option<f64>,
    q_vals: VecDeque<f64>,
}

impl<V> NET<V>
where
    V: View,
{
    /// Create a new NET with a chained View and window length
    pub fn new(view: V, window_len: usize) -> Self {
        NET {
            view,
            window_len,
            out: None,
            q_vals: VecDeque::new(),
        }
    }
}

impl<V> View for NET<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() < 2 {
            return;
        }
        let mut x: Vec<f64> = vec![0.0; self.q_vals.len()];
        let mut y: Vec<f64> = vec![0.0; self.q_vals.len()];
        for count in 1..self.q_vals.len() {
            x[count] = *self.q_vals.get(self.q_vals.len() - count).unwrap();
            y[count] = -(count as f64);
        }

        let mut num: f64 = 0.0;
        for count in 2..self.q_vals.len() {
            for k in 1..count - 1 {
                num -= (x[count] - x[k]).signum();
            }
        }

        let denom: f64 = 0.5 * self.q_vals.len() as f64 * (self.q_vals.len() as f64 - 1.0);
        self.out = Some(num / denom);
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
    use crate::sliding_windows::MyRSI;
    use crate::test_data::TEST_DATA;

    #[test]
    fn net_my_rsi_plot() {
        let mut net = NET::new(MyRSI::new(Echo::new(), 16), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            net.update(*v);
            if let Some(val) = net.last() {
                out.push(val);
            }
        }
        println!("out: {:?}", out);

        let filename = "img/net_my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

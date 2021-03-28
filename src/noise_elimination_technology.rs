use crate::{Echo, View};
use std::collections::VecDeque;

/// John Ehlers Noise elimination technology using kendall correlation
/// from http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf
#[derive(Clone)]
pub struct NET {
    view: Box<dyn View>,
    window_len: usize,
    out: f64,
    q_vals: VecDeque<f64>,
}

impl NET {
    /// Create a new NET with a chained View and window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Box<Self> {
        Box::new(NET {
            view,
            window_len,
            out: 0.0,
            q_vals: VecDeque::new(),
        })
    }

    /// Create a new NET with a window length
    pub fn new_final(window_len: usize) -> Box<Self> {
        Self::new(Echo::new(), window_len)
    }
}

impl View for NET {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        // if self.q_vals.len() == self.window_len {
        //
        // } else {
        //     self.out = val;
        // }
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
        self.out = num / denom;
    }

    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::MyRSI;

    #[test]
    fn net_my_rsi_plot() {
        let mut net = NET::new(MyRSI::new_final(16), 16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            net.update(*v);
            out.push(net.last());
        }
        println!("out: {:?}", out);

        let filename = "img/net_my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

use std::collections::VecDeque;

use super::sliding_window::View;
use crate::Echo;

/// John Ehlers Cyber Cycle Indicator
/// from: https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf
#[derive(Clone)]
pub struct CyberCycle {
    view: Box<dyn View>,
    window_len: usize,
    alpha: f64,
    vals: VecDeque<f64>,
    out: VecDeque<f64>,
}

impl CyberCycle {
    /// Create a new Cyber Cycle Indicator with a chained View
    /// and a given window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Box<Self> {
        Box::new(CyberCycle {
            view,
            window_len,
            alpha: 2.0 / (window_len as f64 + 1.0),
            vals: VecDeque::new(),
            out: VecDeque::new(),
        })
    }

    /// Create a new Cyber Cycle Indicator with a given window length
    pub fn new_final(window_len: usize) -> Box<Self> {
        Self::new(Echo::new(), window_len)
    }
}

impl View for CyberCycle {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.vals.len() >= self.window_len {
            self.vals.pop_front();
            self.out.pop_front();
        }
        self.vals.push_back(val);

        if self.vals.len() < self.window_len {
            self.out.push_back(0.0);
            return;
        }
        let mut smooth: Vec<f64> = vec![0.0; self.vals.len()];
        let last = self.vals.len() - 1;
        for i in 3..self.vals.len() {
            smooth[i] = (val
                + 2.0 * self.vals.get(i - 1).unwrap()
                + 2.0 * self.vals.get(i - 2).unwrap()
                + *self.vals.get(i - 3).unwrap())
                / 6.0
        }
        let cc = (1.0 - 0.5 * self.alpha).powi(2)
            * (smooth[last] - 2.0 * smooth[last - 1] + smooth[last - 2])
            + 2.0 * (1.0 - self.alpha) * self.out.get(last - 1).unwrap()
            - (1.0 - self.alpha).powi(2) * self.out.get(last - 2).unwrap();
        self.out.push_back(cc);
    }
    fn last(&self) -> f64 {
        return *self.out.get(self.out.len() - 1).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn cyber_cycle_plot() {
        let mut cc = CyberCycle::new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cc.update(*v);
            out.push(cc.last());
        }
        let filename = "img/cyber_cycle.png";
        plot_values(out, filename).unwrap();
    }
}

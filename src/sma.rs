use std::collections::VecDeque;

use crate::sliding_window::View;
use crate::Echo;

#[derive(Clone)]
/// SMA - Simple Moving Average
pub struct SMA {
    view: Box<dyn View>,
    window_len: usize,
    q_vals: VecDeque<f64>,
    sum: f64,
}

impl SMA {
    /// Create a new simple moving average with a chained View
    /// and a given sliding window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        SMA {
            view,
            window_len,
            q_vals: VecDeque::new(),
            sum: 0.0,
        }
    }

    /// Create a new simple moving average with a given window length
    pub fn new_final(window_len: usize) -> Self {
        Self::new(Box::new(Echo::new()), window_len)
    }
}

impl View for SMA {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() > self.window_len {
            let old_val = self.q_vals.pop_front().unwrap();
            self.sum -= old_val;
        }
        self.q_vals.push_back(val);

        self.sum += val;
    }

    fn last(&self) -> f64 {
        self.sum / self.q_vals.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use rand::{thread_rng, Rng};

    #[test]
    fn sma() {
        let mut rng = thread_rng();

        let mut sma = SMA::new_final(16);
        for _ in 0..1024 {
            let r = rng.gen::<f64>();
            sma.update(r);
            let last = sma.last();
            assert!(last >= 0.0);
            assert!(last <= 1.0);
        }
    }

    #[test]
    fn sma_plot() {
        let mut sma = SMA::new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            sma.update(*v);
            out.push(sma.last());
        }
        let filename = "img/sma.png";
        plot_values(out, filename).unwrap();
    }
}

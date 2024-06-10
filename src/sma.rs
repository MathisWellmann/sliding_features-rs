//! SMA - Simple Moving Average

use std::collections::VecDeque;

use crate::View;

#[derive(Debug, Clone)]
/// SMA - Simple Moving Average
pub struct SMA<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    sum: f64,
}

impl<V> SMA<V>
where
    V: View,
{
    /// Create a new simple moving average with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        SMA {
            view,
            window_len,
            q_vals: VecDeque::new(),
            sum: 0.0,
        }
    }
}

impl<V> View for SMA<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() > self.window_len {
            let old_val = self.q_vals.pop_front().unwrap();
            self.sum -= old_val;
        }
        self.q_vals.push_back(val);

        self.sum += val;
    }

    fn last(&self) -> Option<f64> {
        if self.q_vals.len() < self.window_len {
            return None;
        }
        Some(self.sum / self.q_vals.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::{plot::plot_values, Echo};
    use rand::{thread_rng, Rng};

    #[test]
    fn sma() {
        let mut rng = thread_rng();

        let mut sma = SMA::new(Echo::new(), 16);
        for _ in 0..1024 {
            let r = rng.gen::<f64>();
            sma.update(r);
            if let Some(last) = sma.last() {
                assert!(last >= 0.0);
                assert!(last <= 1.0);
            }
        }
    }

    #[test]
    fn sma_plot() {
        let mut sma = SMA::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            sma.update(*v);
            if let Some(val) = sma.last() {
                out.push(val);
            }
        }
        let filename = "img/sma.png";
        plot_values(out, filename).unwrap();
    }
}

//! SMA - Simple Moving Average

use std::collections::VecDeque;

use crate::Echo;
use crate::View;

#[derive(Clone)]
/// SMA - Simple Moving Average
pub struct SMA<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
    sum: f64,
}

impl<V> std::fmt::Debug for SMA<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "SMA(window_len: {}, q_vals: {:?}, sum: {})",
            self.window_len, self.q_vals, self.sum
        )
    }
}

/// Create a new simple moving average with a given window length
#[inline(always)]
pub fn new_final(window_len: usize) -> SMA<Echo> {
    SMA::new(Echo::new(), window_len)
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
        let val = self.view.last();

        if self.q_vals.len() > self.window_len {
            let old_val = self.q_vals.pop_front().unwrap();
            self.sum -= old_val;
        }
        self.q_vals.push_back(val);

        self.sum += val;
    }

    #[inline(always)]
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

        let mut sma = new_final(16);
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
        let mut sma = new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            sma.update(*v);
            out.push(sma.last());
        }
        let filename = "img/sma.png";
        plot_values(out, filename).unwrap();
    }
}

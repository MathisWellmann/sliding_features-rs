//! John Ehlers Cyber Cycle Indicator
//! from: https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf

use std::collections::VecDeque;

use super::View;
use crate::Echo;

/// John Ehlers Cyber Cycle Indicator
/// from: https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf
#[derive(Clone)]
pub struct CyberCycle<V> {
    view: V,
    window_len: usize,
    alpha: f64,
    vals: VecDeque<f64>,
    out: VecDeque<f64>,
}

impl<V> std::fmt::Debug for CyberCycle<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "CyberCycle(window_len: {}, alpha: {}, vals: {:?}, out: {:?})",
            self.window_len, self.alpha, self.vals, self.out
        )
    }
}

/// Create a new Cyber Cycle Indicator with a given window length
#[inline(always)]
pub fn new_final(window_len: usize) -> CyberCycle<Echo> {
    CyberCycle::new(Echo::new(), window_len)
}

impl<V> CyberCycle<V>
where
    V: View,
{
    /// Create a new Cyber Cycle Indicator with a chained View
    /// and a given window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        CyberCycle {
            view,
            window_len,
            alpha: 2.0 / (window_len as f64 + 1.0),
            vals: VecDeque::new(),
            out: VecDeque::new(),
        }
    }
}

impl<V> View for CyberCycle<V>
where
    V: View,
{
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

    #[inline(always)]
    fn last(&self) -> f64 {
        *self.out.get(self.out.len() - 1).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn cyber_cycle_plot() {
        let mut cc = new_final(16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cc.update(*v);
            out.push(cc.last());
        }
        let filename = "img/cyber_cycle.png";
        plot_values(out, filename).unwrap();
    }
}

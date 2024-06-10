//! John Ehlers LaguerreRSI
//! from: <http://mesasoftware.com/papers/TimeWarp.pdf>

use std::collections::VecDeque;

use super::View;

/// John Ehlers LaguerreRSI
/// from: <http://mesasoftware.com/papers/TimeWarp.pdf>
#[derive(Debug, Clone)]
pub struct LaguerreRSI<V> {
    view: V,
    value: Option<f64>,
    gamma: f64,
    l0s: VecDeque<f64>,
    l1s: VecDeque<f64>,
    l2s: VecDeque<f64>,
    l3s: VecDeque<f64>,
}

impl<V> LaguerreRSI<V>
where
    V: View,
{
    /// Create a new LaguerreRSI with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        LaguerreRSI {
            view,
            value: None,
            gamma: 2.0 / (window_len as f64 + 1.0),
            l0s: VecDeque::new(),
            l1s: VecDeque::new(),
            l2s: VecDeque::new(),
            l3s: VecDeque::new(),
        }
    }
}

impl<V> View for LaguerreRSI<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.l0s.len() >= 3 {
            self.l0s.pop_front();
            self.l1s.pop_front();
            self.l2s.pop_front();
            self.l3s.pop_front();
        }

        if self.l0s.len() < 2 {
            self.l0s.push_back(0.0);
            self.l1s.push_back(0.0);
            self.l2s.push_back(0.0);
            self.l3s.push_back(0.0);
            return;
        } else {
            let last = self.l0s.len() - 1;
            self.l0s
                .push_back((1.0 - self.gamma) * val + self.gamma * self.l0s.get(last - 1).unwrap());
            self.l1s.push_back(
                -self.gamma * self.l0s.get(last).unwrap()
                    + self.l0s.get(last - 1).unwrap()
                    + self.gamma * self.l1s.get(last - 1).unwrap(),
            );
            self.l2s.push_back(
                -self.gamma * self.l1s.get(last).unwrap()
                    + self.l1s.get(last - 1).unwrap()
                    + self.gamma * self.l2s.get(last - 1).unwrap(),
            );
            self.l3s.push_back(
                -self.gamma * self.l2s.get(last).unwrap()
                    + self.l2s.get(last - 1).unwrap()
                    + self.gamma * self.l3s.get(last - 1).unwrap(),
            );
        }
        let last = self.l0s.len() - 1;

        let mut cu: f64 = 0.0;
        let mut cd: f64 = 0.0;
        if self.l0s.get(last) >= self.l1s.get(last) {
            cu = self.l0s.get(last).unwrap() - self.l1s.get(last).unwrap();
        } else {
            cd = self.l1s.get(last).unwrap() - self.l0s.get(last).unwrap();
        }
        if self.l1s.get(last) >= self.l2s.get(last) {
            cu += self.l1s.get(last).unwrap() - self.l2s.get(last).unwrap();
        } else {
            cd += self.l2s.get(last).unwrap() - self.l1s.get(last).unwrap();
        }
        if self.l2s.get(last) >= self.l3s.get(last) {
            cu += self.l2s.get(last).unwrap() - self.l3s.get(last).unwrap();
        } else {
            cd += self.l3s.get(last).unwrap() - self.l2s.get(last).unwrap();
        }

        if cu + cd != 0.0 {
            self.value = Some(cu / (cu + cd));
        }
    }

    fn last(&self) -> Option<f64> {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::Echo;

    #[test]
    fn laguerre_rsi() {
        let mut lrsi = LaguerreRSI::new(Echo::new(), 16);
        for v in &TEST_DATA {
            lrsi.update(*v);
            if let Some(last) = lrsi.last() {
                assert!(last <= 1.0);
                assert!(last >= -1.0);
            }
        }
    }

    #[test]
    fn laguerre_rsi_plot() {
        let mut lrsi = LaguerreRSI::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            lrsi.update(*v);
            if let Some(val) = lrsi.last() {
                out.push(val);
            }
        }
        // graph the results
        let filename = "img/laguerre_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

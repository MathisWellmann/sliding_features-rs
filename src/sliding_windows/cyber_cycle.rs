//! John Ehlers Cyber Cycle Indicator
//! from: <https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf>

use std::collections::VecDeque;

use crate::View;

/// John Ehlers Cyber Cycle Indicator
/// from: <https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf>
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
        let Some(val) = self.view.last() else { return };

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
        for (i, v) in smooth.iter_mut().enumerate().take(self.vals.len()).skip(3) {
            *v = (val
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

    fn last(&self) -> Option<f64> {
        self.out.back().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn cyber_cycle_plot() {
        let mut cc = CyberCycle::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cc.update(*v);
            out.push(cc.last().unwrap());
        }
        let filename = "img/cyber_cycle.png";
        plot_values(out, filename).unwrap();
    }
}

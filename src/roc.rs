//! Rate of Change Indicator

use std::collections::VecDeque;

use super::View;

/// Rate of Change Indicator
#[derive(Clone)]
pub struct ROC<V> {
    view: V,
    window_len: usize,
    oldest: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V> std::fmt::Debug for ROC<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "ROC(window_len: {}, oldest: {}, q_vals: {:?}, out: {})",
            self.window_len, self.oldest, self.q_vals, self.out
        )
    }
}

impl<V> ROC<V>
where
    V: View,
{
    /// Create a new Rate of Change Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        ROC {
            view,
            window_len,
            oldest: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl<V> View for ROC<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.is_empty() {
            self.oldest = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.front().unwrap();
            self.oldest = *old;
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let roc = ((val - self.oldest) / self.oldest) * 100.0;
        self.out = roc;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::Echo;

    #[test]
    fn roc_plot() {
        let mut r = ROC::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            r.update(*v);
            out.push(r.last());
        }
        let filename = "img/roc.png";
        plot_values(out, filename).unwrap();
    }
}

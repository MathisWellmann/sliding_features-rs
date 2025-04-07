//! Relative Strength Index Indicator

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// Relative Strength Index Indicator
#[derive(Debug, Clone, CopyGetters)]
pub struct Rsi<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
    avg_gain: T,
    avg_loss: T,
    old_ref: T,
    last_val: T,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Rsi<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a Relative Strength Index Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Rsi {
            view,
            window_len,
            avg_gain: T::zero(),
            avg_loss: T::zero(),
            old_ref: T::zero(),
            last_val: T::zero(),
            q_vals: VecDeque::with_capacity(window_len),
            out: None,
        }
    }
}

impl<T, V> View<T> for Rsi<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.is_empty() {
            self.old_ref = val;
            self.last_val = val;
        }
        let window_len = T::from(self.window_len).expect("can convert");
        if self.q_vals.len() >= self.window_len {
            // remove old
            let old_val = *self.q_vals.front().unwrap();
            let change = old_val - self.old_ref;
            self.old_ref = old_val;
            self.q_vals.pop_front();
            if change > T::zero() {
                self.avg_gain = self.avg_gain - change / window_len;
            } else {
                self.avg_loss = self.avg_loss - change.abs() / window_len;
            }
        }
        self.q_vals.push_back(val);

        let change = val - self.last_val;
        self.last_val = val;
        if change > T::zero() {
            self.avg_gain = self.avg_gain + change / window_len;
        } else {
            self.avg_loss = self.avg_loss + change.abs() / window_len;
        }

        if self.q_vals.len() < self.window_len {
            return;
        }

        let hundred = T::from(100.0).expect("can convert");
        if self.avg_loss == T::zero() {
            self.out = Some(hundred);
        } else {
            let rs = self.avg_gain / self.avg_loss;
            let rsi = hundred - hundred / (T::one() + rs);
            debug_assert!(rsi.is_finite(), "value must be finite");
            self.out = Some(rsi);
        }
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn rsi_plot() {
        let mut rsi = Rsi::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rsi.update(*v);
            if let Some(val) = rsi.last() {
                out.push(val);
            }
        }
        let filename = "img/rsi.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn rsi_range() {
        let mut rsi = Rsi::new(Echo::new(), 16);
        for v in &TEST_DATA {
            rsi.update(*v);
            if let Some(last) = rsi.last() {
                assert!(last >= 0.0);
                assert!(last <= 100.0);
            }
        }
    }
}

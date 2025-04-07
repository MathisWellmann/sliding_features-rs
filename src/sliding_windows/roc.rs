//! Rate of Change Indicator

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// Rate of Change Indicator
#[derive(Debug, Clone, CopyGetters)]
pub struct Roc<T, V> {
    view: V,
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: usize,
    oldest: Option<T>,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> Roc<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Rate of Change Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        Roc {
            view,
            window_len,
            oldest: None,
            q_vals: VecDeque::with_capacity(window_len),
            out: None,
        }
    }
}

impl<T, V> View<T> for Roc<T, V>
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
            self.oldest = Some(val);
        }
        if self.q_vals.len() >= self.window_len {
            let old = self.q_vals.front().unwrap();
            self.oldest = Some(*old);
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let Some(oldest) = self.oldest else { return };
        if oldest == T::zero() {
            return;
        }
        let roc = ((val - oldest) / oldest) * T::from(100.0).expect("can convert");
        debug_assert!(roc.is_finite(), "value must be finite");
        self.out = Some(roc);
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
    fn roc_plot() {
        let mut r = Roc::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            r.update(*v);
            if let Some(val) = r.last() {
                out.push(val);
            }
        }
        let filename = "img/roc.png";
        plot_values(out, filename).unwrap();
    }
}

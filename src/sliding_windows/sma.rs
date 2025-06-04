//! SMA - Simple Moving Average

use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

use crate::View;

#[derive(Debug, Clone)]
/// SMA - Simple Moving Average
pub struct Sma<T, V> {
    view: V,
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    sum: T,
}

impl<T, V> Sma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new simple moving average with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Sma {
            view,
            window_len,
            q_vals: VecDeque::new(),
            sum: T::zero(),
        }
    }
}

impl<T, V> View<T> for Sma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() > self.window_len.get() {
            let old_val = self.q_vals.pop_front().unwrap();
            self.sum = self.sum - old_val;
        }
        self.q_vals.push_back(val);

        self.sum = self.sum + val;
    }

    fn last(&self) -> Option<T> {
        if self.q_vals.len() < self.window_len.get() {
            return None;
        }
        let sma = self.sum / T::from(self.q_vals.len()).expect("can convert");
        debug_assert!(sma.is_finite(), "value must be finite");
        Some(sma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::{plot::plot_values, pure_functions::Echo};
    use rand::{rng, Rng};

    #[test]
    fn sma() {
        let mut rng = rng();

        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for _ in 0..1024 {
            let r = rng.random::<f64>();
            sma.update(r);
            if let Some(last) = sma.last() {
                assert!(last >= 0.0);
                assert!(last <= 1.0);
            }
        }
    }

    #[test]
    fn sma_plot() {
        let mut sma = Sma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
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

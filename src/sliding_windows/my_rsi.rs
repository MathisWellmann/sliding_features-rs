//! John Ehlers MyRSI
//! from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>

use crate::View;
use getset::CopyGetters;
use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

/// John Ehlers MyRSI
/// from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>
#[derive(Debug, Clone, CopyGetters)]
pub struct MyRSI<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    cu: T,
    cd: T,
    out: T,
    q_vals: VecDeque<T>,
    last_val: T,
    oldest_val: T,
}

impl<T, V> MyRSI<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new MyRSI indicator with a chained View and a given window length
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        MyRSI {
            view,
            window_len,
            cu: T::zero(),
            cd: T::zero(),
            out: T::zero(),
            q_vals: VecDeque::with_capacity(window_len.get()),
            last_val: T::zero(),
            oldest_val: T::zero(),
        }
    }
}

impl<T, V> View<T> for MyRSI<T, V>
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
            self.oldest_val = val;
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len.get() {
            let old_val = self.q_vals.pop_front().unwrap();
            if old_val > self.oldest_val {
                self.cu = self.cu - (old_val - self.oldest_val);
            } else {
                self.cd = self.cd - (self.oldest_val - old_val);
            }
            self.oldest_val = old_val;
        }
        self.q_vals.push_back(val);

        // accumulate 'closes up' and 'closes down'
        if val > self.last_val {
            self.cu = self.cu + val - self.last_val;
        } else {
            self.cd = self.cd + self.last_val - val;
        }
        self.last_val = val;

        if self.cu + self.cd != T::zero() {
            self.out = (self.cu - self.cd) / (self.cu + self.cd);
        }
    }

    #[inline]
    fn last(&self) -> Option<T> {
        if self.q_vals.len() < self.window_len.get() {
            return None;
        }
        debug_assert!(self.out.is_finite(), "value must be finite");
        Some(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn my_rsi() {
        // TODO: don't be so lazy with this test.
        let mut my_rsi = MyRSI::<f64, _>::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for v in &TEST_DATA {
            my_rsi.update(*v);
            if let Some(val) = my_rsi.last() {
                dbg!(val);
                assert!(val <= 1.0);
                assert!(val >= -1.0);
            }
        }
    }

    #[test]
    fn my_rsi_plot() {
        let mut my_rsi = MyRSI::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            my_rsi.update(*v);
            if let Some(rsi) = my_rsi.last() {
                out.push(rsi);
            }
        }
        println!("out: {:?}", out);
        let filename = "img/my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

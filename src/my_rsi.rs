//! John Ehlers MyRSI
//! from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>

use crate::View;
use std::collections::VecDeque;

/// John Ehlers MyRSI
/// from: <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>
#[derive(Clone)]
pub struct MyRSI<V> {
    view: V,
    window_len: usize,
    cu: f64,
    cd: f64,
    out: f64,
    q_vals: VecDeque<f64>,
    last_val: f64,
    oldest_val: f64,
}

impl<V> std::fmt::Debug for MyRSI<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "MyRSI(window_len: {}, cu: {}, cd: {}, out: {}, q_vals: {:?}, last_val: {}, oldest_val: {})",
               self.window_len, self.cu, self.cd, self.out, self.q_vals, self.last_val, self.oldest_val)
    }
}

impl<V> MyRSI<V>
where
    V: View,
{
    /// Create a new MyRSI indicator with a chained View and a given window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        MyRSI {
            view,
            window_len,
            cu: 0.0,
            cd: 0.0,
            out: 0.0,
            q_vals: VecDeque::new(),
            last_val: 0.0,
            oldest_val: 0.0,
        }
    }
}

impl<V> View for MyRSI<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        if self.q_vals.is_empty() {
            self.oldest_val = val;
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old_val: f64 = self.q_vals.pop_front().unwrap();
            if old_val > self.oldest_val {
                self.cu -= old_val - self.oldest_val;
            } else {
                self.cd -= self.oldest_val - old_val;
            }
            self.oldest_val = old_val;
        }
        self.q_vals.push_back(val);

        // accumulate 'closes up' and 'closes down'
        if val > self.last_val {
            self.cu += val - self.last_val;
        } else {
            self.cd += self.last_val - val;
        }
        self.last_val = val;

        // only output value if window length is satisfied with enough values
        if self.cu + self.cd != 0.0 {
            self.out = (self.cu - self.cd) / (self.cu + self.cd);
        }
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
    fn my_rsi() {
        let mut my_rsi = MyRSI::new(Echo::new(), 16);
        for v in &TEST_DATA {
            my_rsi.update(*v);
            assert!(my_rsi.last() <= 1.0);
            assert!(my_rsi.last() >= -1.0);
        }
    }

    #[test]
    fn my_rsi_plot() {
        let mut my_rsi = MyRSI::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            my_rsi.update(*v);
            out.push(my_rsi.last());
        }
        println!("out: {:?}", out);
        let filename = "img/my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

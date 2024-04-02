//! John Ehlers ReFlex Indicator
//! from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>

use std::collections::VecDeque;

use super::View;

/// John Ehlers ReFlex Indicator
/// from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>
#[derive(Clone)]
pub struct ReFlex<V> {
    view: V,
    window_len: usize,
    last_val: f64,
    last_m: f64,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V> std::fmt::Debug for ReFlex<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "ReFlex(window_len: {}, last_val: {}, last_m: {}, q_vals: {:?}, out: {})",
            self.window_len, self.last_val, self.last_m, self.q_vals, self.out
        )
    }
}

impl<V> ReFlex<V>
where
    V: View,
{
    /// Create a new ReFlex indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        ReFlex {
            view,
            window_len,
            last_val: 0.0,
            last_m: 0.0,
            q_vals: VecDeque::new(),
            out: 0.0,
        }
    }
}

impl<V> View for ReFlex<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.is_empty() {
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        let a1 = 8.88442402435 / self.window_len as f64;
        let b1 = 2.0 * a1 * (4.44221201218 / self.window_len as f64).cos();
        let c3 = -a1 * a1;
        let c1 = 1.0 - b1 - c3;

        let l = self.q_vals.len();
        let mut filt: f64 = 0.0;
        if l == 0 {
            filt = c1 * (val + self.last_val) / 2.0;
        } else if l == 1 {
            let filt1 = self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 * filt1;
        } else if l > 1 {
            let filt2 = self.q_vals.get(l - 2).unwrap();
            let filt1 = self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 * filt1 + c3 * filt2;
        }
        self.last_val = val;
        self.q_vals.push_back(filt);

        let slope = (self.q_vals.front().unwrap() - filt) / self.window_len as f64;

        // sum the differences
        let mut d_sum: f64 = 0.0;
        for i in 0..self.q_vals.len() {
            let index = self.q_vals.len() - 1 - i;
            d_sum += (filt + i as f64 * slope) - *self.q_vals.get(index).unwrap();
        }
        d_sum /= self.window_len as f64;

        // normalize in termsn of standard deviation
        let ms0 = 0.04 * d_sum.powi(2) + 0.96 * self.last_m;
        self.last_m = ms0;
        if ms0 > 0.0 {
            self.out = d_sum / ms0.sqrt();
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
    fn re_flex_plot() {
        let mut rf = ReFlex::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rf.update(*v);
            out.push(rf.last());
        }
        let filename = "img/re_flex.png";
        plot_values(out, filename).unwrap();
    }
}

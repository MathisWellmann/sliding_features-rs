//! John Ehlers TrendFlex Indicators
//! from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>

use std::collections::VecDeque;

use super::View;

/// John Ehlers TrendFlex Indicators
/// from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>
#[derive(Debug, Clone)]
pub struct TrendFlex<V> {
    view: V,
    window_len: usize,
    last_val: f64,
    last_m: f64,
    q_filts: VecDeque<f64>,
    out: Option<f64>,
}

impl<V> TrendFlex<V>
where
    V: View,
{
    /// Create a new TrendFlex Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        TrendFlex {
            view,
            window_len,
            last_val: 0.0,
            last_m: 0.0,
            q_filts: VecDeque::new(),
            out: None,
        }
    }
}

impl<V> View for TrendFlex<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_filts.is_empty() {
            self.last_val = val;
        }
        if self.q_filts.len() > self.window_len {
            self.q_filts.pop_front();
        }
        let a1 = (-8.88442402435 / self.window_len as f64).exp();
        let b1 = 2.0 * a1 * (4.44221201218 / self.window_len as f64).cos();
        let c3 = -a1 * a1;
        let c1 = 1.0 - b1 - c3;

        let l = self.q_filts.len();
        let mut filt: f64 = 0.0;
        if l == 0 {
            filt = c1 * (val + self.last_val) / 2.0
        } else if l == 1 {
            let filt1 = self.q_filts.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 * filt1
        } else if l > 1 {
            let filt2 = self.q_filts.get(l - 2).unwrap();
            let filt1 = self.q_filts.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / 2.0 + b1 * filt1 + c3 * filt2;
        }
        self.last_val = val;
        self.q_filts.push_back(filt);

        // sum the differences
        let mut d_sum: f64 = 0.0;
        for i in 0..self.q_filts.len() {
            let index = self.q_filts.len() - 1 - i;
            d_sum += filt - *self.q_filts.get(index).unwrap();
        }
        d_sum /= self.window_len as f64;

        // normalize in terms of standard deviation;
        let ms0 = 0.04 * d_sum.powi(2) + 0.96 * self.last_m;
        self.last_m = ms0;
        if ms0 > 0.0 {
            self.out = Some(d_sum / ms0.sqrt());
        } else {
            self.out = Some(0.0);
        }
    }

    fn last(&self) -> Option<f64> {
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
    fn trend_flex_plot() {
        let mut tf = TrendFlex::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            tf.update(*v);
            if let Some(val) = tf.last() {
                out.push(val);
            }
        }
        let filename = "img/trend_flex.png";
        plot_values(out, filename).unwrap();
    }
}

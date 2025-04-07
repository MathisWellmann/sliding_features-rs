//! John Ehlers ReFlex Indicator
//! from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// John Ehlers ReFlex Indicator
/// from: <https://financial-hacker.com/petra-on-programming-a-new-zero-lag-indicator/>
#[derive(Debug, Clone, CopyGetters)]
pub struct ReFlex<T, V> {
    view: V,
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: usize,
    last_val: T,
    last_m: T,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> ReFlex<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new ReFlex indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        ReFlex {
            view,
            window_len,
            last_val: T::zero(),
            last_m: T::zero(),
            q_vals: VecDeque::with_capacity(window_len),
            out: None,
        }
    }
}

impl<T, V> View<T> for ReFlex<T, V>
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
            self.last_val = val;
        }
        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        let window_len = T::from(self.window_len).expect("can convert");
        let two = T::from(2.0).expect("can convert");
        let a1 = T::from(8.88442402435).expect("can convert") / window_len;
        let b1 = two * a1 * (T::from(4.44221201218).expect("can convert") / window_len).cos();
        let c3 = -a1 * a1;
        let c1 = T::one() - b1 - c3;

        let l = self.q_vals.len();
        let mut filt = T::zero();
        if l == 0 {
            filt = c1 * (val + self.last_val) / two;
        } else if l == 1 {
            let filt1 = *self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / two + b1 * filt1;
        } else if l > 1 {
            let filt2 = *self.q_vals.get(l - 2).unwrap();
            let filt1 = *self.q_vals.get(l - 1).unwrap();
            filt = c1 * (val + self.last_val) / two + b1 * filt1 + c3 * filt2;
        }
        self.last_val = val;
        self.q_vals.push_back(filt);

        let slope = (*self.q_vals.front().unwrap() - filt) / window_len;

        // sum the differences
        let mut d_sum = T::zero();
        for i in 0..self.q_vals.len() {
            let index = self.q_vals.len() - 1 - i;
            d_sum = d_sum
                + ((filt + T::from(i).expect("can convert") * slope)
                    - *self.q_vals.get(index).unwrap());
        }
        d_sum = d_sum / window_len;

        // normalize in termsn of standard deviation
        let ms0 = T::from(0.04).expect("can convert") * d_sum.powi(2)
            + T::from(0.96).expect("can convert") * self.last_m;
        self.last_m = ms0;
        if ms0 > T::zero() {
            let out = d_sum / ms0.sqrt();
            debug_assert!(out.is_finite(), "value must be finite");
            self.out = Some(out);
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
    fn re_flex_plot() {
        let mut rf = ReFlex::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rf.update(*v);
            if let Some(val) = rf.last() {
                out.push(val);
            }
        }
        let filename = "img/re_flex.png";
        plot_values(out, filename).unwrap();
    }
}

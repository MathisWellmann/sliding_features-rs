//! John Ehlers Fisher Transform Indicator
//! from: <http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf>

use crate::View;
use std::collections::VecDeque;

#[derive(Clone)]
/// John Ehlers Fisher Transform Indicator
/// from: <http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf>
pub struct EhlersFisherTransform<V, M> {
    view: V,
    moving_average: M,
    window_len: usize,
    q_vals: VecDeque<f64>,
    high: f64,
    low: f64,
    q_out: VecDeque<f64>,
}

impl<V, M> std::fmt::Debug for EhlersFisherTransform<V, M>
where
    V: View,
    M: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "EhlersFisherTransform(window_len: {}, q_vals: {:?}, high: {}, low: {}, q_out: {:?})",
            self.window_len, self.q_vals, self.high, self.low, self.q_out
        )
    }
}

impl<V, M> EhlersFisherTransform<V, M>
where
    V: View,
    M: View,
{
    /// Create a new indicator with a view, moving average and window length
    #[inline]
    pub fn new(view: V, ma: M, window_len: usize) -> Self {
        Self {
            view,
            moving_average: ma,
            window_len,
            q_vals: VecDeque::new(),
            high: 0.0,
            low: 0.0,
            q_out: VecDeque::new(),
        }
    }
}

impl<V, M> View for EhlersFisherTransform<V, M>
where
    V: View,
    M: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        if self.q_vals.is_empty() {
            self.high = val;
            self.low = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old_val = self.q_vals.pop_front().unwrap();
            // update high and low values if needed
            if old_val >= self.high {
                // re-compute high
                self.high = *self
                    .q_vals
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
            }
            if old_val <= self.low {
                // re-compute low
                self.low = *self
                    .q_vals
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
            }
        }
        self.q_vals.push_back(val);
        if val > self.high {
            self.high = val;
        } else if val < self.low {
            self.low = val;
        }

        if self.high == self.low {
            self.q_out.push_back(0.0);
            return;
        }
        let val: f64 = 2.0 * ((val - self.low) / (self.high - self.low) - 0.5);
        // smooth with moving average
        self.moving_average.update(val);
        let mut smoothed = self.moving_average.last();
        smoothed = smoothed.clamp(-0.99, 0.99);

        if self.q_out.is_empty() {
            // do not insert values when there are not enough values yet
            self.q_out.push_back(0.0);
            return;
        }
        let fish: f64 =
            0.5 * ((1.0 + smoothed) / (1.0 - smoothed)).ln() + 0.5 * self.q_out.back().unwrap();
        self.q_out.push_back(fish);
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        *self.q_out.back().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::{Echo, EMA};

    #[test]
    fn ehlers_fisher_transform_plot() {
        let mut eft = EhlersFisherTransform::new(Echo::new(), EMA::new(Echo::new(), 16), 16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            eft.update(*v);
            out.push(eft.last());
        }
        println!("out: {:?}", out);
        let filename = "img/ehlers_fisher_transform.png";
        plot_values(out, filename).unwrap();
    }
}

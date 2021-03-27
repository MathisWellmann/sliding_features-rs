use crate::{View, EMA, Echo};
use std::collections::VecDeque;

#[derive(Clone)]
/// John Ehlers Fisher Transform Indicator
/// from: http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf
pub struct EhlersFisherTransform {
    view: Box<dyn View>,
    moving_average: Box<dyn View>,
    window_len: usize,
    q_vals: VecDeque<f64>,
    high: f64,
    low: f64,
    q_out: VecDeque<f64>,
}

impl EhlersFisherTransform {
    /// Create a new indicator with a given chained view and a window length
    /// The default EMA is used as in the paper
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        Self::with_ma(view, Box::new(EMA::new_final(5)), window_len)
    }

    /// Create a new indicator with a window length
    pub fn new_final(window_len: usize) -> Self {
        Self::new(Box::new(Echo::new()), window_len)
    }

    /// Create a new indicator with a view, moving average and window length
    pub fn with_ma(view: Box<dyn View>, ma: Box<dyn View>, window_len: usize) -> Self {
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

impl View for EhlersFisherTransform {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        if self.q_vals.len() == 0 {
            self.high = val;
            self.low = val;
        }
        if self.q_vals.len() >= self.window_len {
            let old_val = self.q_vals.pop_front().unwrap();
            // update high and low values if needed
            if old_val >= self.high {
                // re-compute high
                self.high = *self.q_vals.iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
            }
            if old_val <= self.low {
                // re-compute low
                self.low = *self.q_vals.iter()
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
            return
        }
        let val: f64 = 2.0 * ((val - self.low) / (self.high - self.low) - 0.5);
        // smooth with moving average
        self.moving_average.update(val);
        let mut smoothed = self.moving_average.last();
        if smoothed > 0.99 {
            // slight deviation from paper but clipping the value to 0.99 seems to make more sense
            smoothed = 0.99;
        } else if smoothed < -0.99 {
            smoothed = -0.99;
        }
        if self.q_out.len() < self.window_len {
            // do not insert values when there are not enough values yet
            self.q_out.push_back(0.0);
            return
        }
        let fish: f64 = 0.5 * ((1.0 + smoothed) / (1.0 - smoothed)).ln()
            + 0.5 * self.q_out.back().unwrap();
        self.q_out.push_back(fish);
    }

    fn last(&self) -> f64 {
        *self.q_out.back().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::plot::plot_values;

    #[test]
    fn ehlers_fisher_transform_plot() {
        let mut eft = EhlersFisherTransform::new_final(16);
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
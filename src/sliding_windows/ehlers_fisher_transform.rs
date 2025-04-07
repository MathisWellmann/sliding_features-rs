//! John Ehlers Fisher Transform Indicator
//! from: <http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf>

use crate::View;
use getset::CopyGetters;
use num::Float;
use std::{cmp::Ordering, collections::VecDeque};

/// John Ehlers Fisher Transform Indicator
/// from: <http://www.mesasoftware.com/papers/UsingTheFisherTransform.pdf>
#[derive(Clone, Debug, CopyGetters)]
pub struct EhlersFisherTransform<T, V, M> {
    view: V,
    moving_average: M,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
    q_vals: VecDeque<T>,
    high: T,
    low: T,
    q_out: VecDeque<T>,
}

impl<T, V, M> EhlersFisherTransform<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    /// Create a new indicator with a view, moving average and window length
    #[inline]
    pub fn new(view: V, ma: M, window_len: usize) -> Self {
        Self {
            view,
            moving_average: ma,
            window_len,
            q_vals: VecDeque::with_capacity(window_len),
            high: T::zero(),
            low: T::zero(),
            q_out: VecDeque::with_capacity(window_len),
        }
    }
}

impl<T, V, M> View<T> for EhlersFisherTransform<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

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
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap();
            }
            if old_val <= self.low {
                // re-compute low
                self.low = *self
                    .q_vals
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
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
            self.q_out.push_back(T::zero());
            return;
        }
        let half = T::from(0.5).expect("can convert");
        let val =
            T::from(2.0).expect("can convert") * ((val - self.low) / (self.high - self.low) - half);
        // smooth with moving average
        self.moving_average.update(val);
        let Some(mut smoothed) = self.moving_average.last() else {
            return;
        };
        smoothed = smoothed.clamp(
            T::from(-0.99).expect("can convert"),
            T::from(0.99).expect("can convert"),
        );

        if self.q_out.is_empty() {
            // do not insert values when there are not enough values yet
            self.q_out.push_back(T::zero());
            return;
        }
        let fish = half * ((T::one() + smoothed) / (T::one() - smoothed)).ln()
            + half * *self.q_out.back().unwrap();
        debug_assert!(fish.is_finite(), "value must be finite");
        self.q_out.push_back(fish);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.q_out.back().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::sliding_windows::Ema;
    use crate::test_data::TEST_DATA;

    #[test]
    fn ehlers_fisher_transform_plot() {
        let mut eft = EhlersFisherTransform::new(Echo::new(), Ema::new(Echo::new(), 16), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            eft.update(*v);
            out.push(eft.last().unwrap());
        }
        println!("out: {:?}", out);
        let filename = "img/ehlers_fisher_transform.png";
        plot_values(out, filename).unwrap();
    }
}

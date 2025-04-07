//! John Ehlers LaguerreRSI
//! from: <http://mesasoftware.com/papers/TimeWarp.pdf>

use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

use crate::View;

/// John Ehlers LaguerreRSI
/// from: <http://mesasoftware.com/papers/TimeWarp.pdf>
#[derive(Debug, Clone, CopyGetters)]
pub struct LaguerreRSI<T, V> {
    view: V,
    value: Option<T>,
    gamma: T,
    l0s: VecDeque<T>,
    l1s: VecDeque<T>,
    l2s: VecDeque<T>,
    l3s: VecDeque<T>,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
}

impl<T, V> LaguerreRSI<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new LaguerreRSI with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: usize) -> Self {
        LaguerreRSI {
            view,
            value: None,
            gamma: T::from(2.0).expect("can convert")
                / (T::from(window_len).expect("can convert") + T::one()),
            l0s: VecDeque::with_capacity(window_len),
            l1s: VecDeque::with_capacity(window_len),
            l2s: VecDeque::with_capacity(window_len),
            l3s: VecDeque::with_capacity(window_len),
            window_len,
        }
    }
}

impl<T, V> View<T> for LaguerreRSI<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.l0s.len() >= 3 {
            self.l0s.pop_front();
            self.l1s.pop_front();
            self.l2s.pop_front();
            self.l3s.pop_front();
        }

        if self.l0s.len() < 2 {
            self.l0s.push_back(T::zero());
            self.l1s.push_back(T::zero());
            self.l2s.push_back(T::zero());
            self.l3s.push_back(T::zero());
            return;
        } else {
            let last = self.l0s.len() - 1;
            self.l0s.push_back(
                (T::one() - self.gamma) * val + self.gamma * *self.l0s.get(last - 1).unwrap(),
            );
            self.l1s.push_back(
                -self.gamma * *self.l0s.get(last).unwrap()
                    + *self.l0s.get(last - 1).unwrap()
                    + self.gamma * *self.l1s.get(last - 1).unwrap(),
            );
            self.l2s.push_back(
                -self.gamma * *self.l1s.get(last).unwrap()
                    + *self.l1s.get(last - 1).unwrap()
                    + self.gamma * *self.l2s.get(last - 1).unwrap(),
            );
            self.l3s.push_back(
                -self.gamma * *self.l2s.get(last).unwrap()
                    + *self.l2s.get(last - 1).unwrap()
                    + self.gamma * *self.l3s.get(last - 1).unwrap(),
            );
        }
        let last = self.l0s.len() - 1;

        let mut cu = T::zero();
        let mut cd = T::zero();
        if self.l0s.get(last) >= self.l1s.get(last) {
            cu = *self.l0s.get(last).unwrap() - *self.l1s.get(last).unwrap();
        } else {
            cd = *self.l1s.get(last).unwrap() - *self.l0s.get(last).unwrap();
        }
        if self.l1s.get(last) >= self.l2s.get(last) {
            cu = cu + (*self.l1s.get(last).unwrap() - *self.l2s.get(last).unwrap());
        } else {
            cd = cd + (*self.l2s.get(last).unwrap() - *self.l1s.get(last).unwrap());
        }
        if self.l2s.get(last) >= self.l3s.get(last) {
            cu = cu + (*self.l2s.get(last).unwrap() - *self.l3s.get(last).unwrap());
        } else {
            cd = cd + (*self.l3s.get(last).unwrap() - *self.l2s.get(last).unwrap());
        }

        if cu + cd != T::zero() {
            let value = cu / (cu + cd);
            debug_assert!(value.is_finite(), "value must be finite");
            self.value = Some(value);
        }
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn laguerre_rsi() {
        let mut lrsi = LaguerreRSI::new(Echo::new(), 16);
        for v in &TEST_DATA {
            lrsi.update(*v);
            if let Some(last) = lrsi.last() {
                assert!(last <= 1.0);
                assert!(last >= -1.0);
            }
        }
    }

    #[test]
    fn laguerre_rsi_plot() {
        let mut lrsi = LaguerreRSI::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            lrsi.update(*v);
            if let Some(val) = lrsi.last() {
                out.push(val);
            }
        }
        // graph the results
        let filename = "img/laguerre_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

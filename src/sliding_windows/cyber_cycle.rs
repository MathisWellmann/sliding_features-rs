//! John Ehlers Cyber Cycle Indicator
//! from: <https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf>

use getset::CopyGetters;
use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

use crate::View;

/// John Ehlers Cyber Cycle Indicator
/// from: <https://www.mesasoftware.com/papers/TheInverseFisherTransform.pdf>
#[derive(Clone, Debug, CopyGetters)]
pub struct CyberCycle<T, V> {
    view: V,
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    alpha: T,
    vals: VecDeque<T>,
    out: VecDeque<T>,
    // avoid allocation in `update` step by re-using this buffer.
    smooth: Vec<T>,
}

impl<T, V> CyberCycle<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Cyber Cycle Indicator with a chained View
    /// and a given window length
    #[inline]
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        CyberCycle {
            view,
            window_len,
            alpha: T::from(2.0).expect("can convert")
                / (T::from(window_len.get()).expect("can convert") + T::one()),
            vals: VecDeque::with_capacity(window_len.get()),
            out: VecDeque::with_capacity(window_len.get()),
            smooth: vec![T::zero(); window_len.get()],
        }
    }
}

impl<T, V> View<T> for CyberCycle<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.vals.len() >= self.window_len.get() {
            self.vals.pop_front();
            self.out.pop_front();
        }
        self.vals.push_back(val);

        if self.vals.len() < self.window_len.get() {
            self.out.push_back(T::zero());
            return;
        }
        let last = self.vals.len() - 1;
        let two = T::from(2.0).expect("can convert");
        for (i, v) in self
            .smooth
            .iter_mut()
            .enumerate()
            .take(self.vals.len())
            .skip(3)
        {
            *v = (val
                + two * *self.vals.get(i - 1).unwrap()
                + two * *self.vals.get(i - 2).unwrap()
                + *self.vals.get(i - 3).unwrap())
                / T::from(6.0).expect("can convert")
        }
        let cc = (T::one() - T::from(0.5).expect("can convert") * self.alpha).powi(2)
            * (self.smooth[last] - two * self.smooth[last - 1] + self.smooth[last - 2])
            + two * (T::one() - self.alpha) * *self.out.get(last - 1).unwrap()
            - (T::one() - self.alpha).powi(2) * *self.out.get(last - 2).unwrap();
        debug_assert!(cc.is_finite(), "value must be finite");
        self.out.push_back(cc);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.out.back().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn cyber_cycle_plot() {
        let mut cc = CyberCycle::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cc.update(*v);
            out.push(cc.last().unwrap());
        }
        let filename = "img/cyber_cycle.png";
        plot_values(out, filename).unwrap();
    }
}

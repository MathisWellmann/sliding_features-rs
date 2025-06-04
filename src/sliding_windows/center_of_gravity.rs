//! John Ehlers Center of Gravity Indicator
//! from: <https://mesasoftware.com/papers/TheCGOscillator.pdf>

use getset::CopyGetters;
use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

use crate::View;

/// John Ehlers Center of Gravity Indicator
/// from: <https://mesasoftware.com/papers/TheCGOscillator.pdf>
#[derive(Clone, Debug, CopyGetters)]
pub struct CenterOfGravity<T, V> {
    view: V,
    /// The length of the sliding window.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V> CenterOfGravity<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a Center of Gravity Indicator with a chained View
    /// and a given sliding window length
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::with_capacity(window_len.get()),
            out: None,
        }
    }
}

impl<T, V> View<T> for CenterOfGravity<T, V>
where
    V: View<T>,
    T: Float,
{
    // update receives a new value and updates its internal state
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len.get() {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let mut denom = T::zero();
        let mut num = T::zero();
        let q_len = self.q_vals.len();
        for (i, val) in self.q_vals.iter().enumerate() {
            let weight = q_len - i;
            num = num + T::from(weight).expect("can convert") * *val;
            denom = denom + *val;
        }
        if denom != T::zero() {
            let out = -num / denom
                + (T::from(q_len).expect("can convert") + T::one())
                    / T::from(2.0).expect("can convert");

            debug_assert!(out.is_finite(), "value must be finite");
            self.out = Some(out);
        } else {
            self.out = Some(T::zero());
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
    fn center_of_gravity_plot() {
        let mut cgo = CenterOfGravity::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cgo.update(*v);
            out.push(cgo.last().unwrap());
        }
        let filename = "img/center_of_gravity.png";
        plot_values(out, filename).unwrap();
    }
}

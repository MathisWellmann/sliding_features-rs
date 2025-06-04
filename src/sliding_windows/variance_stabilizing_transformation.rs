//! Variance Stabilizing Transform uses the standard deviation to normalize values

use std::num::NonZeroUsize;

use crate::{pure_functions::Echo, View};
use num::Float;

use super::WelfordOnline;

/// Variance Stabilizing Transform uses the standard deviation to normalize values
#[derive(Debug, Clone)]
pub struct Vst<T: Float, V> {
    view: V,
    last: T,
    welford_online: WelfordOnline<T, Echo<T>>,
}

impl<T, V> Vst<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Variance Stabilizing Transform with a chained View
    /// and a given window length for computing standard deviation
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self {
            view,
            last: T::zero(),
            welford_online: WelfordOnline::new(Echo::new(), window_len),
        }
    }

    /// The sliding window length.
    #[inline]
    pub fn window_len(&self) -> NonZeroUsize {
        self.welford_online.window_len()
    }
}

impl<T, V> View<T> for Vst<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.welford_online.update(val);
        self.last = val;
    }

    fn last(&self) -> Option<T> {
        let std_dev = self.welford_online.last()?;
        if std_dev == T::zero() {
            return Some(self.last);
        }
        let out = self.last / std_dev;
        debug_assert!(out.is_finite(), "value must be finite");
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn variance_stabilizing_transform_plot() {
        let mut tf = Vst::new(Echo::new(), NonZeroUsize::new(16).unwrap());
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

//! Variance Stabilizing Centering Transform Sliding Window

use crate::{pure_functions::Echo, View};
use num::Float;

use super::WelfordOnline;

/// Variance Stabilizing Centering Transform Sliding Window
#[derive(Debug, Clone)]
pub struct Vsct<T: Float, V> {
    view: V,
    welford_online: WelfordOnline<T, Echo<T>>,
    last: T,
}

impl<T, V> Vsct<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Variance Stabilizing Centering Transform with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Vsct {
            view,
            welford_online: WelfordOnline::new(Echo::new(), window_len),
            last: T::zero(),
        }
    }

    /// The sliding window length.
    #[inline(always)]
    pub fn window_len(&self) -> usize {
        self.welford_online.window_len()
    }
}

impl<T, V> View<T> for Vsct<T, V>
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
            return Some(T::zero());
        }
        let mean = self.welford_online.mean();
        let out = (self.last - mean) / std_dev;
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
    fn vsct_plot() {
        let mut vsct = Vsct::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            vsct.update(*v);
            if let Some(val) = vsct.last() {
                out.push(val);
            }
        }
        let filename = "img/vsct.png";
        plot_values(out, filename).unwrap();
    }
}

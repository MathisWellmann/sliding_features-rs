//! Variance Stabilizing Centering Transform Sliding Window

use crate::{pure_functions::Echo, View};
use num::Float;

use super::WelfordOnline;

/// Variance Stabilizing Centering Transform Sliding Window
#[derive(Debug, Clone)]
pub struct VSCT<T: Float, V> {
    view: V,
    welford_online: WelfordOnline<T, Echo<T>>,
    last: T,
}

impl<T, V> VSCT<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Variance Stabilizing Centering Transform with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        VSCT {
            view,
            welford_online: WelfordOnline::new(Echo::new(), window_len),
            last: T::zero(),
        }
    }
}

impl<T, V> View<T> for VSCT<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.welford_online.update(val);
        self.last = val;
    }

    fn last(&self) -> Option<T> {
        let std_dev = self.welford_online.last()?;
        if std_dev == T::zero() {
            return Some(T::zero());
        }
        let mean = self.welford_online.mean();
        Some((self.last - mean) / std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn vsct_plot() {
        let mut vsct = VSCT::new(Echo::new(), 16);
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

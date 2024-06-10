//! Variance Stabilizing Centering Transform Sliding Window

use crate::{Echo, View, WelfordOnline};

/// Variance Stabilizing Centering Transform Sliding Window
#[derive(Debug, Clone)]
pub struct VSCT<V> {
    view: V,
    welford_online: WelfordOnline<Echo>,
    last: f64,
}

impl<V> VSCT<V>
where
    V: View,
{
    /// Create a new Variance Stabilizing Centering Transform with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        VSCT {
            view,
            welford_online: WelfordOnline::new(Echo::new(), window_len),
            last: 0.0,
        }
    }
}

impl<V> View for VSCT<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.welford_online.update(val);
        self.last = val;
    }

    fn last(&self) -> Option<f64> {
        let std_dev = self.welford_online.last()?;
        if std_dev == 0.0 {
            return Some(0.0);
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

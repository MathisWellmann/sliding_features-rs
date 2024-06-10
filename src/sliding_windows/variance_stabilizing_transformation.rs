//! Variance Stabilizing Transform uses the standard deviation to normalize values

use crate::{pure_functions::Echo, View};

use super::WelfordOnline;

/// Variance Stabilizing Transform uses the standard deviation to normalize values
#[derive(Debug, Clone)]
pub struct VST<V> {
    view: V,
    last: f64,
    welford_online: WelfordOnline<Echo>,
}

impl<V> VST<V>
where
    V: View,
{
    /// Create a new Variance Stabilizing Transform with a chained View
    /// and a given window length for computing standard deviation
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            last: 0.0,
            welford_online: WelfordOnline::new(Echo::new(), window_len),
        }
    }
}

impl<V> View for VST<V>
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
            return Some(self.last);
        }
        Some(self.last / std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn variance_stabilizing_transform_plot() {
        let mut tf = VST::new(Echo::new(), 16);
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

//! Variance Stabilizing Transform uses the standard deviation to normalize values

use crate::View;
use crate::{Echo, WelfordOnline};

/// Variance Stabilizing Transform uses the standard deviation to normalize values
#[derive(Clone)]
pub struct VST<V> {
    view: V,
    last: f64,
    welford_online: WelfordOnline<Echo>,
}

impl<V> std::fmt::Debug for VST<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "VST.last: {}", self.last)
    }
}

/// Create a new Variance Stabilizing Transform with the default Echo View
#[inline(always)]
pub fn new_final() -> VST<Echo> {
    VST::new(Echo::new())
}

impl<V> VST<V>
where
    V: View,
{
    /// Create a new Variance Stabilizing Transform with a chained View
    #[inline]
    pub fn new(view: V) -> Self {
        Self {
            view,
            last: 0.0,
            welford_online: crate::welford_online::new_final(),
        }
    }
}

impl<V> View for VST<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let view_last: f64 = self.view.last();

        self.welford_online.update(view_last);
        self.last = view_last;
    }

    fn last(&self) -> f64 {
        let std_dev = self.welford_online.last();
        if std_dev == 0.0 {
            return self.last;
        }
        self.last / std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn variance_stabilizing_transform_plot() {
        let mut tf = new_final();
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            tf.update(*v);
            out.push(tf.last());
        }
        let filename = "img/trend_flex.png";
        plot_values(out, filename).unwrap();
    }
}

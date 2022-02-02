use crate::{Echo, SuperSmoother, View};

/// Roofing Filter by John Ehlers
/// From paper: http://www.stockspotter.com/files/PredictiveIndicators.pdf
#[derive(Debug, Clone)]
pub struct RoofingFilter<V> {
    view: V,
    super_smoother: SuperSmoother<Echo>,
    window_len: usize,
    i: usize,
    alpha_1: f64,
    // previous value
    val_1: f64,
    // value from 2 steps ago
    val_2: f64,
    // high pass filter value of previous step
    hp_1: f64,
    // high pass filter value from two steps ago
    hp_2: f64,
}

impl<V> RoofingFilter<V>
where
    V: View,
{
    /// Create a Roofing Filter with a chained view
    #[inline]
    pub fn new(view: V, window_len: usize, super_smoother_len: usize) -> Self {
        // NOTE: 4.4422 radians from  0.707 * 360 degrees
        let alpha_1 = ((4.4422 / window_len as f64).cos() + (4.4422 / window_len as f64).sin()
            - 1.0)
            / (4.4422 / window_len as f64).cos();

        RoofingFilter {
            view,
            super_smoother: SuperSmoother::new(Echo::new(), super_smoother_len),
            window_len,
            i: 0,
            alpha_1,
            val_1: 0.0,
            val_2: 0.0,
            hp_1: 0.0,
            hp_2: 0.0,
        }
    }
}

impl<V> View for RoofingFilter<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        let hp = (1.0 - self.alpha_1 / 2.0).powi(2) * (val - 2.0 * self.val_1 + self.val_2)
            + 2.0 * (1.0 - self.alpha_1) * self.hp_1
            - (1.0 - self.alpha_1).powi(2) * self.hp_2;
        self.hp_2 = self.hp_1;
        self.hp_1 = hp;

        self.val_2 = self.val_1;
        self.val_1 = val;

        self.super_smoother.update(hp);
        self.i += 1;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        if self.i < self.window_len {
            0.0
        } else {
            self.super_smoother.last()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;
    use crate::Echo;

    #[test]
    fn roofing_filter_plot() {
        let mut rf = RoofingFilter::new(Echo::new(), 48, 10);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rf.update(*v);
            out.push(rf.last());
        }
        let filename = "img/roofing_filter.png";
        plot_values(out, filename).unwrap();
    }
}

use crate::View;

/// John Ehlers SuperSmoother filter
/// from https://www.mesasoftware.com/papers/PredictiveIndicatorsForEffectiveTrading%20Strategies.pdf
#[derive(Debug, Clone)]
pub struct SuperSmoother<V> {
    view: V,
    window_length: usize,
    i: usize,
    c1: f64,
    c2: f64,
    c3: f64,
    /// filter value at current step
    filt: f64,
    // filter one step ago
    filt_1: f64,
    // filter two steps ago
    filt_2: f64,
    last_val: f64,
}

impl<V> SuperSmoother<V>
where
    V: View,
{
    /// Create a new instance of the SuperSmoother with a chained View
    #[inline(always)]
    pub fn new(view: V, window_length: usize) -> Self {
        let a1 = (-1.414 * 3.14159 / window_length as f64).exp();
        // NOTE: 4.4422 is radians of 1.414 * 180 degrees
        let b1 = 2.0 * a1 * (4.4422 / window_length as f64).cos();
        let c2 = b1;
        let c3 = -a1 * a1;

        Self {
            view,
            window_length,
            i: 0,
            c1: 1.0 - c2 - c3,
            c2,
            c3,
            filt: 0.0,
            filt_1: 0.0,
            filt_2: 0.0,
            last_val: 0.0,
        }
    }
}

impl<V> View for SuperSmoother<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.filt = self.c1 * (val + self.last_val) / 2.0
            + (self.c2 * self.filt_1)
            + (self.c3 * self.filt_2);
        self.filt_2 = self.filt_1;
        self.filt_1 = self.filt;
        self.last_val = val;
        self.i += 1;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        // NOTE: filter only kicks in after warmup steps are done
        if self.i < self.window_length {
            self.last_val
        } else {
            self.filt
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{plot::plot_values, test_data::TEST_DATA, Echo};

    use super::*;

    #[test]
    fn super_smoother_plot() {
        let mut ss = SuperSmoother::new(Echo::new(), 20);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            ss.update(*v);
            out.push(ss.last());
        }
        let filename = "img/super_smoother.png";
        plot_values(out, filename).unwrap();
    }
}

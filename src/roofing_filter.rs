use super::sliding_window::View;
use crate::Echo;

/// Roofing Filter
/// From paper: http://www.stockspotter.com/files/PredictiveIndicators.pdf
/// TODO: There is still an implementation error in the roofing filter
#[derive(Clone)]
pub struct RoofingFilter {
    view: Box<dyn View>,
    val1: f64,  // previous value
    val2: f64,  // value from 2 steps ago
    hps0: f64,
    hps1: f64,
    hps2: f64,
    filt0: f64,
    filt1: f64,
    filt2: f64,
}

impl RoofingFilter {
    /// Create a Roofing Filter with a chained view
    pub fn new(view: Box<dyn View>) -> Self {
        RoofingFilter {
            view,
            val1: 0.0,
            val2: 0.0,
            hps0: 0.0,
            hps1: 0.0,
            hps2: 0.0,
            filt0: 0.0,
            filt1: 0.0,
            filt2: 0.0,
        }
    }

    /// Create a new Roofing Filter with the default Echo View
    pub fn new_final() -> Self {
        Self::new(Box::new(Echo::new()))
    }
}

impl View for RoofingFilter {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.hps2 = self.hps1;
        self.hps1 = self.hps0;
        self.hps0 = 0.36134756541 * (val - 2.0 * self.val1 + self.val2) + 0.40448768902 * self.hps1
            - 0.0409025726385 * self.hps2;

        self.val2 = self.val1;
        self.val1 = val;

        // smooth with a super smoother
        let b1 = 1.16265311587;
        let c3 = -0.411295887559;
        let c1 = 0.124321385845;

        self.filt2 = self.filt1;
        self.filt1 = self.filt0;
        self.filt0 = c1 * (self.hps0 + self.hps1) * b1 * self.filt1 + c3 * self.filt2;
    }

    fn last(&self) -> f64 {
        return self.filt0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn roofing_filter_plot() {
        let mut rf = RoofingFilter::new_final();
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rf.update(*v);
            out.push(rf.last());
        }
        let filename = "img/roofing_filter.png";
        plot_values(out, filename).unwrap();
    }
}

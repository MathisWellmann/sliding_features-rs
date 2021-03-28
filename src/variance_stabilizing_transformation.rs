use crate::sliding_window::View;
use crate::{Echo, WelfordOnline};

/// Variance Stabilizing Transform uses the standard deviation to normalize values
#[derive(Clone)]
pub struct VST {
    view: Box<dyn View>,
    last: f64,
    welford_online: Box<WelfordOnline>,
}

impl VST {
    /// Create a new Variance Stabilizing Transform with a chained View
    pub fn new(view: Box<dyn View>) -> Box<Self> {
        Box::new(Self {
            view,
            last: 0.0,
            welford_online: WelfordOnline::new_final(),
        })
    }

    /// Create a new Variance Stabilizing Transform with the default Echo View
    pub fn new_final() -> Box<Self> {
        Self::new(Echo::new())
    }
}

impl View for VST {
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
        let mut tf = VST::new_final();
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            tf.update(*v);
            out.push(tf.last());
        }
        let filename = "img/trend_flex.png";
        plot_values(out, filename).unwrap();
    }
}

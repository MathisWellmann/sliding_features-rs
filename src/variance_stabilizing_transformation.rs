use crate::sliding_window::View;
use crate::{Echo, WelfordOnline};

/// Variance Stabilizing Transform uses the standard deviation to normalize values
#[derive(Clone)]
pub struct VST {
    view: Box<dyn View>,
    last: f64,
    welford_online: WelfordOnline,
}

impl VST {
    /// Create a new Variance Stabilizing Transform with a chained View
    pub fn new(view: Box<dyn View>) -> Self {
        Self {
            view,
            last: 0.0,
            welford_online: WelfordOnline::new_final(),
        }
    }

    /// Create a new Variance Stabilizing Transform with the default Echo View
    pub fn new_final() -> Self {
        Self::new(Box::new(Echo::new()))
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
            return 0.0;
        }
        self.last / std_dev
    }
}

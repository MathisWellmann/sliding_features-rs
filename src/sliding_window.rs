use crate::View;

/// A Sliding Window holding any number of Sliding Features (Views)
pub struct SlidingWindow {
    /// A Vector of individual, chainable sliding windows (View)
    pub views: Vec<Box<dyn View>>,
}

impl SlidingWindow {
    /// Create a new, empty SlidingWindow
    pub fn new() -> SlidingWindow {
        return SlidingWindow { views: Vec::new() };
    }

    /// Update all the Views with a new value
    pub fn update(&mut self, val: f64) {
        for i in 0..self.views.len() {
            self.views[i].update(val);
        }
    }

    /// Return the last values from all Views
    pub fn last(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for i in 0..self.views.len() {
            let last = self.views[i].last();
            out.push(last)
        }
        return out;
    }

    /// Add the given View
    pub fn register_view(&mut self, view: Box<dyn View>) {
        self.views.push(view);
    }
}

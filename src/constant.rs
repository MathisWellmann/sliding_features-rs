use crate::View;

/// Provides a float value to other views
#[derive(Default, Debug, Clone)]
pub struct Constant {
    val: f64,
}

impl Constant {
    /// Create a new instance with the given value
    pub fn new(val: f64) -> Self {
        Self { val }
    }
}

impl View for Constant {
    fn update(&mut self, _val: f64) {}

    fn last(&self) -> Option<f64> {
        Some(self.val)
    }
}

use crate::View;

/// Provides a float value to other views
#[derive(Default, Debug, Clone)]
pub struct Constant<T> {
    val: T,
}

impl<T> Constant<T> {
    /// Create a new instance with the given value
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

impl<T: num::Float> View<T> for Constant<T> {
    fn update(&mut self, _val: T) {}

    fn last(&self) -> Option<T> {
        Some(self.val)
    }
}

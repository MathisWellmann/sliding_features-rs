use crate::View;

/// Applies the Tanh function to the output of its View component
#[derive(Debug, Clone)]
pub struct Tanh<V> {
    view: V,
}

impl<V> Tanh<V>
where
    V: View,
{
    /// Create a new instance with a chained View
    pub fn new(view: V) -> Self {
        Self { view }
    }
}

impl<V> View for Tanh<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
    }

    fn last(&self) -> Option<f64> {
        self.view.last().map(|v| v.tanh())
    }
}

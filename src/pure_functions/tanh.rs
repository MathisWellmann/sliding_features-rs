use crate::View;
use num::Float;

/// Applies the Tanh function to the output of its View component
#[derive(Debug, Clone)]
pub struct Tanh<T, V> {
    view: V,
    _marker: std::marker::PhantomData<T>,
}

impl<T, V> Tanh<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View
    pub fn new(view: V) -> Self {
        Self {
            view,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, V> View<T> for Tanh<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
    }

    fn last(&self) -> Option<T> {
        self.view.last().map(|v| v.tanh())
    }
}

use num::Float;

use crate::View;

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
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
    }

    fn last(&self) -> Option<T> {
        self.view.last().map(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
            v.tanh()
        })
    }
}

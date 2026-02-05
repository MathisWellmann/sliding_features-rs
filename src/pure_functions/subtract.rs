use num::Float;

use crate::View;

/// Subtract View a from b
#[derive(Debug, Clone)]
pub struct Subtract<T, A, B> {
    a: A,
    b: B,
    _marker: std::marker::PhantomData<T>,
}

impl<T, A, B> Subtract<T, A, B>
where
    A: View<T>,
    B: View<T>,
    T: Float,
{
    /// Create a new instance with Views a and b
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, A, B> View<T> for Subtract<T, A, B>
where
    A: View<T>,
    B: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.a.update(val);
        self.b.update(val);
    }

    fn last(&self) -> Option<T> {
        match (self.a.last(), self.b.last()) {
            (Some(a), Some(b)) => {
                debug_assert!(a.is_finite(), "value must be finite");
                debug_assert!(b.is_finite(), "value must be finite");
                Some(a - b)
            }
            (None, None) | (None, Some(_)) | (Some(_), None) => None,
        }
    }
}

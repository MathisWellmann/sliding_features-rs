use crate::View;

/// Add View a to b
#[derive(Debug)]
pub struct Add<A, B> {
    a: A,
    b: B,
}

impl<A, B> Add<A, B>
where
    A: View,
    B: View,
{
    /// Create a new instance with Views a and b
    #[inline]
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> View for Add<A, B>
where
    A: View,
    B: View,
{
    #[inline]
    fn update(&mut self, val: f64) {
        self.a.update(val);
        self.b.update(val);
    }

    #[inline]
    fn last(&self) -> Option<f64> {
        match (self.a.last(), self.b.last()) {
            (Some(a), Some(b)) => Some(a + b),
            (None, None) | (None, Some(_)) | (Some(_), None) => None,
        }
    }
}

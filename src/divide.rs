use crate::View;

/// Divide View a by b
#[derive(Debug, Clone)]
pub struct Divide<A, B> {
    a: A,
    b: B,
}

impl<A, B> Divide<A, B>
where
    A: View,
    B: View,
{
    /// Create a new instance with Views a and b
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> View for Divide<A, B>
where
    A: View,
    B: View,
{
    fn update(&mut self, val: f64) {
        self.a.update(val);
        self.b.update(val);
    }

    fn last(&self) -> Option<f64> {
        match (self.a.last(), self.b.last()) {
            (Some(a), Some(b)) => Some(a / b),
            (None, None) | (None, Some(_)) | (Some(_), None) => None,
        }
    }
}

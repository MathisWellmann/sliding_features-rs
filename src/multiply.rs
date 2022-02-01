use crate::View;

/// Multiply View a by b
pub struct Multiply<A, B> {
    a: A,
    b: B,
}

impl<A, B> Multiply<A, B>
where
    A: View,
    B: View,
{
    /// Create a new Instance with Views a and b
    #[inline]
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> View for Multiply<A, B>
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
    fn last(&self) -> f64 {
        self.a.last() * self.b.last()
    }
}

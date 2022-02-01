use crate::View;

/// Add View a to b
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
    fn last(&self) -> f64 {
        self.a.last() + self.b.last()
    }
}

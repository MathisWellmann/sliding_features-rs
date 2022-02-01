use crate::View;

/// Subtract View a from b
pub struct Subtract<A, B> {
    a: A,
    b: B,
}

impl<A, B> View for Subtract<A, B>
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
        self.a.last() - self.b.last()
    }
}

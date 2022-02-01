use crate::View;

/// Divide View a by b
pub struct Divide<A, B> {
    a: A,
    b: B,
}

impl<A, B> View for Divide<A, B>
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
        self.a.last() / self.b.last()
    }
}

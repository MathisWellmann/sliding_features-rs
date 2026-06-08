//! Retains the first value it sees and never updates again.

use crate::View;

/// Retains the first value it ever sees and never updates it again.
#[derive(Default, Clone, Debug)]
pub struct First<T> {
    out: Option<T>,
}

impl<T> First<T> {
    /// Create a new First View
    #[inline(always)]
    pub fn new() -> First<T> {
        First { out: None }
    }
}

impl<T: num::Float> View<T> for First<T> {
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        if self.out.is_none() {
            self.out = Some(val);
        }
    }

    fn last(&self) -> Option<T> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first() {
        let mut f = First::new();
        assert_eq!(f.last(), None);
        f.update(42.0);
        assert_eq!(f.last(), Some(42.0));
        f.update(7.0);
        assert_eq!(f.last(), Some(42.0));
        f.update(99.0);
        assert_eq!(f.last(), Some(42.0));
    }
}

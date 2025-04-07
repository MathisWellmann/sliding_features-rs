use crate::{View, pure_functions::Echo};
use num::Float;

/// Computes the natural logarithm and keep track of the last value.
/// Usually applied to price data.
#[derive(Debug, Clone)]
pub struct LnReturn<T, V> {
    view: V,
    last_val: T,
    current_val: T,
}

impl<T: Float> Default for LnReturn<T, Echo<T>> {
    fn default() -> Self {
        Self::new(Echo::new())
    }
}

impl<T, V> LnReturn<T, V>
where
    T: Float,
    V: View<T>,
{
    /// Create a new instance of `Self` with a chained `View`, whose output will be used to feed the ln return computation.
    pub fn new(view: V) -> Self {
        Self {
            view,
            last_val: T::zero(),
            current_val: T::zero(),
        }
    }
}

impl<T, V> View<T> for LnReturn<T, V>
where
    T: Float,
    V: View<T>,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.last_val = self.current_val;
        self.current_val = val;
    }

    fn last(&self) -> Option<T> {
        if self.last_val == T::zero() {
            return None;
        }

        Some((self.current_val / self.last_val).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_return() {
        let mut ln_return = LnReturn::new(Echo::new());
        ln_return.update(100.0);
        assert!(ln_return.last().is_none());
        ln_return.update(110.0);
        assert_eq!(ln_return.last().unwrap(), 0.09531017980432493);
    }
}

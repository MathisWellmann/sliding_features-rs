use num::Float;

use crate::View;

/// The natural logarithm of the output of its View component, e.g. log
/// prices, whose differences are returns.
///
/// The logarithm is only defined for positive values: for zero or negative
/// input, `last` is `None` rather than a non-finite value that would poison
/// every window downstream.
#[derive(Debug, Clone)]
pub struct Ln<T, V> {
    view: V,
    _marker: std::marker::PhantomData<T>,
}

impl<T, V> Ln<T, V>
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

impl<T, V> View<T> for Ln<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
    }

    fn last(&self) -> Option<T> {
        let v = self.view.last()?;
        debug_assert!(v.is_finite(), "value must be finite");
        (v > T::zero()).then(|| v.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_functions::Echo;

    #[test]
    fn ln_of_positive_values() {
        let mut ln = Ln::new(Echo::new());
        assert_eq!(ln.last(), None);
        ln.update(1.0);
        assert_eq!(ln.last(), Some(0.0));
        ln.update(std::f64::consts::E);
        assert!((ln.last().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ln_of_non_positive_values_is_none() {
        let mut ln = Ln::new(Echo::new());
        ln.update(0.0);
        assert_eq!(ln.last(), None);
        ln.update(-1.0);
        assert_eq!(ln.last(), None);
    }
}

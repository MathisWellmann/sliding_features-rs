// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use num::Float;

use crate::View;

/// The absolute value of the output of its View component, e.g. the size of
/// a return regardless of its direction.
#[derive(Debug, Clone)]
pub struct Abs<T, V> {
    view: V,
    _marker: std::marker::PhantomData<T>,
}

impl<T, V> Abs<T, V>
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

impl<T, V> View<T> for Abs<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
    }

    fn last(&self) -> Option<T> {
        self.view.last().map(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
            v.abs()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_functions::Echo;

    #[test]
    fn abs() {
        let mut abs = Abs::new(Echo::new());
        assert_eq!(abs.last(), None);
        abs.update(-2.5);
        assert_eq!(abs.last(), Some(2.5));
        abs.update(3.0);
        assert_eq!(abs.last(), Some(3.0));
    }
}

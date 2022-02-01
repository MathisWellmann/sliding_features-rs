//! Simply multiply the output of View by a certain number

use crate::View;

#[derive(Clone)]
/// Simply multiply the output of View by a certain number
pub struct Multiplier<V> {
    view: V,
    multiplier: f64,
    out: f64,
}

impl<V> std::fmt::Debug for Multiplier<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "Multiplier(multiplier: {}, out: {})",
            self.multiplier, self.out
        )
    }
}

impl<V> Multiplier<V>
where
    V: View,
{
    /// Create a new multiplier with a chained view and a given value
    #[inline]
    pub fn new(view: V, multiplier: f64) -> Self {
        Self {
            view,
            multiplier,
            out: 0.0,
        }
    }
}

impl<V> View for Multiplier<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.out = val * self.multiplier;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

use crate::View;

/// Computes the natural logarithm and keep track of the last value.
/// Usually applied to price data.
#[derive(Debug, Clone)]
pub struct LnReturn<V> {
    view: V,
    last_val: f64,
    current_val: f64,
}

impl<V> LnReturn<V>
where
    V: View,
{
    /// Create a new instance of `Self` with a chained `View`, whose output will be used to feed the ln return computation.
    pub fn new(view: V) -> Self {
        Self {
            view,
            last_val: 0.0,
            current_val: 0.0,
        }
    }
}

impl<V> View for LnReturn<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.last_val = self.current_val;
        self.current_val = val;
    }

    fn last(&self) -> f64 {
        // TODO: return `None` if `self.last_val == 0`. change api of `View`
        (self.current_val / self.last_val).ln()
    }
}

#[cfg(test)]
mod tests {
    use crate::Echo;

    use super::*;

    #[test]
    fn ln_return() {
        let mut ln_return = LnReturn::new(Echo::new());
        ln_return.update(100.0);
        ln_return.update(110.0);
        assert_eq!(ln_return.last(), 0.09531017980432493);
    }
}

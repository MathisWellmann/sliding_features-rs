use crate::{pure_functions::Echo, View};
use num::Float;

/// Keep track of the current peak to valley.
#[derive(Debug, Clone)]
pub struct Drawdown<T, V> {
    view: V,
    max_val: T,
    current_val: Option<T>,
}

impl<T: Float> Default for Drawdown<T, Echo<T>> {
    fn default() -> Self {
        Self::new(Echo::new())
    }
}

impl<T, V> Drawdown<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance of `Self` with a chained `View`, so that the `view` will be updated first and its value will be used by `Self`.
    pub fn new(view: V) -> Self {
        Self {
            view,
            max_val: T::min_value(),
            current_val: None,
        }
    }
}

impl<T, V> View<T> for Drawdown<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if val > self.max_val {
            self.max_val = val;
        }
        self.current_val = Some(val);
    }

    fn last(&self) -> Option<T> {
        self.current_val
            .map(|current| (self.max_val - current) / self.max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawdown() {
        let mut dd = Drawdown::default();
        dd.update(100.0);
        assert_eq!(dd.last().unwrap(), 0.0);
        dd.update(80.0);
        assert_eq!(dd.last().unwrap(), 0.2);
        dd.update(110.0);
        assert_eq!(dd.last().unwrap(), 0.0);
    }
}

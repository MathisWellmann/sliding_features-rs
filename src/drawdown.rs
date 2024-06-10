use crate::{Echo, View};

/// Keep track of the current peak to valley.
#[derive(Debug, Clone)]
pub struct Drawdown<V> {
    view: V,
    max_val: f64,
    current_val: Option<f64>,
}

impl Default for Drawdown<Echo> {
    fn default() -> Self {
        Self::new(Echo::new())
    }
}

impl<V> Drawdown<V>
where
    V: View,
{
    /// Create a new instance of `Self` with a chained `View`, so that the `view` will be updated first and its value will be used by `Self`.
    pub fn new(view: V) -> Self {
        Self {
            view,
            max_val: f64::MIN,
            current_val: None,
        }
    }
}

impl<V> View for Drawdown<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if val > self.max_val {
            self.max_val = val;
        }
        self.current_val = Some(val);
    }

    fn last(&self) -> Option<f64> {
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

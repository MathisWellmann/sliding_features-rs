use crate::{pure_functions::Echo, View};
use num::Float;

/// Keep track of the current peak to valley.
#[derive(Debug, Clone)]
pub struct Drawdown<T, V> {
    view: V,
    max_drawdown: T,
    peak: T,
    min_after_peak: T,
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
            max_drawdown: T::zero(),
            // The highest value observed.
            peak: T::min_value(),
            // The minimum value observed after the peak.
            min_after_peak: T::max_value(),
        }
    }
}

impl<T, V> View<T> for Drawdown<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if val > self.peak {
            self.peak = val;
            self.min_after_peak = val;
        }
        if val < self.min_after_peak {
            self.min_after_peak = val;
        }
        let dd = (self.peak - self.min_after_peak) / self.peak;
        if dd > self.max_drawdown {
            self.max_drawdown = dd;
        }
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        debug_assert!(self.max_drawdown.is_finite(), "value must be finite");
        Some(self.max_drawdown)
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
        assert_eq!(dd.last().unwrap(), 0.2);
        dd.update(95.0);
        assert_eq!(dd.last().unwrap(), 0.2);
        dd.update(87.0);
        assert_eq!(dd.last().unwrap(), 0.20909090909090908);
    }
}

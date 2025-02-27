//! John Ehlers Correlation Trend Indicator
//! from: <https://financial-hacker.com/petra-on-programming-a-unique-trend-indicator/>

use crate::View;
use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

/// John Ehlers Correlation Trend Indicator
/// from: <https://financial-hacker.com/petra-on-programming-a-unique-trend-indicator/>
#[derive(Clone, Debug, CopyGetters)]
pub struct CorrelationTrendIndicator<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
    q_vals: VecDeque<T>,
}

impl<T, V> CorrelationTrendIndicator<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Correlation Trend Indicator with a chained View
    /// and a given sliding window length
    #[inline]
    pub fn new(view: V, window_len: usize) -> Self {
        Self {
            view,
            window_len,
            q_vals: VecDeque::new(),
        }
    }
}

impl<T, V> View<T> for CorrelationTrendIndicator<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            let _ = self.q_vals.pop_front().unwrap();
        }
        self.q_vals.push_back(val);
    }

    fn last(&self) -> Option<T> {
        let mut sx = T::zero();
        let mut sy = T::zero();
        let mut sxx = T::zero();
        let mut sxy = T::zero();
        let mut syy = T::zero();

        for (i, v) in self.q_vals.iter().enumerate() {
            let count = T::from(i).expect("can convert");
            sx = sx + *v;
            sy = sy + count;
            sxx = sxx + v.powi(2);
            sxy = sxy + *v * count;
            syy = syy + count.powi(2);
        }
        let window_len = T::from(self.window_len).expect("Can convert");
        if window_len * sxx - sx.powi(2) > T::zero() && window_len * syy - sy.powi(2) > T::zero() {
            return Some(
                (window_len * sxy - sx * sy)
                    / ((window_len * sxx - sx.powi(2)) * (window_len * syy - sy.powi(2))).sqrt(),
            );
        }
        Some(T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn correlation_trend_indicator() {
        // Test if indicator is bounded in range [-1, 1.0]
        let mut cti = CorrelationTrendIndicator::new(Echo::new(), 10);
        for v in &TEST_DATA {
            cti.update(*v);
            let last = cti.last().unwrap();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }

    #[test]
    fn correlation_trend_indicator_plot() {
        let mut cti = CorrelationTrendIndicator::new(Echo::new(), 16);
        let mut outs: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            cti.update(*v);
            let last = cti.last().unwrap();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
            outs.push(last);
        }

        let filename = "./img/correlation_trend_indicator.png";
        plot_values(outs, filename).unwrap();
    }
}

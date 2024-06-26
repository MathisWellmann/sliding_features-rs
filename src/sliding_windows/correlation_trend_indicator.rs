//! John Ehlers Correlation Trend Indicator
//! from: <https://financial-hacker.com/petra-on-programming-a-unique-trend-indicator/>

use crate::View;
use std::collections::VecDeque;

/// John Ehlers Correlation Trend Indicator
/// from: <https://financial-hacker.com/petra-on-programming-a-unique-trend-indicator/>
#[derive(Clone)]
pub struct CorrelationTrendIndicator<V> {
    view: V,
    window_len: usize,
    q_vals: VecDeque<f64>,
}

impl<V> std::fmt::Debug for CorrelationTrendIndicator<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "CorrelationTrendIndicator(window_len: {}, q_vals: {:?})",
            self.window_len, self.q_vals
        )
    }
}

impl<V> CorrelationTrendIndicator<V>
where
    V: View,
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

impl<V> View for CorrelationTrendIndicator<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            let _ = self.q_vals.pop_front().unwrap();
        }
        self.q_vals.push_back(val);
    }

    fn last(&self) -> Option<f64> {
        let mut sx: f64 = 0.0;
        let mut sy: f64 = 0.0;
        let mut sxx: f64 = 0.0;
        let mut sxy: f64 = 0.0;
        let mut syy: f64 = 0.0;

        for (count, v) in self.q_vals.iter().enumerate() {
            sx += v;
            sy += count as f64;
            sxx += v.powi(2);
            sxy += v * count as f64;
            syy += (count as f64).powi(2);
        }
        if self.window_len as f64 * sxx - sx.powi(2) > 0.0
            && self.window_len as f64 * syy - sy.powi(2) > 0.0
        {
            return Some(
                (self.window_len as f64 * sxy - sx * sy)
                    / ((self.window_len as f64 * sxx - sx.powi(2))
                        * (self.window_len as f64 * syy - sy.powi(2)))
                    .sqrt(),
            );
        }
        Some(0.0)
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

use crate::View;
use std::collections::VecDeque;

pub struct CorrelationTrendIndicator {
    window_len: usize,
    q_vals: VecDeque<f64>,
}

impl CorrelationTrendIndicator {
    pub fn new(window_len: usize) -> Self {
        Self {
            window_len,
            q_vals: VecDeque::new(),
        }
    }
}

impl View for CorrelationTrendIndicator {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() >= self.window_len {
            let _ = self.q_vals.pop_front().unwrap();
        }
        self.q_vals.push_back(val);
    }

    fn last(&self) -> f64 {
        let mut sx: f64 = 0.0;
        let mut sy: f64 = 0.0;
        let mut sxx: f64 = 0.0;
        let mut sxy: f64 = 0.0;
        let mut syy: f64 = 0.0;

        let mut count: usize = 0;
        for v in self.q_vals.iter() {
            sx += v;
            sy += count as f64;
            sxx += v.powi(2);
            sxy += v * count as f64;
            syy += (count as f64).powi(2);
            count += 1;
        }
        if self.window_len as f64 * sxx - sx.powi(2) > 0.0
            && self.window_len as f64 * syy - sy.powi(2) > 0.0 {
            return (self.window_len as f64 * sxy - sx * sy)
                / ((self.window_len as f64 * sxx - sx.powi(2))
                * (self.window_len as f64 * syy - sy.powi(2))).sqrt()
        }
        return 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_timeseries_generator::gaussian_process::gen;
    use rust_timeseries_generator::plt::plt;

    #[test]
    fn correlation_trend_indicator() {
        // Test if indicator is bounded in range [-1, 1.0]
        let vals: Vec<f64> = gen(1000, 100.0);

        let mut cti = CorrelationTrendIndicator::new(10);
        for v in &vals {
            cti.update(*v);
            let last = cti.last();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }

    #[test]
    fn plot_correlation_trend_indicator() {
        // Test if indicator is bounded in range [-1.0, 1.0]
        let vals: Vec<f64> = gen(1000, 100.0);

        let mut cti = CorrelationTrendIndicator::new(100);
        let mut outs: Vec<f64> = vec![];
        for v in &vals {
            cti.update(*v);
            let last = cti.last();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
            outs.push(last);
        }
        let filename = "./img/plot_correlation_trend_indicator_vals.png";
        plt(vals, filename).unwrap();

        let filename = "./img/plot_correlation_trend_indicator_cti.png";
        plt(outs, filename).unwrap();
    }
}
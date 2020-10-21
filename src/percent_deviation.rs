use super::sliding_window::View;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PercentDeviation {
    window_len: usize,
    mean: f64,
    d_squared: f64,
    q_vals: VecDeque<f64>,
}

impl PercentDeviation {
    pub fn new(window_len: usize) -> PercentDeviation {
        return PercentDeviation{
            window_len,
            mean: 0.0,
            d_squared: 0.0,
            q_vals: VecDeque::new(),
        }
    }

    fn variance(&self) -> f64 {
        if self.q_vals.len() > 1 {
            return self.d_squared / self.q_vals.len() as f64;
        }
        return 0.0
    }
}

#[tonic::async_trait]
impl View for PercentDeviation {
    fn update(&mut self, val: f64) {
        if self.q_vals.len() >= self.window_len {
            self.q_vals.push_back(val);

            // once queue is full, adjust welford method for window size
            let old_val = *self.q_vals.front().unwrap();
            self.q_vals.pop_front();

            let mean_incr = (val - old_val) / self.window_len as f64;
            let new_mean = self.mean + mean_incr;

            let d_squared_incr = (val - old_val) * (val - new_mean + old_val - self.mean);

            self.mean = new_mean;
            self.d_squared += d_squared_incr;

        } else {
            self.q_vals.push_back(val);

            // welford method
            let mean_incr = (val - self.mean) / self.q_vals.len() as f64;
            let new_mean = self.mean + mean_incr;

            let d_squared_incr = (val - new_mean) * (val - self.mean);

            self.mean = new_mean;
            self.d_squared += d_squared_incr;
        }
    }

    fn last(&self) -> f64 {
        return self.variance().sqrt() / self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_timeseries_generator::plt;

    #[test]
    fn percent_deviation() {
        let window_len: usize = 16;
        let mut pd = PercentDeviation::new(window_len);
        for i in 0..1_000 {
            let val: f64 = if i % 2 == 0 {
                8.0
            } else {
                6.0
            };
            pd.update(val);
            if i > window_len {
                let last = pd.last();
                println!("last: {}", last);
                assert!(!last.is_nan());
                assert!(pd.mean > 6.0);
                assert!(pd.mean < 8.0);
            }
        }
        assert_eq!(pd.mean, 7.0);
        assert!(pd.last() <= 0.2);
    }

    #[test]
    fn percent_deviation_graph() {
        let window_len: usize = 16;
        let mut pd = PercentDeviation::new(window_len);
        let mut out: Vec<f64> = Vec::new();
        for i in 0..1_000 {
            let val: f64 = if i % 2 == 0 {
                8.0
            } else {
                6.0
            };
            pd.update(val);
            if i > window_len {
                out.push(pd.last());
            }
        }
        let filename = "img/percent_deviation.png";
        plt::plt(out, filename).unwrap();
    }
}
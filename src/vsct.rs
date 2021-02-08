use crate::{Echo, View, WelfordOnline};

/// Variance Stabilizing Centering Transform Sliding Window
#[derive(Clone)]
pub struct VSCT {
    view: Box<dyn View>,
    window_len: usize,
    welford_online: WelfordOnline,
    last: f64,
}

impl VSCT {
    /// Create a new Variance Stabilizing Centering Transform with a chained View
    /// and a given sliding window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        VSCT {
            view,
            window_len,
            welford_online: WelfordOnline::new_final(),
            last: 0.0,
        }
    }

    /// Create a new Variance Stabilizing Centering Transform with a given window length
    pub fn new_final(window_len: usize) -> Self {
        Self::new(Box::new(Echo::new()), window_len)
    }
}

impl View for VSCT {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let view_last = self.view.last();
        self.welford_online.update(view_last);
        self.last = view_last;
    }

    fn last(&self) -> f64 {
        let std_dev = self.welford_online.last();
        if std_dev == 0.0 {
            return 0.0;
        }
        let mean = self.welford_online.mean();
        (self.last - mean) / std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_timeseries_generator::plt;

    #[test]
    fn vsct() {
        let rands: Vec<f64> = (0..100)
            .map(|_| rand::random::<f64>() * 20.0 + 100.0)
            .collect();
        println!("rands: {:?}", rands);

        let mut vsct = VSCT::new_final(20);
        let mut out: Vec<f64> = Vec::with_capacity(rands.len());
        for v in &rands {
            vsct.update(*v);
            out.push(vsct.last());
        }
        let filename = "img/vsct.png";
        plt::plt(out, filename).unwrap();
    }
}

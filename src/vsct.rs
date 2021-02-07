use crate::{View, WelfordOnline};

/// Variance Stabilizing Centering Transform Sliding Window
#[derive(Clone)]
pub struct VSCT {
    view: Box<dyn View>,
    window_len: usize,
    welford_online: WelfordOnline,
    last: f64,
}

impl VSCT {
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        VSCT {
            view,
            window_len,
            welford_online: WelfordOnline::new(),
            last: 0.0,
        }
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
            return 0.0
        }
        let mean = self.welford_online.mean;
        (self.last - mean) / std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Echo;
    use rust_timeseries_generator::plt;

    #[test]
    fn vsct() {
        let rands: Vec<f64> = (0..100).map(|v| rand::random::<f64>() * 20.0 + 100.0).collect();
        println!("rands: {:?}", rands);

        let mut vsct = VSCT::new(Box::new(Echo::new()), 20);
        let mut out: Vec<f64> = Vec::with_capacity(rands.len());
        for v in &rands {
            vsct.update(*v);
            out.push(vsct.last());
        }
        let filename = "img/vsct.png";
        plt::plt(out, filename);
    }
}
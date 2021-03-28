use std::collections::VecDeque;

use crate::{Echo, View, EMA};

#[derive(Clone)]
/// A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
/// it is also possible to use a custom moving average instead of the default EMA in the original
pub struct PolarizedFractalEfficiency {
    view: Box<dyn View>,
    moving_average: Box<dyn View>, // defines which moving average to use, default is EMA
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl PolarizedFractalEfficiency {
    /// Create a new PolarizedFractalEfficiency indicator with a chained view and a given window
    /// length. The window length will also be used for the EMA
    pub fn new(view: Box<dyn View>, window_len: usize) -> Box<Self> {
        Self::with_ma(view, EMA::new_final(window_len), window_len)
    }

    /// Create a new PolarizedFractalEfficiency indicator with a given window length
    pub fn new_final(window_len: usize) -> Box<Self> {
        Self::new(Echo::new(), window_len)
    }

    /// Create a new PolarizedFractalEfficiency indicator with a chained view, custom moving
    /// average and a window length
    pub fn with_ma(
        view: Box<dyn View>,
        moving_average: Box<dyn View>,
        window_len: usize,
    ) -> Box<Self> {
        Box::new(Self {
            view,
            moving_average,
            window_len,
            q_vals: VecDeque::new(),
            out: 0.0,
        })
    }
}

impl View for PolarizedFractalEfficiency {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() < self.window_len {
            self.out = 0.0;
        } else {
            let mut s: f64 = 0.0;
            let wl: usize = self.window_len - 1;
            for i in 0..self.window_len - 2 {
                let v_0: f64 = *self.q_vals.get(wl - i).unwrap();
                let v_1: f64 = *self.q_vals.get(wl - i - 1).unwrap();
                s += ((v_0 - v_1).powi(2) + 1.0).sqrt();
            }
            let mut p: f64 = ((val - self.q_vals.front().unwrap()).powi(2)
                + (self.window_len as f64).powi(2))
            .sqrt()
                / s;
            if val < *self.q_vals.get(self.window_len - 2).unwrap() {
                p = -p;
            }
            // apply a moving average
            self.moving_average.update(p);
            self.out = self.moving_average.last();
        }
    }

    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn polarized_fractal_efficiency() {
        let mut pfe = PolarizedFractalEfficiency::new_final(16);
        for v in &TEST_DATA {
            pfe.update(*v);
            assert!(pfe.last() <= 1.0);
            assert!(pfe.last() >= -1.0);
        }
    }

    #[test]
    fn polarized_fractal_efficiency_plot() {
        let mut pfe = PolarizedFractalEfficiency::new_final(16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            pfe.update(*v);
            out.push(pfe.last());
        }
        let filename = "img/polarized_fractal_efficiency.png";
        plot_values(out, filename).unwrap();
    }
}

//! A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
//! it is also possible to use a custom moving average instead of the default EMA in the original

use std::collections::VecDeque;

use crate::View;

#[derive(Debug, Clone)]
/// A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
/// it is also possible to use a custom moving average instead of the default EMA in the original
pub struct PolarizedFractalEfficiency<V, M> {
    view: V,
    moving_average: M, // defines which moving average to use, default is EMA
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: Option<f64>,
}

impl<V, M> PolarizedFractalEfficiency<V, M>
where
    V: View,
    M: View,
{
    /// Create a new PolarizedFractalEfficiency indicator with a chained view, custom moving
    /// average and a window length
    pub fn new(view: V, moving_average: M, window_len: usize) -> Self {
        Self {
            view,
            moving_average,
            window_len,
            q_vals: VecDeque::new(),
            out: None,
        }
    }
}

impl<V, M> View for PolarizedFractalEfficiency<V, M>
where
    V: View,
    M: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() >= self.window_len {
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

    fn last(&self) -> Option<f64> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::sliding_windows::Ema;
    use crate::test_data::TEST_DATA;

    #[test]
    fn polarized_fractal_efficiency() {
        // TODO: don't be so lazy with this test. maybe compare against a reference implementation.
        let mut pfe = PolarizedFractalEfficiency::new(Echo::new(), Ema::new(Echo::new(), 16), 16);
        for v in &TEST_DATA {
            pfe.update(*v);
            if let Some(val) = pfe.last() {
                assert!(val <= 1.0);
                assert!(val >= -1.0);
            }
        }
    }

    #[test]
    fn polarized_fractal_efficiency_plot() {
        let mut pfe = PolarizedFractalEfficiency::new(Echo::new(), Ema::new(Echo::new(), 16), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            pfe.update(*v);
            if let Some(val) = pfe.last() {
                out.push(val);
            }
        }
        let filename = "img/polarized_fractal_efficiency.png";
        plot_values(out, filename).unwrap();
    }
}

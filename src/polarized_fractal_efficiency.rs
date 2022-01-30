//! A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
//! it is also possible to use a custom moving average instead of the default EMA in the original

use std::collections::VecDeque;

use crate::{Echo, View, EMA};

#[derive(Clone)]
/// A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
/// it is also possible to use a custom moving average instead of the default EMA in the original
pub struct PolarizedFractalEfficiency<V, M> {
    view: V,
    moving_average: M, // defines which moving average to use, default is EMA
    window_len: usize,
    q_vals: VecDeque<f64>,
    out: f64,
}

impl<V, M> std::fmt::Debug for PolarizedFractalEfficiency<V, M>
where
    V: View,
    M: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "PolarizedFractalEfficiency(window_len: {}, q_vals: {:?}, out: {})",
            self.window_len, self.q_vals, self.out
        )
    }
}

/// Create a new PolarizedFractalEfficiency indicator with a given window length
/// and the default moving average
#[inline(always)]
pub fn new_final(window_len: usize) -> PolarizedFractalEfficiency<Echo, EMA<Echo>> {
    new_with_default_ma(Echo::new(), window_len)
}

/// Create a new PolarizedFractalEfficiency indicator with a chained view and a given window
/// length. The window length will also be used for the EMA
#[inline(always)]
pub fn new_with_default_ma<V: View>(
    view: V,
    window_len: usize,
) -> PolarizedFractalEfficiency<V, EMA<Echo>> {
    PolarizedFractalEfficiency::new(view, crate::ema::new_final(window_len), window_len)
}

impl<V, M> PolarizedFractalEfficiency<V, M>
where
    V: View,
    M: View,
{
    /// Create a new PolarizedFractalEfficiency indicator with a chained view, custom moving
    /// average and a window length
    #[inline]
    pub fn new(view: V, moving_average: M, window_len: usize) -> Self {
        Self {
            view,
            moving_average,
            window_len,
            q_vals: VecDeque::new(),
            out: 0.0,
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

    #[inline(always)]
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
        let mut pfe = new_final(16);
        for v in &TEST_DATA {
            pfe.update(*v);
            assert!(pfe.last() <= 1.0);
            assert!(pfe.last() >= -1.0);
        }
    }

    #[test]
    fn polarized_fractal_efficiency_plot() {
        let mut pfe = new_final(16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            pfe.update(*v);
            out.push(pfe.last());
        }
        let filename = "img/polarized_fractal_efficiency.png";
        plot_values(out, filename).unwrap();
    }
}

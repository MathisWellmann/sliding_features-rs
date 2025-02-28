//! A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
//! it is also possible to use a custom moving average instead of the default EMA in the original

use crate::View;
use getset::CopyGetters;
use num::Float;
use std::collections::VecDeque;

#[derive(Debug, Clone, CopyGetters)]
/// A PolarizedFractalEfficiency indicator with output range [-1.0 and 1.0] rather than [-100, 100]
/// it is also possible to use a custom moving average instead of the default EMA in the original
pub struct PolarizedFractalEfficiency<T, V, M> {
    view: V,
    moving_average: M, // defines which moving average to use, default is EMA
    /// The sliding window length
    #[getset(get_copy = "pub")]
    window_len: usize,
    q_vals: VecDeque<T>,
    out: Option<T>,
}

impl<T, V, M> PolarizedFractalEfficiency<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    /// Create a new PolarizedFractalEfficiency indicator with a chained view, custom moving
    /// average and a window length
    pub fn new(view: V, moving_average: M, window_len: usize) -> Self {
        Self {
            view,
            moving_average,
            window_len,
            q_vals: VecDeque::with_capacity(window_len),
            out: None,
        }
    }
}

impl<T, V, M> View<T> for PolarizedFractalEfficiency<T, V, M>
where
    V: View<T>,
    M: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        let window_len = T::from(self.window_len).expect("can convert");
        if self.q_vals.len() >= self.window_len {
            let mut s = T::zero();
            let wl: usize = self.window_len - 1;
            for i in 0..self.window_len - 2 {
                let v_0 = *self.q_vals.get(wl - i).unwrap();
                let v_1 = *self.q_vals.get(wl - i - 1).unwrap();
                s = s + ((v_0 - v_1).powi(2) + T::one()).sqrt();
            }
            let mut p =
                ((val - *self.q_vals.front().unwrap()).powi(2) + window_len.powi(2)).sqrt() / s;
            if val < *self.q_vals.get(self.window_len - 2).unwrap() {
                p = -p;
            }
            // apply a moving average
            self.moving_average.update(p);
            self.out = self.moving_average.last();
        }
    }

    fn last(&self) -> Option<T> {
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

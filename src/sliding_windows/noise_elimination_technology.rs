//! John Ehlers Noise elimination technology using kendall correlation
//! from <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>

use crate::View;
use num::Float;
use std::collections::VecDeque;

/// John Ehlers Noise elimination technology using kendall correlation
/// from <http://www.mesasoftware.com/papers/Noise%20Elimination%20Technology.pdf>
#[derive(Debug, Clone)]
pub struct NoiseEliminationTechnology<T, V> {
    view: V,
    window_len: usize,
    out: Option<T>,
    q_vals: VecDeque<T>,
}

impl<T, V> NoiseEliminationTechnology<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new NET with a chained View and window length
    pub fn new(view: V, window_len: usize) -> Self {
        NoiseEliminationTechnology {
            view,
            window_len,
            out: None,
            q_vals: VecDeque::new(),
        }
    }
}

impl<T, V> View<T> for NoiseEliminationTechnology<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.q_vals.len() >= self.window_len {
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(val);

        if self.q_vals.len() < 2 {
            return;
        }
        let mut x: Vec<T> = vec![T::zero(); self.q_vals.len()];
        let mut y: Vec<T> = vec![T::zero(); self.q_vals.len()];
        for count in 1..self.q_vals.len() {
            x[count] = *self.q_vals.get(self.q_vals.len() - count).unwrap();
            y[count] = -T::from(count).expect("can convert");
        }

        let mut num = T::zero();
        for count in 2..self.q_vals.len() {
            for k in 1..count - 1 {
                num = num - ((x[count] - x[k]).signum());
            }
        }

        let n = T::from(self.q_vals.len()).expect("can convert");
        let denom = T::from(0.5).expect("can convert") * n * (n - T::one());
        self.out = Some(num / denom);
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
    use crate::sliding_windows::MyRSI;
    use crate::test_data::TEST_DATA;

    #[test]
    fn net_my_rsi_plot() {
        let mut net = NoiseEliminationTechnology::new(MyRSI::new(Echo::new(), 16), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            net.update(*v);
            if let Some(val) = net.last() {
                out.push(val);
            }
        }
        println!("out: {:?}", out);

        let filename = "img/net_my_rsi.png";
        plot_values(out, filename).unwrap();
    }
}

//! EMA - Exponential Moving Average

use crate::View;

#[derive(Clone)]
/// EMA - Exponential Moving Average
pub struct Ema<V> {
    view: V,
    window_len: usize,
    alpha: f64,
    last_ema: f64,
    out: f64,
    n_observed_values: usize,
}

impl<V> std::fmt::Debug for Ema<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "EMA(window_len: {}, alpha: {}, last_ema: {}, out: {})",
            self.window_len, self.alpha, self.last_ema, self.out
        )
    }
}

impl<V> Ema<V>
where
    V: View,
{
    /// Create a new EMA with a chained view and a given window length
    /// and a default alpha value of 2.0
    pub fn new(view: V, window_len: usize) -> Self {
        Self::with_alpha(view, window_len, 2.0)
    }

    /// Create a new EMA with a custom alpha as well
    pub fn with_alpha(view: V, window_len: usize, alpha: f64) -> Self {
        Self {
            view,
            window_len,
            alpha,
            last_ema: 0.0,
            out: 0.0,
            n_observed_values: 0,
        }
    }
}

impl<V> View for Ema<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.n_observed_values += 1;
        let weight: f64 = self.alpha / (1.0 + self.window_len as f64);

        if self.last_ema == 0.0 {
            self.out = val;
            self.last_ema = val;
            return;
        }

        self.out = val * weight + self.last_ema * (1.0 - weight);
        self.last_ema = self.out;
    }

    fn last(&self) -> Option<f64> {
        if self.n_observed_values < self.window_len {
            return None;
        }
        Some(self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::pure_functions::Echo;
    use crate::test_data::TEST_DATA;

    #[test]
    fn ema_plot() {
        let mut ema = Ema::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            ema.update(*v);
            if let Some(ema) = ema.last() {
                out.push(ema);
            }
        }
        let filename = "img/ema.png";
        plot_values(out, filename).unwrap();
    }
}

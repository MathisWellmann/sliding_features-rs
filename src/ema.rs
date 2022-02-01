//! EMA - Exponential Moving Average

use crate::View;

#[derive(Clone)]
/// EMA - Exponential Moving Average
pub struct EMA<V> {
    view: V,
    window_len: usize,
    alpha: f64,
    last_ema: f64,
    out: f64,
}

impl<V> std::fmt::Debug for EMA<V>
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

impl<V> EMA<V>
where
    V: View,
{
    /// Create a new EMA with a chained view and a given window length
    /// and a default alpha value of 2.0
    #[inline(always)]
    pub fn new(view: V, window_len: usize) -> Self {
        Self::with_alpha(view, window_len, 2.0)
    }

    /// Create a new EMA with a custom alpha as well
    #[inline]
    pub fn with_alpha(view: V, window_len: usize, alpha: f64) -> Self {
        Self {
            view,
            window_len,
            alpha,
            last_ema: 0.0,
            out: 0.0,
        }
    }
}

impl<V> View for EMA<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        let weight: f64 = self.alpha / (1.0 + self.window_len as f64);
        if self.last_ema == 0.0 {
            self.out = val;
            self.last_ema = val;
            return;
        }
        self.out = val * weight + self.last_ema * (1.0 - weight);
        self.last_ema = self.out;
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
    use crate::Echo;

    #[test]
    fn ema_plot() {
        let mut ema = EMA::new(Echo::new(), 16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            ema.update(*v);
            out.push(ema.last());
        }
        let filename = "img/ema.png";
        plot_values(out, filename).unwrap();
    }
}

use crate::sliding_window::View;
use crate::Echo;

#[derive(Clone)]
/// EMA - Exponential Moving Average
pub struct EMA {
    view: Box<dyn View>,
    window_len: usize,
    alpha: f64,
    last_ema: f64,
    out: f64,
}

impl EMA {
    /// Create a new EMA with a chained view and a given window length
    /// and a default alpha value of 2.0
    pub fn new(view: Box<dyn View>, window_len: usize) -> Self {
        Self::with_alpha(view, window_len, 2.0)
    }

    /// Create a new EMA with a given window length
    pub fn new_final(window_len: usize) -> Self {
        Self::new(Box::new(Echo::new()), window_len)
    }

    /// Create a new EMA with a custom alpha as well
    pub fn with_alpha(view: Box<dyn View>, window_len: usize, alpha: f64) -> Self {
        Self {
            view,
            window_len,
            alpha,
            last_ema: 0.0,
            out: 0.0,
        }
    }
}

impl View for EMA {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val: f64 = self.view.last();

        let weight: f64 = self.alpha / ( 1.0 + self.window_len as f64);
        if self.last_ema == 0.0 {
            self.out = val;
            self.last_ema = val;
            return
        }
        self.out = val * weight 
            + self.last_ema * (1.0 - weight); 
        self.last_ema = self.out;
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
    fn ema_plot() {
        let mut ema = EMA::new_final(16);
        let mut out: Vec<f64> = vec![];
        for v in &TEST_DATA {
            ema.update(*v);
            out.push(ema.last());
        }
        let filename = "img/ema.png";
        plot_values(out, filename).unwrap();
    }
}

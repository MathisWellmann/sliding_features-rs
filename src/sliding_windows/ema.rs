//! EMA - Exponential Moving Average

use std::num::NonZeroUsize;

use crate::View;
use num::Float;

#[derive(Clone, Debug)]
/// EMA - Exponential Moving Average
pub struct Ema<T, V> {
    view: V,
    window_len: usize,
    alpha: T,
    last_ema: T,
    out: T,
    n_observed_values: usize,
}

impl<T, V> Ema<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new EMA with a chained view and a given window length
    /// and a default alpha value of 2.0
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Self::with_alpha(view, window_len, T::from(2.0).expect("can convert"))
    }

    /// Create a new EMA with a custom alpha as well
    pub fn with_alpha(view: V, window_len: NonZeroUsize, alpha: T) -> Self {
        Self {
            view,
            window_len: window_len.get(),
            alpha,
            last_ema: T::zero(),
            out: T::zero(),
            n_observed_values: 0,
        }
    }
}

impl<T, V> View<T> for Ema<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        self.n_observed_values += 1;
        let weight = self.alpha / (T::one() + T::from(self.window_len).expect("can convert"));

        if self.last_ema == T::zero() {
            self.out = val;
            self.last_ema = val;
            return;
        }

        self.out = val * weight + self.last_ema * (T::one() - weight);
        self.last_ema = self.out;
    }

    fn last(&self) -> Option<T> {
        if self.n_observed_values < self.window_len {
            return None;
        }
        debug_assert!(self.out.is_finite(), "value must be finite");
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
        let mut ema = Ema::new(Echo::new(), NonZeroUsize::new(16).unwrap());
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

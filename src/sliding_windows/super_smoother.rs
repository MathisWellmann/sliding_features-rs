use getset::CopyGetters;
use num::Float;
use std::f64::consts::PI;

use crate::View;

/// John Ehlers SuperSmoother filter
/// from <https://www.mesasoftware.com/papers/PredictiveIndicatorsForEffectiveTrading%20Strategies.pdf>
#[derive(Debug, Clone, CopyGetters)]
pub struct SuperSmoother<T, V> {
    view: V,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: usize,
    i: usize,
    c1: T,
    c2: T,
    c3: T,
    /// filter value at current step
    filt: T,
    // filter one step ago
    filt_1: T,
    // filter two steps ago
    filt_2: T,
    last_val: T,
}

impl<T, V> SuperSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance of the SuperSmoother with a chained View
    pub fn new(view: V, window_len: usize) -> Self {
        let wl = T::from(window_len).expect("can convert");
        let a1 =
            (-T::from(1.414).expect("can convert") * T::from(PI).expect("can convert") / wl).exp();
        // NOTE: 4.4422 is radians of 1.414 * 180 degrees
        let b1 = T::from(2.0).expect("can convert")
            * a1
            * (T::from(4.4422).expect("can convert") / wl).cos();
        let c2 = b1;
        let c3 = -a1 * a1;

        Self {
            view,
            window_len,
            i: 0,
            c1: T::one() - c2 - c3,
            c2,
            c3,
            filt: T::zero(),
            filt_1: T::zero(),
            filt_2: T::zero(),
            last_val: T::zero(),
        }
    }
}

impl<T, V> View<T> for SuperSmoother<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        self.filt = self.c1 * (val + self.last_val) / T::from(2.0).expect("can convert")
            + (self.c2 * self.filt_1)
            + (self.c3 * self.filt_2);
        self.filt_2 = self.filt_1;
        self.filt_1 = self.filt;
        self.last_val = val;
        self.i += 1;
    }

    #[inline]
    fn last(&self) -> Option<T> {
        // NOTE: filter only kicks in after warmup steps are done
        if self.i < self.window_len {
            None
        } else {
            Some(self.filt)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{plot::plot_values, pure_functions::Echo, test_data::TEST_DATA};

    use super::*;

    #[test]
    fn super_smoother_plot() {
        let mut ss = SuperSmoother::new(Echo::new(), 20);
        let mut out: Vec<f64> = Vec::with_capacity(TEST_DATA.len());
        for v in &TEST_DATA {
            ss.update(*v);
            if let Some(val) = ss.last() {
                out.push(val);
            }
        }
        let filename = "img/super_smoother.png";
        plot_values(out, filename).unwrap();
    }
}

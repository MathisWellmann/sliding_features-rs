use std::num::NonZeroUsize;

use getset::CopyGetters;
use num::Float;

use super::SuperSmoother;
use crate::{
    View,
    pure_functions::Echo,
};

/// Roofing Filter by John Ehlers
/// From paper: <http://www.stockspotter.com/files/PredictiveIndicators.pdf>
#[derive(Debug, Clone, CopyGetters)]
pub struct RoofingFilter<T, V> {
    view: V,
    super_smoother: SuperSmoother<T, Echo<T>>,
    /// The sliding window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    i: usize,
    alpha_1: T,
    // previous value
    val_1: T,
    // value from 2 steps ago
    val_2: T,
    // high pass filter value of previous step
    hp_1: T,
    // high pass filter value from two steps ago
    hp_2: T,
}

impl<T, V> RoofingFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a Roofing Filter with a chained view
    pub fn new(
        view: V,
        window_len_low_pass: NonZeroUsize,
        super_smoother_len_high_pass: NonZeroUsize,
    ) -> Self {
        // NOTE: 4.4422 radians from  0.707 * 360 degrees
        let wl = T::from(window_len_low_pass.get()).expect("can convert");
        let f = T::from(4.4422).expect("Can convert");
        let alpha_1 = ((f / wl).cos() + (f / wl).sin() - T::one()) / (f / wl).cos();

        RoofingFilter {
            view,
            super_smoother: SuperSmoother::new(Echo::new(), super_smoother_len_high_pass),
            window_len: window_len_low_pass,
            i: 0,
            alpha_1,
            val_1: T::zero(),
            val_2: T::zero(),
            hp_1: T::zero(),
            hp_2: T::zero(),
        }
    }
}

impl<T, V> View<T> for RoofingFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        let two = T::from(2.0).expect("can convert");
        let hp = (T::one() - self.alpha_1 / two).powi(2) * (val - two * self.val_1 + self.val_2)
            + two * (T::one() - self.alpha_1) * self.hp_1
            - (T::one() - self.alpha_1).powi(2) * self.hp_2;
        self.hp_2 = self.hp_1;
        self.hp_1 = hp;

        self.val_2 = self.val_1;
        self.val_1 = val;

        if self.i > self.window_len.get() {
            // to avoid weird output, only update, once warmup stage is done
            self.super_smoother.update(hp);
        }
        self.i += 1;
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        self.super_smoother.last().inspect(|v| {
            debug_assert!(v.is_finite(), "value must be finite");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plot::plot_values,
        test_data::TEST_DATA,
    };

    #[test]
    fn roofing_filter_plot() {
        let mut rf = RoofingFilter::new(
            Echo::new(),
            NonZeroUsize::new(48).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            rf.update(*v);
            if let Some(val) = rf.last() {
                out.push(val);
            }
        }
        let filename = "img/roofing_filter.png";
        plot_values(out, filename).unwrap();
    }
}

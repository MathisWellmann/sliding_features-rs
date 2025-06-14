//! ALMA - Arnaud Legoux Moving Average
//! reference: <https://forex-station.com/download/file.php?id=3326661&sid=d6b440bfbba5e1905b4c75188c2797ce>

use getset::CopyGetters;
use num::Float;
use std::{collections::VecDeque, num::NonZeroUsize};

use crate::View;

/// ALMA - Arnaud Legoux Moving Average
/// reference: <https://forex-station.com/download/file.php?id=3326661&sid=d6b440bfbba5e1905b4c75188c2797ce>
#[derive(Clone, Debug, CopyGetters)]
pub struct Alma<T, V> {
    view: V,
    /// The configured window length.
    #[getset(get_copy = "pub")]
    window_len: NonZeroUsize,
    wtd_sum: T,
    cum_wt: T,
    m: T,
    s: T,
    q_vals: VecDeque<T>,
    q_wtd: VecDeque<T>,
    q_out: VecDeque<T>,
}

impl<T, V> Alma<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new Arnaud Legoux Moving Average with a chained View
    /// and a given window length
    pub fn new(view: V, window_len: NonZeroUsize) -> Self {
        Alma::new_custom(
            view,
            window_len,
            T::from(6.0).expect("Can convert"),
            T::from(0.85).expect("Can convert"),
        )
    }

    /// Create a Arnaud Legoux Moving Average with custom parameters
    pub fn new_custom(view: V, window_len: NonZeroUsize, sigma: T, offset: T) -> Self {
        let wl = T::from(window_len.get()).expect("can convert");
        let m = offset * (wl + T::one());
        let s = wl / sigma;
        Alma {
            view,
            window_len,
            m,
            s,
            wtd_sum: T::zero(),
            cum_wt: T::zero(),
            q_vals: VecDeque::with_capacity(window_len.get()),
            q_wtd: VecDeque::with_capacity(window_len.get()),
            q_out: VecDeque::with_capacity(window_len.get()),
        }
    }
}

impl<T, V> View<T> for Alma<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        // first, apply the internal view update
        self.view.update(val);
        let Some(val) = self.view.last() else { return };
        debug_assert!(val.is_finite(), "value must be finite");

        if self.q_vals.len() >= self.window_len.get() {
            let old_val = self.q_vals.front().unwrap();
            let old_wtd = self.q_wtd.front().unwrap();
            self.wtd_sum = self.wtd_sum - *old_wtd * *old_val;
            self.cum_wt = self.cum_wt - *old_wtd;

            self.q_vals.pop_front();
            self.q_wtd.pop_front();
            self.q_out.pop_front();
        }
        let count = T::from(self.q_vals.len()).expect("can convert");
        let wtd = (-(count - self.m).powi(2)
            / (T::from(2.0).expect("can convert") * self.s * self.s))
            .exp();
        self.wtd_sum = self.wtd_sum + wtd * val;
        self.cum_wt = self.cum_wt + wtd;

        self.q_vals.push_back(val);
        self.q_wtd.push_back(wtd);

        let ala = self.wtd_sum / self.cum_wt;
        debug_assert!(ala.is_finite(), "value must be finite");
        self.q_out.push_back(ala);
    }

    fn last(&self) -> Option<T> {
        self.q_out.back().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::{plot::plot_values, pure_functions::Echo};
    use rand::{rng, Rng};

    #[test]
    fn alma() {
        let mut rng = rng();
        let mut alma = Alma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        for _ in 0..1_000_000 {
            let v = rng.random::<f64>();
            alma.update(v);
            let last = alma.last().unwrap();

            assert!(last >= 0.0);
            assert!(last <= 1.0);
        }
    }

    #[test]
    fn alma_plot() {
        let mut alma = Alma::new(Echo::new(), NonZeroUsize::new(16).unwrap());
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            alma.update(*v);
            out.push(alma.last().unwrap())
        }
        let filename = "img/alma.png";
        plot_values(out, filename).unwrap();
    }
}

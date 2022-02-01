//! ALMA - Arnaud Legoux Moving Average
//! reference: https://forex-station.com/download/file.php?id=3326661&sid=d6b440bfbba5e1905b4c75188c2797ce

use std::collections::VecDeque;

use crate::View;

/// ALMA - Arnaud Legoux Moving Average
/// reference: https://forex-station.com/download/file.php?id=3326661&sid=d6b440bfbba5e1905b4c75188c2797ce
#[derive(Clone)]
pub struct ALMA<V> {
    view: V,
    window_len: usize,
    wtd_sum: f64,
    cum_wt: f64,
    m: f64,
    s: f64,
    q_vals: VecDeque<f64>,
    q_wtd: VecDeque<f64>,
    q_out: VecDeque<f64>,
}

impl<V> std::fmt::Debug for ALMA<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "ALMA(window_len: {}, wtd_sum: {}, cum_wt: {}, m: {}, s: {}, q_vals: {:?}, q_wtd: {:?}, q_out: {:?})",
               self.window_len, self.wtd_sum, self.cum_wt, self.m, self.s, self.q_vals, self.q_wtd, self.q_out)
    }
}

impl<V> ALMA<V>
where
    V: View,
{
    /// Create a new Arnaud Legoux Moving Average with a chained View
    /// and a given window length
    #[inline(always)]
    pub fn new(view: V, window_len: usize) -> Self {
        ALMA::new_custom(view, window_len, 6.0, 0.85)
    }

    /// Create a Arnaud Legoux Moving Average with custom parameters
    #[inline]
    pub fn new_custom(view: V, window_len: usize, sigma: f64, offset: f64) -> Self {
        let m = offset * (window_len as f64 + 1.0);
        let s = window_len as f64 / sigma;
        ALMA {
            view,
            window_len,
            m,
            s,
            wtd_sum: 0.0,
            cum_wt: 0.0,
            q_vals: VecDeque::new(),
            q_wtd: VecDeque::new(),
            q_out: VecDeque::new(),
        }
    }
}

impl<V> View for ALMA<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        // first, apply the internal view update
        self.view.update(val);
        let val = self.view.last();

        if self.q_vals.len() >= self.window_len {
            let old_val = self.q_vals.front().unwrap();
            let old_wtd = self.q_wtd.front().unwrap();
            self.wtd_sum -= old_wtd * old_val;
            self.cum_wt -= *old_wtd;

            self.q_vals.pop_front();
            self.q_wtd.pop_front();
            self.q_out.pop_front();
        }
        let count = self.q_vals.len() as f64;
        let wtd = (-(count - self.m).powi(2) / (2.0 * self.s * self.s)).exp();
        self.wtd_sum += wtd * val;
        self.cum_wt += wtd;

        self.q_vals.push_back(val);
        self.q_wtd.push_back(wtd);

        let ala = self.wtd_sum / self.cum_wt;
        self.q_out.push_back(ala);
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        return *self.q_out.back().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::{plot::plot_values, Echo};
    use rand::{thread_rng, Rng};

    #[test]
    fn alma() {
        let mut rng = thread_rng();
        let mut alma = ALMA::new(Echo::new(), 16);
        for _ in 0..1_000_000 {
            let v = rng.gen::<f64>();
            alma.update(v);
            let last = alma.last();

            assert!(last >= 0.0);
            assert!(last <= 1.0);
        }
    }

    #[test]
    fn alma_plot() {
        let mut alma = ALMA::new(Echo::new(), 16);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            alma.update(*v);
            out.push(alma.last())
        }
        let filename = "img/alma.png";
        plot_values(out, filename).unwrap();
    }
}

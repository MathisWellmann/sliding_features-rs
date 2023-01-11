//! John Ehlers Laguerre Filter
//! from: <http://mesasoftware.com/papers/TimeWarp.pdf>

use crate::View;

/// John Ehlers Laguerre Filter
/// from: <http://mesasoftware.com/papers/TimeWarp.pdf>
#[derive(Clone)]
pub struct LaguerreFilter<V> {
    view: V,
    gamma: f64,
    l0s: Vec<f64>,
    l1s: Vec<f64>,
    l2s: Vec<f64>,
    l3s: Vec<f64>,
    filts: Vec<f64>,
    init: bool,
}

impl<V> std::fmt::Debug for LaguerreFilter<V>
where
    V: View,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "LaguerreFilter(gamma: {}, l0s: {:?}, l1s: {:?}, l2s: {:?}, l3s: {:?}, filts: {:?}, init: {})",
               self.gamma, self.l0s, self.l1s, self.l2s, self.l3s, self.filts, self.init)
    }
}

impl<V> LaguerreFilter<V>
where
    V: View,
{
    /// Create a new LaguerreFilter with a chained View
    /// and a gamma parameter
    #[inline]
    pub fn new(view: V, gamma: f64) -> Self {
        LaguerreFilter {
            view,
            gamma,
            l0s: vec![],
            l1s: vec![],
            l2s: vec![],
            l3s: vec![],
            filts: vec![],
            init: true,
        }
    }
}

impl<V> View for LaguerreFilter<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if self.init {
            self.init = false;
            self.l0s.push(val);
            self.l1s.push(val);
            self.l2s.push(val);
            self.l3s.push(val);
            self.filts
                .push((self.l0s[0] + 2.0 * self.l1s[0] + 2.0 * self.l2s[0] + self.l3s[0]) / 6.0);
            return;
        }
        self.l0s
            .push((1.0 - self.gamma) * val + self.gamma * self.l0s[self.l0s.len() - 1]);
        self.l1s.push(
            -self.gamma * self.l0s[self.l0s.len() - 1]
                + self.l0s[self.l0s.len() - 2]
                + self.gamma * self.l1s[self.l1s.len() - 1],
        );
        self.l2s.push(
            -self.gamma * self.l1s[self.l1s.len() - 1]
                + self.l1s[self.l1s.len() - 2]
                + self.gamma * self.l2s[self.l2s.len() - 1],
        );
        self.l3s.push(
            -self.gamma * self.l2s[self.l2s.len() - 1]
                + self.l2s[self.l2s.len() - 2]
                + self.gamma * self.l3s[self.l3s.len() - 1],
        );
        self.filts.push(
            (self.l0s[self.l0s.len() - 1]
                + 2.0 * self.l1s[self.l1s.len() - 1]
                + 2.0 * self.l2s[self.l2s.len() - 1]
                + self.l3s[self.l3s.len() - 1])
                / 6.0,
        );
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.filts[self.filts.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::TEST_DATA;
    use crate::{plot::plot_values, Echo};
    use rand::{thread_rng, Rng};

    #[test]
    fn laguerre_filter() {
        let mut laguerre = LaguerreFilter::new(Echo::new(), 0.8);
        let mut rng = thread_rng();
        for _ in 0..10_000 {
            let v = rng.gen::<f64>();

            laguerre.update(v);
            let last = laguerre.last();

            assert!(last <= 1.0);
            assert!(last >= 0.0);
        }
    }

    #[test]
    fn laguerre_filter_plot() {
        let mut laguerre = LaguerreFilter::new(Echo::new(), 0.8);
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            laguerre.update(*v);
            out.push(laguerre.last());
        }
        let filename = "img/laguerre_filter.png";
        plot_values(out, filename).unwrap();
    }
}

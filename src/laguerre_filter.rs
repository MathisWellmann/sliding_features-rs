//! John Ehlers Laguerre Filter
//! from: <http://mesasoftware.com/papers/TimeWarp.pdf>

use crate::View;

/// John Ehlers Laguerre Filter
/// from: <http://mesasoftware.com/papers/TimeWarp.pdf>
#[derive(Debug, Clone)]
pub struct LaguerreFilter<V>
where
    V: View,
{
    view: V,
    gamma: f64,
    l0s: Vec<f64>,
    l1s: Vec<f64>,
    l2s: Vec<f64>,
    l3s: Vec<f64>,
    filts: Vec<f64>,
}

impl<V> LaguerreFilter<V>
where
    V: View,
{
    /// Create a new LaguerreFilter with a chained View
    /// and a gamma parameter
    pub fn new(view: V, gamma: f64) -> Self {
        LaguerreFilter {
            view,
            gamma,
            l0s: Vec::new(),
            l1s: Vec::new(),
            l2s: Vec::new(),
            l3s: Vec::new(),
            filts: Vec::new(),
        }
    }
}

impl<V> View for LaguerreFilter<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if self.l0s.is_empty() {
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

    fn last(&self) -> Option<f64> {
        self.filts.last().copied()
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
            let last = laguerre.last().unwrap();

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
            out.push(laguerre.last().unwrap());
        }
        let filename = "img/laguerre_filter.png";
        plot_values(out, filename).unwrap();
    }
}

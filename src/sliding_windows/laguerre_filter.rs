//! John Ehlers Laguerre Filter
//! from: <http://mesasoftware.com/papers/TimeWarp.pdf>

use num::Float;

use crate::View;

/// John Ehlers Laguerre Filter
/// from: <http://mesasoftware.com/papers/TimeWarp.pdf>
#[derive(Debug, Clone)]
pub struct LaguerreFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    view: V,
    gamma: T,
    // TODO: use `VecDeque` and remove unused values again.
    l0s: Vec<T>,
    l1s: Vec<T>,
    l2s: Vec<T>,
    l3s: Vec<T>,
    filts: Vec<T>,
}

impl<T, V> LaguerreFilter<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new LaguerreFilter with a chained View
    /// and a gamma parameter
    pub fn new(view: V, gamma: T) -> Self {
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

impl<T, V> View<T> for LaguerreFilter<T, V>
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
        if self.l0s.is_empty() {
            self.l0s.push(val);
            self.l1s.push(val);
            self.l2s.push(val);
            self.l3s.push(val);
            self.filts.push(
                (self.l0s[0] + two * self.l1s[0] + two * self.l2s[0] + self.l3s[0])
                    / T::from(6.0).expect("can convert"),
            );
            return;
        }
        self.l0s
            .push((T::one() - self.gamma) * val + self.gamma * self.l0s[self.l0s.len() - 1]);
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
        let out = (self.l0s[self.l0s.len() - 1]
            + two * self.l1s[self.l1s.len() - 1]
            + two * self.l2s[self.l2s.len() - 1]
            + self.l3s[self.l3s.len() - 1])
            / T::from(6.0).expect("can convert");
        debug_assert!(out.is_finite(), "value must be finite");
        self.filts.push(out);
    }

    fn last(&self) -> Option<T> {
        self.filts.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        Rng,
        rng,
    };

    use super::*;
    use crate::{
        plot::plot_values,
        pure_functions::Echo,
        test_data::TEST_DATA,
    };

    #[test]
    fn laguerre_filter() {
        let mut laguerre = LaguerreFilter::new(Echo::new(), 0.8);
        let mut rng = rng();
        for _ in 0..10_000 {
            let v = rng.random::<f64>();

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

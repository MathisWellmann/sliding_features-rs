use crate::View;

/// Lower Than or Equal filter,
/// which only allows values lower than the specified clipping point through
pub struct LTE<V> {
    view: V,
    clipping_value: f64,
    out: f64,
}

impl<V> LTE<V>
where
    V: View,
{
    /// Create a new instance with a chained View
    #[inline(always)]
    pub fn new(view: V, clipping_value: f64) -> Self {
        Self {
            view,
            clipping_value,
            out: 0.0,
        }
    }
}

impl<V> View for LTE<V>
where
    V: View,
{
    #[inline]
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if val <= self.clipping_value {
            self.out = val;
        } else {
            self.out = self.clipping_value;
        }
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use crate::Echo;

    use super::*;

    #[test]
    fn lte() {
        let mut lte = LTE::new(Echo::new(), 1.0);
        lte.update(0.5);
        assert_eq!(lte.last(), 0.5);
        lte.update(1.5);
        assert_eq!(lte.last(), 1.0);
    }
}

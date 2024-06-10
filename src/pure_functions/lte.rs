use crate::View;

/// Lower Than or Equal filter,
/// which only allows values lower than the specified clipping point through
#[derive(Debug, Clone)]
pub struct LTE<V> {
    view: V,
    clipping_value: f64,
    out: Option<f64>,
}

impl<V> LTE<V>
where
    V: View,
{
    /// Create a new instance with a chained View
    pub fn new(view: V, clipping_value: f64) -> Self {
        Self {
            view,
            clipping_value,
            out: None,
        }
    }
}

impl<V> View for LTE<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if val <= self.clipping_value {
            self.out = Some(val);
        } else {
            self.out = Some(self.clipping_value);
        }
    }

    fn last(&self) -> Option<f64> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use crate::pure_functions::Echo;

    use super::*;

    #[test]
    fn lte() {
        let mut lte = LTE::new(Echo::new(), 1.0);
        lte.update(0.5);
        assert_eq!(lte.last().unwrap(), 0.5);
        lte.update(1.5);
        assert_eq!(lte.last().unwrap(), 1.0);
    }
}

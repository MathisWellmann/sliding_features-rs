use crate::View;
use num::Float;

/// Lower Than or Equal filter,
/// which only allows values lower than the specified clipping point through
#[derive(Debug, Clone)]
pub struct LTE<T, V> {
    view: V,
    clipping_value: T,
    out: Option<T>,
}

impl<T, V> LTE<T, V>
where
    V: View<T>,
    T: Float,
{
    /// Create a new instance with a chained View
    pub fn new(view: V, clipping_value: T) -> Self {
        Self {
            view,
            clipping_value,
            out: None,
        }
    }
}

impl<T, V> View<T> for LTE<T, V>
where
    V: View<T>,
    T: Float,
{
    fn update(&mut self, val: T) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if val <= self.clipping_value {
            self.out = Some(val);
        } else {
            self.out = Some(self.clipping_value);
        }
    }

    fn last(&self) -> Option<T> {
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

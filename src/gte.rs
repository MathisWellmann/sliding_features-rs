use crate::View;

/// Greater Than or Equal
/// Will allow values >= clipping_point tough and clip other values to the clipping point
#[derive(Debug, Clone)]
pub struct GTE<V> {
    view: V,
    clipping_point: f64,
    out: f64,
}

impl<V> GTE<V>
where
    V: View,
{
    /// Create a new instance with a chained View and a given clipping point
    #[inline(always)]
    pub fn new(view: V, clipping_point: f64) -> Self {
        Self {
            view,
            clipping_point,
            out: 0.0,
        }
    }
}

impl<V> View for GTE<V>
where
    V: View,
{
    #[inline]
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        if val >= self.clipping_point {
            self.out = val;
        } else {
            self.out = self.clipping_point;
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
    fn gte() {
        let mut gte = GTE::new(Echo::new(), 1.0);
        gte.update(2.0);
        assert_eq!(gte.last(), 2.0);
        gte.update(0.5);
        assert_eq!(gte.last(), 1.0);
    }
}

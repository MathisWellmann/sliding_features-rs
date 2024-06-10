use crate::View;

/// Greater Than or Equal
/// Will allow values >= clipping_point through and clip other values to the clipping point
#[derive(Debug, Clone)]
pub struct GTE<V> {
    view: V,
    clipping_point: f64,
    out: Option<f64>,
}

impl<V> GTE<V>
where
    V: View,
{
    /// Create a new instance with a chained View and a given clipping point
    pub fn new(view: V, clipping_point: f64) -> Self {
        Self {
            view,
            clipping_point,
            out: None,
        }
    }
}

impl<V> View for GTE<V>
where
    V: View,
{
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let Some(val) = self.view.last() else { return };

        if val >= self.clipping_point {
            self.out = Some(val);
        } else {
            self.out = Some(self.clipping_point);
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
    fn gte() {
        let mut gte = GTE::new(Echo::new(), 1.0);
        gte.update(2.0);
        assert_eq!(gte.last().unwrap(), 2.0);
        gte.update(0.5);
        assert_eq!(gte.last().unwrap(), 1.0);
    }
}

use crate::{Echo, View};

#[derive(Clone)]
/// Simply multiply the output of View by a certain number
pub struct Multiplier {
    view: Box<dyn View>,
    multiplier: f64,
    out: f64,
}

impl std::fmt::Debug for Multiplier {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "Multiplier(multiplier: {}, out: {})", self.multiplier, self.out)
    }
}

impl Multiplier {
    /// Create a new multiplier with a chained view and a given value
    pub fn new(view: Box<dyn View>, multiplier: f64) -> Box<Self> {
        Box::new(Self {
            view,
            multiplier,
            out: 0.0,
        })
    }

    /// Create a new multiplier with a given multiplier value
    pub fn new_final(multiplier: f64) -> Box<Self> {
        Self::new(Echo::new(), multiplier)
    }
}

impl View for Multiplier {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let val = self.view.last();

        self.out = val * self.multiplier;
    }

    fn last(&self) -> f64 {
        self.out
    }
}

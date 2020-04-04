use super::sliding_window::View;

pub struct Echo {
    out: f64,
}

pub fn new() -> Echo {
    return Echo{
        out: 0.0,
    }
}

impl View for Echo {
    fn update(&mut self, val: f64) {
        self.out = val;
    }

    fn last(&self) -> f64 {
        return self.out;
    }
}

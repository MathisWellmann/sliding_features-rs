use super::sliding_window::View;

#[derive(Clone)]
/// Echo always return the last value just like an echo
pub struct Echo {
    out: f64,
}

impl Echo {
    /// Create a new Echo View
    pub fn new() -> Box<Echo> {
        Box::new(Echo { out: 0.0 })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::plot_values;
    use crate::test_data::TEST_DATA;

    #[test]
    fn echo_plot() {
        let mut echo = Echo::new();
        let mut out: Vec<f64> = Vec::new();
        for v in &TEST_DATA {
            echo.update(*v);
            out.push(echo.last());
        }
        let filename = "img/echo.png";
        plot_values(out, filename).unwrap();
    }
}

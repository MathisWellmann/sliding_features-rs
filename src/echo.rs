//! Echo always return the last value just like an echo

use super::View;

#[derive(Default, Clone, Debug)]
/// Echo always return the last value just like an echo
pub struct Echo {
    out: f64,
}

impl Echo {
    /// Create a new Echo View
    #[inline(always)]
    pub fn new() -> Echo {
        Echo { out: 0.0 }
    }
}

impl View for Echo {
    #[inline(always)]
    fn update(&mut self, val: f64) {
        self.out = val;
    }

    #[inline(always)]
    fn last(&self) -> f64 {
        self.out
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

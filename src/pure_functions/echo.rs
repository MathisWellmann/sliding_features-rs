//! Echo always return the last value just like an echo

use crate::View;

#[derive(Default, Clone, Debug)]
/// Echo always return the last value just like an echo
pub struct Echo {
    out: Option<f64>,
}

impl Echo {
    /// Create a new Echo View
    #[inline(always)]
    pub fn new() -> Echo {
        Echo { out: None }
    }
}

impl View for Echo {
    fn update(&mut self, val: f64) {
        self.out = Some(val);
    }

    fn last(&self) -> Option<f64> {
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
            out.push(echo.last().unwrap());
        }
        let filename = "img/echo.png";
        plot_values(out, filename).unwrap();
    }
}

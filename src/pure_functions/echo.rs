//! Echo always return the last value just like an echo

use crate::View;

#[derive(Default, Clone, Debug)]
/// Echo always return the last value just like an echo
pub struct Echo<T> {
    out: Option<T>,
}

impl<T> Echo<T> {
    /// Create a new Echo View
    #[inline(always)]
    pub fn new() -> Echo<T> {
        Echo { out: None }
    }
}

impl<T: num::Float> View<T> for Echo<T> {
    fn update(&mut self, val: T) {
        debug_assert!(val.is_finite(), "value must be finite");
        self.out = Some(val);
    }

    fn last(&self) -> Option<T> {
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

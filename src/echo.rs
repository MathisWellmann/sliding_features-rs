use super::sliding_window::View;

pub struct Echo {
    out: f64,
}

impl Echo {
    pub fn new() -> Echo {
        return Echo{
            out: 0.0,
        }
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
mod tests{
    use super::*;
    extern crate rust_timeseries_generator;
    use rust_timeseries_generator::{plt, gaussian_process};

    # [test]
    fn test_echo_graph() {
        let vals = gaussian_process::gen(1024, 100.0);
        let mut echo = Echo::new();
        let mut out: Vec<f64> = Vec::new();
        for v in &vals {
            echo.update(*v);
            out.push(echo.last());
        }
        let filename = "img/echo.png";
        plt::plt(out, filename).unwrap();
    }
}
use super::sliding_window::View;

pub struct RoofingFilter {
    val1: f64,
    val2: f64,
    hps0: f64,
    hps1: f64,
    hps2: f64,
    filt0: f64,
    filt1: f64,
    filt2: f64,
}

pub fn new() -> RoofingFilter {
    return RoofingFilter{
        val1: 0.0,
        val2: 0.0,
        hps0: 0.0,
        hps1: 0.0,
        hps2: 0.0,
        filt0: 0.0,
        filt1: 0.0,
        filt2: 0.0,
    }
}

impl View for RoofingFilter {
    fn update(&mut self, val: f64) {

        self.hps2 = self.hps1;
        self.hps1 = self.hps0;
        self.hps0 = 0.36134756541
            * (val - 2.0 * self.val1 + self.val2)
            + 0.40448768902 * self.hps1
            - 0.0409025726385 * self.hps2;

        self.val2 = self.val1;
        self.val1 = val;

        // smooth with a super smoother
        let b1 = 1.16265311587;
        let c3 = -0.411295887559;
        let c1 = 0.124321385845;

        self.filt2 = self.filt1;
        self.filt1 = self.filt0;
        self.filt0 = c1 * (self.hps0 + self.hps1)
            * b1 * self.filt1
            + c3 * self.filt2;
    }

    fn last(&self) -> f64 {
        return self.filt0;
    }
}

#[cfg(test)]
mod tests {
    extern crate rust_timeseries_generator;
    use self::rust_timeseries_generator::gaussian_process::gen;
    use self::rust_timeseries_generator::plt;
    use super::*;

    #[test]
    fn graph_roofing_filter() {
        let vals = gen(1024, 100.0);
        let mut rf = new();
        let mut out: Vec<f64> = Vec::new();
        for i in 0..vals.len() {
            rf.update(vals[i]);
            out.push(rf.last());
        }
        let filename = "img/roofing_filter.png";
        plt::plt(out, filename);
    }
}

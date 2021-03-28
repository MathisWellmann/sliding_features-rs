use std::collections::VecDeque;

use super::sliding_window::View;
use crate::Echo;

/// A sliding High - Low Normalizer
#[derive(Clone)]
pub struct HLNormalizer {
    view: Box<dyn View>,
    window_len: usize,
    q_vals: VecDeque<f64>,
    min: f64,
    max: f64,
    last: f64,
    init: bool,
}

impl HLNormalizer {
    /// Create a new HLNormalizer with a chained View
    /// and a given sliding window length
    pub fn new(view: Box<dyn View>, window_len: usize) -> Box<Self> {
        Box::new(HLNormalizer {
            view,
            window_len,
            q_vals: VecDeque::new(),
            min: 0.0,
            max: 0.0,
            last: 0.0,
            init: true,
        })
    }

    /// Create a new HLNormalizer with a given window length
    pub fn new_final(window_len: usize) -> Box<Self> {
        Self::new(Echo::new(), window_len)
    }
}

pub fn extent_queue(q: &VecDeque<f64>) -> (f64, f64) {
    let mut min: &f64 = q.front().unwrap();
    let mut max: &f64 = q.front().unwrap();

    for i in 0..q.len() {
        let val = q.get(i).unwrap();
        if val > max {
            max = val;
        }
        if val < min {
            min = val;
        }
    }
    return (*min, *max);
}

impl View for HLNormalizer {
    fn update(&mut self, val: f64) {
        self.view.update(val);
        let view_last = self.view.last();

        if self.init {
            self.init = false;
            self.min = view_last;
            self.max = view_last;
            self.last = view_last;
        }
        if self.q_vals.len() >= self.window_len {
            let old = *self.q_vals.front().unwrap();
            if old <= self.min || old >= self.max {
                let (min, max) = extent_queue(&self.q_vals);
                self.min = min;
                self.max = max;
            }
            self.q_vals.pop_front();
        }
        self.q_vals.push_back(view_last);
        if view_last > self.max {
            self.max = view_last;
        }
        if view_last < self.min {
            self.min = view_last;
        }
        self.last = view_last;
    }

    fn last(&self) -> f64 {
        if self.last == self.min && self.last == self.max {
            return 0.0;
        }
        return -1.0 + (((self.last - self.min) * 2.0) / (self.max - self.min));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::center_of_gravity::CenterOfGravity;
    use crate::cyber_cycle::CyberCycle;
    use crate::plot::plot_values;
    use crate::re_flex::ReFlex;
    use crate::roc::ROC;
    use crate::rsi::RSI;
    use crate::test_data::TEST_DATA;
    use crate::trend_flex::TrendFlex;

    #[test]
    fn normalizer() {
        let mut n = HLNormalizer::new_final(16);
        for v in &TEST_DATA {
            n.update(*v);
            let last = n.last();
            assert!(last <= 1.0);
            assert!(last >= -1.0);
        }
    }

    #[test]
    fn normalizer_center_of_gravity_plot() {
        let window_len = 16;
        let cgo = CenterOfGravity::new_final(window_len);
        let mut n = HLNormalizer::new(cgo, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/center_of_gravity_normalized.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn normalizer_cyber_cycle_plot() {
        let window_len = 16;
        let cc = CyberCycle::new_final(window_len);
        let mut n = HLNormalizer::new(cc, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/cyber_cycle_normalized.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn normalizer_re_flex_plot() {
        let window_len = 16;
        let rf = ReFlex::new_final(window_len);
        let mut n = HLNormalizer::new(rf, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/re_flex_normalized.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn normalizer_roc_plot() {
        let window_len = 16;
        let r = ROC::new_final(window_len);
        let mut n = HLNormalizer::new(r, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/roc_normalized.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn normalizer_rsi_plot() {
        let window_len = 16;
        let r = RSI::new_final(window_len);
        let mut n = HLNormalizer::new(r, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/rsi_normalized.png";
        plot_values(out, filename).unwrap();
    }

    #[test]
    fn normalizer_trend_flex_plot() {
        let window_len = 16;
        let tf = TrendFlex::new_final(window_len);
        let mut n = HLNormalizer::new(tf, window_len);
        let mut out: Vec<f64> = Vec::new();

        for v in &TEST_DATA {
            n.update(*v);
            out.push(n.last());
        }

        let filename = "img/trend_flex_normalized.png";
        plot_values(out, filename).unwrap();
    }
}

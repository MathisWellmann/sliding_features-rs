use crate::sliding_window::View;
use std::collections::VecDeque;


#[derive(Debug, Clone)]
pub struct ShannonEntropyDiscrete {
    window_len: usize,
    init: bool,
    last_val: f64,
    sum: f64,
    q_incr: VecDeque<f64>,
    gain_counter: f64,
}

impl ShannonEntropyDiscrete {
    pub fn new(window_len: usize) -> ShannonEntropyDiscrete {
        return ShannonEntropyDiscrete {
            window_len,
            init: true,
            last_val: 0.0,
            sum: 0.0,
            q_incr: VecDeque::with_capacity(window_len),
            gain_counter: 0.0,
        }
    }
}

impl View for ShannonEntropyDiscrete {
    fn update(&mut self, val: f64) {
        if self.init {
            self.last_val = val;
            self.init = false;
        }

        if self.q_incr.len() >= self.window_len {
            let old_incr = *self.q_incr.front().unwrap();
            self.sum -= old_incr;
            self.q_incr.pop_front();
        }

        let p_val: f64 = self.gain_counter / self.q_incr.len() as f64;
        let mut incr: f64 = p_val * (p_val).log2();
        if incr.is_nan() {
            incr = 0.0;
        }
        self.q_incr.push_back(incr);
        self.sum += incr;

        self.last_val = val;
    }

    fn last(&self) -> f64 {
        -self.sum
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn shannon_entropy_discrete() {
        let mut se = ShannonEntropyDiscrete::new(4);

        se.update(1.0);
        println!("se: {:?}", se);
        se.update(0.5);
        println!("se: {:?}", se);
        se.update(1.0);
        println!("se: {:?}", se);
        se.update(0.5);

        assert_eq!(se.last(), 0.5);

        assert!(false);
    }
}
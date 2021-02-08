// use crate::View;
//
// pub struct ZeroLag {
//     alpha: f64,
// }
//
// impl ZeroLag {
//     pub fn new(window_len: usize) -> Self {
//         Self {
//             alpha: 2.0 / (window_len as f64 + 1.0),
//         }
//     }
// }
//
// impl View for ZeroLag {
//     fn update(&mut self, val: f64) {
//         unimplemented!()
//     }
//
//     fn last(&self) -> f64 {
//         let last_ec: f64 = self.q_ecs.get(self.q_ecs.len() - 2).unwrap();
//         self.alpha * (ema + best_gain * (close - last_ec)) + ( 1.0 - self.alpha) * last_ec
//     }
// }
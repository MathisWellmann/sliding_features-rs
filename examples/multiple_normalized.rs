extern crate rust_timeseries_generator;

use sliding_features::*;
use rust_timeseries_generator::gaussian_process::gen;

fn main() {
    // new sliding window
    let mut sf = SlidingWindow::new();

    // lets register alot of views
    let norm_len = 50;
    sf.register_view(Box::new(Normalizer::new(Box::new(RSI::new(14)), norm_len)));
    sf.register_view(Box::new(Normalizer::new(Box::new(ROC::new(14)), norm_len)));
    sf.register_view(Box::new(Normalizer::new(Box::new(ReFlex::new(14)), norm_len)));
    sf.register_view(Box::new(Normalizer::new(Box::new(TrendFlex::new(14)), norm_len)));
    sf.register_view(Box::new(Normalizer::new(Box::new(CenterOfGravity::new(14)), norm_len)));
    sf.register_view(Box::new(Normalizer::new(Box::new(CyberCycle::new(14)), norm_len)));

    // generate dummy values
    let vals = gen(1024, 100.0);
    for i in 0..vals.len() {
        sf.update(vals[i]);
        let last = sf.last();  // get the latest values from sliding window
        println!("last: {:?}", last);
    }
}

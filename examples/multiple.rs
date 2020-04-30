extern crate rust_timeseries_generator;

use sliding_features::*;
use rust_timeseries_generator::gaussian_process::gen;
use sliding_features::sliding_window::SlidingWindow;

fn main() {
    // new sliding window
    let mut sf = SlidingWindow::new();

    // lets register alot of views
    sf.register_view(Box::new(rsi::RSI::new(14)));
    sf.register_view(Box::new(roc::ROC::new(14)));
    sf.register_view(Box::new(re_flex::ReFlex::new(14)));
    sf.register_view(Box::new(trend_flex::TrendFlex::new(14)));
    sf.register_view(Box::new(center_of_gravity::CenterOfGravity::new(14)));
    sf.register_view(Box::new(cyber_cycle::CyberCycle::new(14)));

    // generate dummy values
    let vals = gen(1024, 100.0);
    for i in 0..vals.len() {
        sf.update(vals[i]);
        let last = sf.last();  // get the latest values from sliding window
        println!("last: {:?}", last);
    }
}

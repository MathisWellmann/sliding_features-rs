extern crate rust_timeseries_generator;

use sliding_features;
use rust_timeseries_generator::gaussian_process::gen;

fn main() {
    // new sliding window
    let mut sf = sliding_features::sliding_window::new();

    // lets register alot of views
    sf.register_view(Box::new(sliding_features::rsi::new(14)));
    sf.register_view(Box::new(sliding_features::roc::new(14)));
    sf.register_view(Box::new(sliding_features::re_flex::new(14)));
    sf.register_view(Box::new(sliding_features::trend_flex::new(14)));
    sf.register_view(Box::new(sliding_features::center_of_gravity::new(14)));
    sf.register_view(Box::new(sliding_features::cyber_cycle::new(14)));

    // generate dummy values
    let vals = gen(1024, 100.0);
    for i in 0..vals.len() {
        sf.update(vals[i]);
        let last = sf.last();  // get the latest values from sliding window
        println!("last: {:?}", last);
    }
}

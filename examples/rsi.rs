extern crate rust_timeseries_generator;

use sliding_features;
use rust_timeseries_generator::gaussian_process::gen;

fn main() {
    // new sliding window
    let mut sf = sliding_features::sliding_window::new();

    // view with rsi function
    let rsi = Box::new(sliding_features::rsi::new(14));
    sf.register_view(rsi);

    // generate dummy values
    let vals = gen(1024, 100.0);
    for i in 0..vals.len() {
        sf.update(vals[i]);
        let last = sf.last();  // get the latest values from sliding window
        println!("last: {:?}", last);
    }
}

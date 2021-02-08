/// Example showing how to use a single View
extern crate rust_timeseries_generator;

use rust_timeseries_generator::gaussian_process::gen;
use sliding_features::*;

fn main() {
    let mut rsi = Box::new(RSI::new_final(14));

    // generate dummy values
    let vals = gen(1024, 100.0);
    for r in &vals {
        rsi.update(*r); // update the rsi computation with the newest value
        let last = rsi.last(); // get the latest rsi value
        println!("last rsi value: {:?}", last);
    }
}

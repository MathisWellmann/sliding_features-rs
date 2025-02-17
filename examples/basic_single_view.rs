/// Example showing how to use a single View
extern crate time_series_generator;

use sliding_features::{pure_functions::Echo, sliding_windows::Rsi, View};
use time_series_generator::generate_standard_normal;

fn main() {
    let mut rsi = Rsi::new(Echo::new(), 14);

    // generate dummy values
    let vals = generate_standard_normal(1024, 100.0);
    for r in &vals {
        rsi.update(*r); // update the rsi computation with the newest value
        let last = rsi.last(); // get the latest rsi value
        println!("last rsi value: {:?}", last);
    }
}

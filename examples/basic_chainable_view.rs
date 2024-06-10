/// This Example provides a basic overview of chainable View definitions.
/// Assume you want to first transform your values with a Variance Stabilizing Centering Transform
//  and after that, smooth the values with an ALMA

// import the needed structs, and the View trait
use sliding_features::{
    pure_functions::Echo,
    sliding_windows::{ALMA, VSCT},
    View,
};

fn main() {
    // generate random value shifted up by 100.0 and scaled by 20.0,
    // a series which is neither centered around 0 nor variance stabilized
    let rands: Vec<f64> = (0..100)
        .map(|_| rand::random::<f64>() * 20.0 + 100.0)
        .collect();
    println!("rands: {:?}", rands);

    let window_len: usize = 20;
    let mut chain = ALMA::new(
        // first, define the last function which gets applied in the chain
        VSCT::new(Echo::new(), window_len), // Make the first transformation in the chain a VSCT
        window_len,
    );
    for v in &rands {
        // the chain will first call the inner most view, which is Echo.
        // after that it will apply the VSCT transform
        // and finally apply an Arnaux Legoux moving average
        chain.update(*v);
        if let Some(last_value) = chain.last() {
            println!("transformed value: {}", last_value);
        }
    }
}

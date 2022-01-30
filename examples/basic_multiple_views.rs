/// Basic Example showing how to utilize a SlidingWindow to combine multiple chained views
use sliding_features::*;

fn main() {
    let mut sf = SlidingWindow::new();

    // lets register some of views, which will later be updated in a single step
    let window_len: usize = 16;
    sf.register_view(Box::new(rsi::new_final(window_len)));
    sf.register_view(Box::new(roc::new_final(window_len)));
    // now a more complex view chain
    sf.register_view(Box::new(ALMA::new(
        VSCT::new(sma::new_final(window_len), window_len),
        window_len,
    )));

    // generate random dummy values
    let rands: Vec<f64> = (0..100)
        .map(|_| rand::random::<f64>() * 20.0 + 100.0)
        .collect();
    for r in &rands {
        sf.update(*r); // update all registered views with the newest value
        let last: Vec<f64> = sf.last(); // get the latest values from all views
        assert_eq!(last.len(), 3); // because there are 3 registered views we get three ordered values
        println!("last values: {:?}", last);
    }
}

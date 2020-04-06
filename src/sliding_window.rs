pub struct SlidingWindow {
    pub views: Vec<Box<dyn View>>,
}

pub fn new() -> SlidingWindow {
    return SlidingWindow{
        views: Vec::new(),
    }
}

impl SlidingWindow {
    // update propagates the newly observed candle through all views
    pub fn update(&mut self, val: f64) {
        for i in 0..self.views.len() {
            self.views[i].update(val);
        }
    }
    pub fn last(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for i in 0..self.views.len() {
            let mut last = self.views[i].last();
            if last.is_nan() {
                println!("last is nan. i: {}", i);
                last = 0.0;
            }
            out.push(last)
        }
        return out
    }

    // register_view adds the given view to SlidingFeatures
    pub fn register_view(&mut self, view: Box<dyn View>) {
        self.views.push(view);
    }
}

pub trait View {
    fn update(&mut self, val: f64);
    fn last(&self) -> f64;
}

#![deny(missing_docs, rustdoc::missing_crate_level_docs)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

pub mod pipeline;
pub mod pure_functions;
pub mod rolling;
pub mod sliding_windows;
pub mod spec;

#[cfg(test)]
mod plot;
#[cfg(test)]
mod test_data;

/// The most important Trait, defining methods which each sliding feature needs to implement
pub trait View<T: num::Float> {
    /// Update the state with a new value
    fn update(&mut self, val: T);

    /// Return the last value, if `Some`, then its ready.
    fn last(&self) -> Option<T>;
}

/// A boxed view is a view, so a chain can be assembled at runtime: every
/// combinator takes its input by value as `V: View<T>`, and
/// `Box<dyn View<T>>` is the only way to nest views whose types are not
/// known at compile time.
impl<T: num::Float, V: View<T> + ?Sized> View<T> for Box<V> {
    #[inline(always)]
    fn update(&mut self, val: T) {
        (**self).update(val);
    }

    #[inline(always)]
    fn last(&self) -> Option<T> {
        (**self).last()
    }
}

#[cfg(test)]
mod tests {
    // Used in benchmarks.
    #[allow(unused_imports)]
    use criterion::*;

    use super::*;
    use crate::{
        pure_functions::Echo,
        sliding_windows::Sma,
    };

    #[test]
    fn boxed_views_chain_like_concrete_ones() {
        let window = std::num::NonZeroUsize::new(3).expect("non zero");
        let inner: Box<dyn View<f64>> = Box::new(Sma::new(Echo::new(), window));
        let mut boxed = Sma::new(inner, window);
        let mut concrete = Sma::new(Sma::new(Echo::new(), window), window);
        for v in [1.0, 4.0, 2.0, 8.0, 5.0, 7.0] {
            boxed.update(v);
            concrete.update(v);
            assert_eq!(boxed.last(), concrete.last());
        }
        assert!(boxed.last().is_some());
    }
}

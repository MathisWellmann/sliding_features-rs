#![deny(
    missing_docs,
    rustdoc::missing_crate_level_docs,
    unused_crate_dependencies
)]
#![warn(clippy::all)]
#![doc = include_str!("../README.md")]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

pub mod pure_functions;
pub mod rolling;
pub mod sliding_windows;

#[cfg(test)]
mod plot;
#[cfg(test)]
mod test_data;

/// The most important Trait, defining methods which each sliding feature needs to implement
pub trait View: Send + Sync + std::fmt::Debug {
    /// Update the state with a new value
    fn update(&mut self, val: f64);

    /// Return the last value, if `Some`, then its ready.
    fn last(&self) -> Option<f64>;
}

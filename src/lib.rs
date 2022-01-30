#![deny(missing_docs, missing_crate_level_docs)]
#![warn(clippy::all)]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

pub mod alma;
pub mod center_of_gravity;
pub mod correlation_trend_indicator;
pub mod cyber_cycle;
pub mod echo;
pub mod entropy;
pub mod hl_normalizer;
pub mod laguerre_filter;
pub mod laguerre_rsi;
pub mod re_flex;
pub mod roc;
//pub mod roofing_filter;  // temporarily disabled roofing_filter until it is working properly
pub mod cumulative;
pub mod ehlers_fisher_transform;
pub mod ema;
pub mod multiplier;
pub mod my_rsi;
pub mod noise_elimination_technology;
pub mod polarized_fractal_efficiency;
pub mod rsi;
mod sliding_window;
pub mod sma;
pub mod trend_flex;
pub mod variance_stabilizing_transformation;
pub mod vsct;
pub mod welford_online;
pub mod welford_online_sliding;

#[cfg(test)]
mod plot;
#[cfg(test)]
mod test_data;

pub use alma::ALMA;
pub use center_of_gravity::CenterOfGravity;
pub use correlation_trend_indicator::CorrelationTrendIndicator;
pub use cyber_cycle::CyberCycle;
pub use echo::Echo;
pub use ehlers_fisher_transform::EhlersFisherTransform;
pub use ema::EMA;
pub use hl_normalizer::HLNormalizer;
pub use laguerre_filter::LaguerreFilter;
pub use laguerre_rsi::LaguerreRSI;
pub use multiplier::Multiplier;
pub use my_rsi::MyRSI;
pub use noise_elimination_technology::NET;
pub use polarized_fractal_efficiency::PolarizedFractalEfficiency;
pub use re_flex::ReFlex;
pub use roc::ROC;
pub use rsi::RSI;
pub use sliding_window::SlidingWindow;
pub use sma::SMA;
pub use trend_flex::TrendFlex;
pub use variance_stabilizing_transformation::VST;
pub use vsct::VSCT;
pub use welford_online::WelfordOnline;
pub use welford_online_sliding::WelfordOnlineSliding;
// pub use roofing_filter::RoofingFilter

// Does not impl View
pub use entropy::Entropy;

/// The most important Trait, defining methods which each sliding feature needs to implement
pub trait View: Send + Sync {
    /// Update the state with a new value
    fn update(&mut self, val: f64);

    /// Return the last value
    fn last(&self) -> f64;
}

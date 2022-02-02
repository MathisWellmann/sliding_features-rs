#![deny(missing_docs, rustdoc::missing_crate_level_docs)]
#![warn(clippy::all)]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

mod alma;
mod binary_entropy;
mod center_of_gravity;
mod correlation_trend_indicator;
mod cumulative;
mod cyber_cycle;
mod echo;
mod ehlers_fisher_transform;
mod ema;
mod hl_normalizer;
mod laguerre_filter;
mod laguerre_rsi;
mod my_rsi;
mod noise_elimination_technology;
mod polarized_fractal_efficiency;
mod re_flex;
mod roc;
mod rsi;
mod sma;
mod super_smoother;
mod tanh;
mod trend_flex;
mod variance_stabilizing_transformation;
mod vsct;
mod welford_online;
//pub mod roofing_filter;  // temporarily disabled roofing_filter until it is working properly

mod add;
mod constant;
mod divide;
mod multiply;
mod subtract;

#[cfg(test)]
mod plot;
#[cfg(test)]
mod test_data;

pub use alma::ALMA;
pub use binary_entropy::BinaryEntropy;
pub use center_of_gravity::CenterOfGravity;
pub use correlation_trend_indicator::CorrelationTrendIndicator;
pub use cumulative::Cumulative;
pub use cyber_cycle::CyberCycle;
pub use echo::Echo;
pub use ehlers_fisher_transform::EhlersFisherTransform;
pub use ema::EMA;
pub use hl_normalizer::HLNormalizer;
pub use laguerre_filter::LaguerreFilter;
pub use laguerre_rsi::LaguerreRSI;
pub use my_rsi::MyRSI;
pub use noise_elimination_technology::NET;
pub use polarized_fractal_efficiency::PolarizedFractalEfficiency;
pub use re_flex::ReFlex;
pub use roc::ROC;
pub use rsi::RSI;
pub use sma::SMA;
pub use super_smoother::SuperSmoother;
pub use tanh::Tanh;
pub use trend_flex::TrendFlex;
pub use variance_stabilizing_transformation::VST;
pub use vsct::VSCT;
pub use welford_online::WelfordOnline;
// Pub use roofing_filter::RoofingFilter

pub use add::Add;
pub use constant::Constant;
pub use divide::Divide;
pub use multiply::Multiply;
pub use subtract::Subtract;

/// The most important Trait, defining methods which each sliding feature needs to implement
pub trait View: Send + Sync {
    /// Update the state with a new value
    fn update(&mut self, val: f64);

    /// Return the last value
    fn last(&self) -> f64;
}

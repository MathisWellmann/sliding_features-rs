#![deny(missing_docs, missing_crate_level_docs)]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

mod alma;
mod center_of_gravity;
mod correlation_trend_indicator;
mod cyber_cycle;
mod echo;
mod entropy;
mod hl_normalizer;
mod laguerre_filter;
mod laguerre_rsi;
mod re_flex;
mod roc;
//mod roofing_filter;  // temporarily disabled roofing_filter until it is working properly
mod ehlers_fisher_transform;
mod ema;
mod multiplier;
mod my_rsi;
mod noise_elimination_technology;
mod polarized_fractal_efficiency;
mod rsi;
mod sliding_window;
mod sma;
mod trend_flex;
mod variance_stabilizing_transformation;
mod vsct;
mod welford_online;
mod welford_online_sliding;

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
pub use sliding_window::View;
pub use sma::SMA;
pub use trend_flex::TrendFlex;
pub use variance_stabilizing_transformation::VST;
pub use vsct::VSCT;
pub use welford_online::WelfordOnline;
pub use welford_online_sliding::WelfordOnlineSliding;
// pub use roofing_filter::RoofingFilter

// Does not impl View
pub use entropy::Entropy;

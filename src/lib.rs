#![deny(missing_docs, missing_crate_level_docs)]

//! The sliding_features crate provides modular, chainable sliding windows
//! for various signal processing function and technical indicators

mod alma;
mod center_of_gravity;
mod correlation_trend_indicator;
mod cyber_cycle;
mod echo;
mod entropy;
mod laguerre_filter;
mod laguerre_rsi;
mod normalizer;
mod re_flex;
mod roc;
//mod roofing_filter;  // temporarily disabled roofing_filter until it is working properly
mod rsi;
mod sliding_window;
mod sma;
mod std_dev;
mod trend_flex;
mod variance_stabilizing_transformation;
mod vsct;
mod welford_online;
mod my_rsi;
mod net;

#[cfg(test)]
mod plot;
#[cfg(test)]
mod test_data;

pub use alma::ALMA;
pub use center_of_gravity::CenterOfGravity;
pub use correlation_trend_indicator::CorrelationTrendIndicator;
pub use cyber_cycle::CyberCycle;
pub use echo::Echo;
pub use laguerre_filter::LaguerreFilter;
pub use laguerre_rsi::LaguerreRSI;
pub use normalizer::HLNormalizer;
pub use re_flex::ReFlex;
pub use roc::ROC;
pub use rsi::RSI;
pub use sliding_window::SlidingWindow;
pub use sliding_window::View;
pub use sma::SMA;
pub use std_dev::StdDev;
pub use trend_flex::TrendFlex;
pub use variance_stabilizing_transformation::VST;
pub use vsct::VSCT;
pub use welford_online::WelfordOnline;
pub use my_rsi::MyRSI;
pub use net::NET;
// pub use roofing_filter::RoofingFilter

// Does not impl View
pub use entropy::Entropy;

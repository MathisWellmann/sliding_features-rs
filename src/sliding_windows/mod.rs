//! This module contains `View` implementations that act like sliding windows, usually containinig a history of values over the window.
//! Usually more memory intensive than the implementations in `rolling`.

mod alma;
mod binary_entropy;
mod center_of_gravity;
mod correlation_trend_indicator;
mod cumulative;
mod cyber_cycle;
mod ehlers_fisher_transform;
mod ema;
mod hl_normalizer;
mod laguerre_filter;
mod laguerre_rsi;
mod max;
mod min;
mod my_rsi;
mod noise_elimination_technology;
mod polarized_fractal_efficiency;
mod re_flex;
mod roc;
mod roofing_filter;
mod rsi;
mod sma;
mod super_smoother;
mod trend_flex;
mod variance_stabilizing_transformation;
mod vsct;
mod welford_online;

pub use alma::Alma;
pub use binary_entropy::BinaryEntropy;
pub use center_of_gravity::CenterOfGravity;
pub use correlation_trend_indicator::CorrelationTrendIndicator;
pub use cumulative::Cumulative;
pub use cyber_cycle::CyberCycle;
pub use ehlers_fisher_transform::EhlersFisherTransform;
pub use ema::Ema;
pub use hl_normalizer::HLNormalizer;
pub use laguerre_filter::LaguerreFilter;
pub use laguerre_rsi::LaguerreRSI;
pub use max::Max;
pub use min::Min;
pub use my_rsi::MyRSI;
pub use noise_elimination_technology::NoiseEliminationTechnology;
pub use polarized_fractal_efficiency::PolarizedFractalEfficiency;
pub use re_flex::ReFlex;
pub use roc::Roc;
pub use roofing_filter::RoofingFilter;
pub use rsi::Rsi;
pub use sma::Sma;
pub use super_smoother::SuperSmoother;
pub use trend_flex::TrendFlex;
pub use variance_stabilizing_transformation::Vst;
pub use vsct::Vsct;
pub use welford_online::WelfordOnline;

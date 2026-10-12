//! This module contains `View` implementations that are updated on a rolling basis, but don't maintain a sliding window with history.

mod alpha_beta_filter;
mod bocpd;
mod cusum;
mod drawdown;
mod first;
mod huber_ema;
mod kalman1d;
mod ln_return;
mod one_euro_filter;
mod page_hinkley;
mod welford_rolling;

pub use alpha_beta_filter::AlphaBetaFilter;
pub use bocpd::Bocpd;
pub use cusum::Cusum;
pub use drawdown::Drawdown;
pub use first::First;
pub use huber_ema::HuberEma;
pub use kalman1d::Kalman1d;
pub use ln_return::LnReturn;
pub use one_euro_filter::OneEuroFilter;
pub use page_hinkley::{
    PageHinkley,
    PageHinkleyDirection,
};
pub use welford_rolling::WelfordRolling;

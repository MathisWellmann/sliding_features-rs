//! This module contains `View` implementations that are updated on a rolling basis, but don't maintain a sliding window with history.

mod drawdown;
mod ln_return;
mod welford_online;

pub use drawdown::Drawdown;
pub use ln_return::LnReturn;
pub use welford_online::WelfordRolling;

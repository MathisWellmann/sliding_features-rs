//! This module contains `View` implementations that are updated on a rolling basis, but don't maintain a sliding window with history.

mod drawdown;
mod ln_return;

pub use drawdown::Drawdown;
pub use ln_return::LnReturn;

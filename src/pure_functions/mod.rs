//! This module contains `View` implementations that act like pure functions.
//! They don't really rely on an internal state, but rather process value from other `View`s

mod add;
mod constant;
mod divide;
mod echo;
mod gte;
mod lte;
mod multiply;
mod subtract;
mod tanh;

pub use add::Add;
pub use constant::Constant;
pub use divide::Divide;
pub use echo::Echo;
pub use gte::GTE;
pub use lte::LTE;
pub use multiply::Multiply;
pub use subtract::Subtract;
pub use tanh::Tanh;

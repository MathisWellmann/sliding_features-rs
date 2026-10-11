//! Feature pipelines as data: many features over several input series,
//! compiled into one flat program at runtime.
//!
//! A [`Builder`] assembles a [`Pipeline`] from inputs, constants, the
//! arithmetic [`BinaryOp`]s and calls of [`Function`]s, and names its
//! outputs. Equal computations are shared, so features that use the same
//! `ema(x, 10)` compute it once. The pipeline runs as a flat list of
//! operations over a value buffer: one pass per update, no allocation, no
//! recursion and no dynamic dispatch.
//!
//! Each function declares what it needs of its argument ([`Requirement`])
//! and the [`Domain`] of its output, so a pipeline can be checked before it
//! sees any data.
//!
//! ```
//! use sliding_features::pipeline::{BinaryOp, Builder, Domain, Function, Input, Limits};
//!
//! let inputs = vec![Input::new("price", Domain::positive())];
//! let mut b = Builder::new(inputs, Limits::default()).unwrap();
//! let price = b.lookup("price").unwrap();
//! let r = b.call(Function::LnReturn, price, &[]).unwrap();
//! let fast = b.call(Function::Std, r, &[2.0]).unwrap();
//! let slow = b.call(Function::Std, r, &[4.0]).unwrap();
//! let ratio = b.binary(BinaryOp::Div, fast, slow).unwrap();
//! b.output("vol_ratio", ratio).unwrap();
//! let mut pipeline = b.build().unwrap();
//! for p in [100.0, 101.0, 99.0, 102.0, 100.0] {
//!     pipeline.update(&[p]);
//! }
//! assert!(pipeline.output(0).is_some());
//! ```

mod builder;
mod compiled;
mod domain;
mod error;
mod function;
mod graph;
mod program;
mod state;
#[cfg(test)]
mod tests;

pub use builder::{
    Builder,
    Input,
    Limits,
};
pub use compiled::{
    Feature,
    Pipeline,
};
pub use domain::Domain;
pub use error::{
    CompileError,
    Diagnostic,
    Severity,
};
pub use function::{
    BinaryOp,
    Function,
    Param,
    Requirement,
};
pub use graph::NodeId;

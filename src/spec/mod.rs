//! Features as text: a small expression language over this crate's views.
//!
//! A spec describes a chain of views fed one input series `x`. It is parsed
//! into an [`Expr`], checked against resource [`Limits`] and built into a
//! [`DynView`] at runtime, so features can be written, stored, compared and
//! shared as data, for example by a search or an LLM agent, without
//! compiling code.
//!
//! ```text
//! zscore(ema(x, 10) - ema(x, 50), 200)     // a normalized EMA crossover
//! std(ln_return(x), 60) / std(ln_return(x), 600)   // short vs long volatility
//! ```
//!
//! Grammar: numbers, `x`, `+ - * /` with the usual precedence, unary `-`,
//! parentheses and calls `name(expr, param, ...)` whose parameters are
//! number literals; see [`Function::reference`] for every function. Binary
//! operators feed both sides the same input.
//!
//! A built view only ever outputs finite values: a division by zero or a
//! logarithm of a non-positive value makes the node not ready instead.
//!
//! ```
//! use sliding_features::{View, spec::{Expr, Limits}};
//!
//! let spec: Expr = "x - sma(x,2)".parse().unwrap();
//! assert_eq!(spec.to_string(), "(x - sma(x, 2))");
//! let mut view = spec.build(&Limits::default()).unwrap();
//! view.update(100.0);
//! assert_eq!(view.last(), None, "the SMA needs two values");
//! view.update(104.0);
//! assert_eq!(view.last(), Some(2.0));
//! ```

mod build;
mod error;
mod expr;
mod function;
mod parse;

pub use build::{
    DynView,
    Limits,
};
pub use error::SpecError;
pub use expr::{
    BinaryOp,
    Expr,
};
pub use function::{
    Function,
    Param,
};

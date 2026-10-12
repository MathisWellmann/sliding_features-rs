//! Feature pipelines as data: many features over several input series,
//! written as text (e.g. by an LLM) or assembled by code, checked, and
//! compiled into one flat program at runtime.
//!
//! ```
//! use sliding_features::pipeline::{Domain, Input, Limits, Pipeline};
//!
//! let source = "
//!     let r = ln_return(price)  # a helper, not an output
//!     vol_ratio = std(r, 20) / std(r, 100)
//!     trend = zscore(ema(price, 10) - ema(price, 50), 200)
//! ";
//! let inputs = vec![Input::new("price", Domain::positive())];
//! let mut pipeline = Pipeline::compile(source, inputs, Limits::default()).unwrap();
//! assert_eq!(Vec::from_iter(pipeline.output_names()), ["vol_ratio", "trend"]);
//! for i in 0..300 {
//!     pipeline.update(&[100.0 + (f64::from(i) * 0.1).sin()]);
//! }
//! assert!(pipeline.outputs().all(|v| v.is_some()));
//! ```
//!
//! [`reference()`] describes the language and every function, to put into an
//! LLM's prompt. A [`Builder`] assembles the same pipelines node by node,
//! e.g. for a search.
//!
//! Equal computations are shared, so features that use the same
//! `ema(x, 10)` compute it once. The pipeline runs as a flat list of
//! operations over a value buffer: one pass per update, no allocation, no
//! recursion and no dynamic dispatch.
//!
//! Each [`Function`] declares what it needs of its argument
//! ([`Requirement`]) and the [`Domain`] of its output. From the domains of
//! the inputs the compiler derives every node's domain and warm-up before
//! any data is seen: it rejects calls that are never ready and constant
//! outputs, and warns about calls that are not always ready, each with a
//! [`Location`] and a message that says how to fix it.

mod analysis;
mod builder;
mod compiled;
mod display;
mod domain;
mod error;
mod function;
mod graph;
mod lex;
mod parse;
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
    Location,
    Severity,
};
pub use function::{
    BinaryOp,
    Function,
    Param,
    Requirement,
};
pub use graph::NodeId;

/// The language guide and the table of every function, in markdown: what
/// an LLM needs in its prompt to write pipelines.
pub fn reference() -> String {
    let mut out = String::from(LANGUAGE);
    out.push_str(&Function::reference());
    out
}

const LANGUAGE: &str = "\
# Feature pipelines

A pipeline computes features (its outputs) from input series, one update at a time.

- One statement per line (or separated by `;`); `#` starts a comment.
- `name = expr` defines an output. `let name = expr` defines a helper that later \
statements can use by name. A bare `expr` is an output named by its text.
- An expression is a number, an input or helper name, `a + b`, `a - b`, `a * b`, `a / b` \
(usual precedence), `-a`, `(a)`, or a call `function(a, param, ...)`: the first argument \
is an expression, the parameters after it are number literals.
- Parameter kinds in the table: `window` is a whole number of updates, `integer` a whole number \
in the range the description gives (e.g. a number of poles), `fraction` a number strictly \
between 0 and 1, `positive` a number above 0, `real` any number. Some descriptions add limits, \
e.g. a bandwidth below window / 4.
- Operators combine the values of the same update. Equal subexpressions are computed once.
- An output is not ready (no value) until its windows fill, or while an argument is outside \
what a function accepts (a division by zero, a logarithm of a non-positive value). Ready \
outputs are always finite. A function whose argument is not ready keeps its last value.
- The compiler rejects calls that can never be ready and constant outputs, and warns about \
calls that are not always ready.

```text
let r = ln_return(price)
vol_ratio = std(r, 20) / std(r, 100)
trend = zscore(ema(price, 10) - ema(price, 50), 200)
```

## Functions

";

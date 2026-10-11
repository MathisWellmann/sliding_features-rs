//! Feature pipelines as data: many features over several input series,
//! compiled into one flat program at runtime.
//!
//! This module holds the catalogue of operators and [`Function`]s that
//! pipelines are made of, with what each needs of its argument
//! ([`Requirement`]) and the [`Domain`] of its output, so a pipeline can be
//! checked before it sees any data.

mod domain;
mod function;

pub use domain::Domain;
pub use function::{
    BinaryOp,
    Function,
    Param,
    Requirement,
};

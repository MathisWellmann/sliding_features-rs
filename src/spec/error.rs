//! The error of parsing or validating a spec.

use std::fmt;

/// Why a spec was rejected.
#[deprecated(
    note = "use the `pipeline` module: its text form compiles every spec to the same values"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecError {
    /// The byte offset in the spec text, for syntax errors.
    position: Option<usize>,
    message: String,
}

impl SpecError {
    pub(super) fn parse(position: usize, message: String) -> Self {
        Self {
            position: Some(position),
            message,
        }
    }

    pub(super) fn invalid(message: String) -> Self {
        Self {
            position: None,
            message,
        }
    }

    /// The byte offset in the spec text the error points at, if any.
    pub fn position(&self) -> Option<usize> {
        self.position
    }

    /// What is wrong, without the position.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.position {
            Some(at) => write!(f, "at byte {at}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for SpecError {}

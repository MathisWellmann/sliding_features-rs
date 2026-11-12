// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! What the compiler reports about a pipeline.

use std::fmt;

/// How bad a [`Diagnostic`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// The pipeline is rejected.
    Error,
    /// The pipeline compiles, but likely does not do what was meant.
    Warning,
}

/// Where in the text form a problem is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    offset: usize,
    line: usize,
    column: usize,
}

impl Location {
    pub(super) fn new(offset: usize, line: usize, column: usize) -> Self {
        assert!(line >= 1, "lines count from 1");
        assert!(column >= 1, "columns count from 1");
        Self {
            offset,
            line,
            column,
        }
    }

    /// The byte offset in the text.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The line, counted from 1.
    pub fn line(&self) -> usize {
        self.line
    }

    /// The column in characters, counted from 1.
    pub fn column(&self) -> usize {
        self.column
    }
}

/// One problem the compiler found, worded to be fixed by whoever wrote the
/// pipeline, a person or an LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    severity: Severity,
    location: Option<Location>,
    message: String,
}

impl Diagnostic {
    pub(super) fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            location: None,
            message: message.into(),
        }
    }

    pub(super) fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            location: None,
            message: message.into(),
        }
    }

    /// The diagnostic at `location`, unless it already has one.
    pub(super) fn or_at(mut self, location: Option<Location>) -> Self {
        self.location = self.location.or(location);
        self
    }

    /// How bad the problem is.
    pub fn severity(&self) -> Severity {
        self.severity
    }

    /// Where the problem is in the text form, if the pipeline was written
    /// as text.
    pub fn location(&self) -> Option<Location> {
        self.location
    }

    /// What is wrong.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        match self.location {
            Some(at) => write!(
                f,
                "{severity} at {}:{}: {}",
                at.line, at.column, self.message
            ),
            None => write!(f, "{severity}: {}", self.message),
        }
    }
}

/// Why a pipeline was rejected: at least one error, and every warning
/// found on the way, one per line in the [`fmt::Display`] form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    diagnostics: Vec<Diagnostic>,
}

impl CompileError {
    pub(super) fn new(diagnostics: Vec<Diagnostic>) -> Self {
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            "a compile error holds an error"
        );
        Self { diagnostics }
    }

    pub(super) fn single(message: impl Into<String>) -> Self {
        Self::new(vec![Diagnostic::error(message)])
    }

    pub(super) fn at(location: Location, message: impl Into<String>) -> Self {
        Self::new(vec![Diagnostic::error(message).or_at(Some(location))])
    }

    /// The error with `location` on every diagnostic that has none.
    pub(super) fn or_at(self, location: Option<Location>) -> Self {
        Self {
            diagnostics: Vec::from_iter(self.diagnostics.into_iter().map(|d| d.or_at(location))),
        }
    }

    /// Every problem found, errors and warnings, in the order found.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, diagnostic) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompileError {}

/// Why a [`PipelineState`](super::PipelineState) does not fit a pipeline.
/// The pipeline is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RestoreError {
    /// The state was saved by a version of this crate with another layout
    /// of the state of the views.
    Format {
        /// The format of the saved state.
        saved: u32,
        /// The format this version of the crate reads.
        expected: u32,
    },
    /// The state was saved from another pipeline: other inputs, functions,
    /// parameters or outputs. Both are the inputs in order and the
    /// canonical text.
    Program {
        /// The pipeline the state was saved from.
        saved: String,
        /// The pipeline the state was restored into.
        expected: String,
    },
    /// The same pipeline, but the state does not fit its operations, e.g.
    /// from a version of this crate that compiles it into other operations.
    Layout,
}

impl fmt::Display for RestoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RestoreError::Format { saved, expected } => write!(
                f,
                "the state has format {saved}, this version of sliding_features reads format {expected}"
            ),
            RestoreError::Program { saved, expected } => write!(
                f,
                "the state is of another pipeline:\n{saved}\nand cannot continue:\n{expected}"
            ),
            RestoreError::Layout => {
                f.write_str("the state does not fit the operations of the pipeline")
            }
        }
    }
}

impl std::error::Error for RestoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_lists_every_diagnostic() {
        let err = CompileError::new(vec![Diagnostic::error("a"), Diagnostic::error("b")]);
        assert_eq!(err.to_string(), "error: a\nerror: b");
        assert_eq!(err.diagnostics().len(), 2);
        let located = err.or_at(Some(Location::new(4, 2, 3)));
        assert_eq!(located.to_string(), "error at 2:3: a\nerror at 2:3: b");
        let kept = located.or_at(Some(Location::new(0, 1, 1)));
        assert_eq!(
            kept.diagnostics()[0].location().map(|l| l.offset()),
            Some(4)
        );
    }

    #[test]
    #[should_panic(expected = "holds an error")]
    fn a_compile_error_needs_an_error() {
        let _ = CompileError::new(Vec::new());
    }
}

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

/// One problem the compiler found, worded to be fixed by whoever wrote the
/// pipeline, a person or an LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    severity: Severity,
    message: String,
}

impl Diagnostic {
    pub(super) fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
        }
    }

    pub(super) fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
        }
    }

    /// How bad the problem is.
    pub fn severity(&self) -> Severity {
        self.severity
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
        write!(f, "{severity}: {}", self.message)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_lists_every_diagnostic() {
        let err = CompileError::new(vec![Diagnostic::error("a"), Diagnostic::error("b")]);
        assert_eq!(err.to_string(), "error: a\nerror: b");
        assert_eq!(err.diagnostics().len(), 2);
    }

    #[test]
    #[should_panic(expected = "holds an error")]
    fn a_compile_error_needs_an_error() {
        let _ = CompileError::new(Vec::new());
    }
}

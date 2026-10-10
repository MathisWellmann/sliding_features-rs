//! The expression tree of a spec, its canonical text form and its checks.

use std::fmt;

use super::{
    Function,
    Limits,
    Param,
    SpecError,
};

/// An arithmetic operator between two expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`, not ready while `b` is zero.
    Div,
}

impl BinaryOp {
    fn symbol(self) -> char {
        match self {
            BinaryOp::Add => '+',
            BinaryOp::Sub => '-',
            BinaryOp::Mul => '*',
            BinaryOp::Div => '/',
        }
    }
}

/// A feature as an expression over one input series `x`.
///
/// Its [`fmt::Display`] form is canonical: parsing it yields the same tree,
/// so two specs that print the same compute the same feature.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// The input series, `x`.
    Input,
    /// A constant.
    Const(f64),
    /// The negated expression, `-a`.
    Neg(Box<Expr>),
    /// Arithmetic between two expressions fed the same input.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// The left operand.
        lhs: Box<Expr>,
        /// The right operand.
        rhs: Box<Expr>,
    },
    /// A function applied to an expression, with its parameters.
    Call {
        /// The function.
        function: Function,
        /// The expression the function is applied to.
        arg: Box<Expr>,
        /// The parameters after the argument, see [`Function::params`].
        params: Vec<f64>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Input => f.write_str("x"),
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Neg(e) => write!(f, "-{e}"),
            Expr::Binary { op, lhs, rhs } => write!(f, "({lhs} {} {rhs})", op.symbol()),
            Expr::Call {
                function,
                arg,
                params,
            } => {
                write!(f, "{}({arg}", function.name())?;
                for p in params {
                    write!(f, ", {p}")?;
                }
                f.write_str(")")
            }
        }
    }
}

impl std::str::FromStr for Expr {
    type Err = SpecError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        super::parse::parse(s)
    }
}

impl Expr {
    /// Parse a spec, e.g. `zscore(ema(x, 10) - ema(x, 50), 200)`.
    ///
    /// Checks the syntax and every call's signature; [`Expr::validate`]
    /// checks the resource limits.
    pub fn parse(s: &str) -> Result<Self, SpecError> {
        super::parse::parse(s)
    }

    /// The number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        1 + self.children().map(Expr::node_count).sum::<usize>()
    }

    /// The length of the longest path from the root to a leaf.
    pub fn depth(&self) -> usize {
        1 + self.children().map(Expr::depth).max().unwrap_or(0)
    }

    /// The largest window parameter anywhere in the tree (0 without one).
    pub fn max_window(&self) -> usize {
        let own = match self {
            Expr::Call {
                function, params, ..
            } => window_params(*function, params).max().unwrap_or(0),
            _ => 0,
        };
        self.children().map(Expr::max_window).fold(own, usize::max)
    }

    /// An estimate of the input updates needed before the feature is ready:
    /// the window lengths summed along the longest chain. Use it to size the
    /// history replayed before the feature is read.
    pub fn warm_up(&self) -> usize {
        match self {
            Expr::Input | Expr::Const(_) => 0,
            Expr::Neg(e) => e.warm_up(),
            Expr::Binary { lhs, rhs, .. } => lhs.warm_up().max(rhs.warm_up()),
            Expr::Call {
                function,
                arg,
                params,
            } => {
                let own: usize = window_params(*function, params).sum();
                arg.warm_up() + own + usize::from(*function == Function::LnReturn)
            }
        }
    }

    fn children(&self) -> impl Iterator<Item = &Expr> {
        let (a, b) = match self {
            Expr::Input | Expr::Const(_) => (None, None),
            Expr::Neg(e) | Expr::Call { arg: e, .. } => (Some(e.as_ref()), None),
            Expr::Binary { lhs, rhs, .. } => (Some(lhs.as_ref()), Some(rhs.as_ref())),
        };
        a.into_iter().chain(b)
    }

    /// Check the tree against `limits` and every call against its
    /// signature, so that [`Expr::build`] cannot fail or panic.
    pub fn validate(&self, limits: &Limits) -> Result<(), SpecError> {
        if self.node_count() > limits.max_nodes {
            return Err(SpecError::invalid(format!(
                "{} nodes exceed the limit of {}",
                self.node_count(),
                limits.max_nodes
            )));
        }
        if self.depth() > limits.max_depth {
            return Err(SpecError::invalid(format!(
                "depth {} exceeds the limit of {}",
                self.depth(),
                limits.max_depth
            )));
        }
        self.validate_nodes(limits)
    }

    fn validate_nodes(&self, limits: &Limits) -> Result<(), SpecError> {
        match self {
            Expr::Const(c) if !c.is_finite() => {
                return Err(SpecError::invalid(format!("constant {c} is not finite")));
            }
            Expr::Call {
                function, params, ..
            } => validate_call(*function, params, limits)?,
            _ => {}
        }
        self.children().try_for_each(|c| c.validate_nodes(limits))
    }
}

/// The window parameters of a call, as lengths.
fn window_params(function: Function, params: &[f64]) -> impl Iterator<Item = usize> + '_ {
    function
        .params()
        .iter()
        .zip(params)
        .filter(|(kind, _)| **kind == Param::Window)
        // Validated windows are small positive integers, exact as f64.
        .map(|(_, value)| *value as usize)
}

/// Check one call's parameters against its signature and `limits`.
pub(super) fn validate_call(
    function: Function,
    params: &[f64],
    limits: &Limits,
) -> Result<(), SpecError> {
    let expected = function.params();
    if params.len() != expected.len() {
        return Err(SpecError::invalid(format!(
            "{} takes {} parameter(s) after its argument, got {}",
            function.signature(),
            expected.len(),
            params.len()
        )));
    }
    for (kind, value) in expected.iter().zip(params) {
        if !value.is_finite() {
            return Err(SpecError::invalid(format!(
                "{}: parameter {value} is not finite",
                function.name()
            )));
        }
        if *kind == Param::Window {
            let valid = value.fract() == 0.0
                && *value >= function.min_window() as f64
                && *value <= limits.max_window as f64;
            if !valid {
                return Err(SpecError::invalid(format!(
                    "{}: window {value} must be an integer in [{}, {}]",
                    function.name(),
                    function.min_window(),
                    limits.max_window
                )));
            }
        }
    }
    if function == Function::LaguerreFilter && !(params[0] > 0.0 && params[0] < 1.0) {
        return Err(SpecError::invalid(format!(
            "laguerre_filter: gamma {} must lie in (0, 1)",
            params[0]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(function: Function, arg: Expr, params: &[f64]) -> Expr {
        Expr::Call {
            function,
            arg: Box::new(arg),
            params: params.to_vec(),
        }
    }

    #[test]
    fn display_is_canonical() {
        let e = Expr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(call(Function::Ema, Expr::Input, &[10.0])),
            rhs: Box::new(Expr::Neg(Box::new(Expr::Const(0.5)))),
        };
        assert_eq!(e.to_string(), "(ema(x, 10) - -0.5)");
        let pfe = call(Function::Pfe, Expr::Input, &[20.0, 5.0]);
        assert_eq!(pfe.to_string(), "pfe(x, 20, 5)");
    }

    #[test]
    fn size_and_warm_up() {
        let inner = call(Function::Sma, Expr::Input, &[10.0]);
        let e = call(Function::ZScore, inner, &[50.0]);
        assert_eq!(e.node_count(), 3);
        assert_eq!(e.depth(), 3);
        assert_eq!(e.max_window(), 50);
        assert_eq!(e.warm_up(), 60);
        let both = Expr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(e.clone()),
            rhs: Box::new(call(Function::Lag, Expr::Input, &[100.0])),
        };
        assert_eq!(both.warm_up(), 100);
    }

    #[test]
    fn validation_rejects_what_build_cannot_handle() {
        let limits = Limits::default();
        let ok = call(Function::Pfe, Expr::Input, &[3.0, 3.0]);
        assert!(ok.validate(&limits).is_ok());
        for bad in [
            call(Function::Sma, Expr::Input, &[0.0]),
            call(Function::Sma, Expr::Input, &[2.5]),
            call(Function::Sma, Expr::Input, &[]),
            call(Function::Pfe, Expr::Input, &[2.0, 3.0]),
            call(Function::Sma, Expr::Input, &[1e9]),
            call(Function::LaguerreFilter, Expr::Input, &[1.0]),
            call(Function::Gte, Expr::Input, &[f64::NAN]),
            Expr::Const(f64::INFINITY),
        ] {
            assert!(bad.validate(&limits).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn validation_bounds_the_tree() {
        let mut e = Expr::Input;
        for _ in 0..20 {
            e = Expr::Neg(Box::new(e));
        }
        assert!(e.validate(&Limits::default()).is_err(), "too deep");
        let limits = Limits {
            max_depth: 100,
            max_nodes: 5,
            ..Limits::default()
        };
        assert!(e.validate(&limits).is_err(), "too many nodes");
    }
}

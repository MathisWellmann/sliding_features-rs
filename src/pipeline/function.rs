//! The operators and functions of a pipeline: one table holds each
//! function's name, parameters, argument requirement, output range, runtime
//! form and description, so they cannot drift apart.

use std::{
    fmt::Write as _,
    num::NonZeroUsize,
};

use super::{
    Domain,
    state::{
        Kind,
        Pure,
    },
};
use crate::{
    pure_functions::Echo,
    rolling::{
        Drawdown,
        LnReturn,
    },
    sliding_windows::{
        Alma,
        BinaryEntropy,
        CenterOfGravity,
        CorrelationTrendIndicator,
        Cumulative,
        CyberCycle,
        EhlersFisherTransform,
        Ema,
        Lag,
        LaguerreFilter,
        LaguerreRSI,
        MadScaler,
        Max,
        Min,
        MinMaxNormalizer,
        MyRSI,
        NoiseEliminationTechnology,
        PolarizedFractalEfficiency,
        ReFlex,
        Roc,
        RoofingFilter,
        Rsi,
        Sma,
        SuperSmoother,
        TrendFlex,
        Vst,
        WelfordOnline,
        ZScoreStandardization,
    },
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
    /// The operator's symbol in the text form.
    pub fn symbol(self) -> char {
        match self {
            BinaryOp::Add => '+',
            BinaryOp::Sub => '-',
            BinaryOp::Mul => '*',
            BinaryOp::Div => '/',
        }
    }

    /// `a op b`. NaN (not ready) operands give NaN, and so does a division
    /// by zero once the caller filters the infinity.
    #[inline]
    pub(super) fn eval(self, a: f64, b: f64) -> f64 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
        }
    }

    /// The domain of `a op b` for `a` in `lhs` and `b` in `rhs`, `None`
    /// when the result is never finite (a division by a zero constant).
    pub fn output_domain(self, lhs: Domain, rhs: Domain) -> Option<Domain> {
        Some(match self {
            BinaryOp::Add => lhs.add(rhs),
            BinaryOp::Sub => lhs.sub(rhs),
            BinaryOp::Mul => lhs.mul(rhs),
            BinaryOp::Div => match lhs.div(rhs) {
                Some(d) => d,
                // A divisor that is always zero is never valid; one that
                // can be zero leaves the quotient unbounded.
                None if rhs.as_point() == Some(0.0) => return None,
                None => Domain::REAL,
            },
        })
    }
}

/// The kind of a parameter that follows the expression argument of a call.
/// Parameters are number literals, so they can be checked before the
/// pipeline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    /// A window length, counted in updates of the argument: an integer of
    /// at least [`Function::min_window`].
    Window,
    /// A finite real number.
    Real,
    /// A real number strictly between 0 and 1, e.g. a damping factor.
    Fraction,
}

impl Param {
    /// The name of the parameter kind in signatures.
    pub fn name(self) -> &'static str {
        match self {
            Param::Window => "window",
            Param::Real => "real",
            Param::Fraction => "fraction",
        }
    }
}

/// What a function needs of its argument to update. While the argument
/// does not meet it, the call is not ready (or holds its last value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Requirement {
    /// Any finite value.
    Any,
    /// Values above zero, e.g. for logarithms.
    Positive,
    /// Values of at least zero, e.g. for square roots.
    NonNegative,
    /// Any finite value, but an argument that never changes sign makes the
    /// output constant, e.g. the entropy of signs.
    BothSigns,
}

impl Requirement {
    /// The values the requirement admits.
    pub fn domain(self) -> Domain {
        match self {
            Requirement::Any | Requirement::BothSigns => Domain::REAL,
            Requirement::Positive => Domain::positive(),
            Requirement::NonNegative => Domain::non_negative(),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Requirement::Any => "any",
            Requirement::Positive => "positive",
            Requirement::NonNegative => "non-negative",
            Requirement::BothSigns => "changes sign",
        }
    }
}

/// The checked parameters of a call.
#[derive(Debug, Clone, Copy)]
pub(super) struct Params<'a>(&'a [f64]);

impl<'a> Params<'a> {
    /// Wrap parameters that passed [`Function::check_params`].
    pub(super) fn new(function: Function, params: &'a [f64]) -> Self {
        assert_eq!(params.len(), function.params().len(), "checked parameters");
        Self(params)
    }

    /// The `i`-th parameter as a window length.
    pub(super) fn window(self, i: usize) -> NonZeroUsize {
        window_len(self.0[i]).expect("checked window")
    }

    /// The `i`-th parameter as a real number.
    pub(super) fn real(self, i: usize) -> f64 {
        let value = self.0[i];
        assert!(value.is_finite(), "checked parameter");
        value
    }
}

/// The window length `value` denotes: an integer from 1 to 2^53, where
/// every integer is exact as `f64`.
pub(super) fn window_len(value: f64) -> Option<NonZeroUsize> {
    const MAX_EXACT: f64 = 9_007_199_254_740_992.0;
    if value.fract() != 0.0 || !(1.0..=MAX_EXACT).contains(&value) {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "an integer in [1, 2^53] is exact as usize on 64-bit targets"
    )]
    let len = value as usize;
    NonZeroUsize::new(len)
}

/// The output domain of a function given its (admitted) argument domain.
type RangeFn = fn(Domain, Params<'_>) -> Domain;

/// What a call becomes at runtime given its checked parameters.
type BuildFn = fn(Params<'_>) -> Kind;

/// Declares [`Function`] and its metadata from one table.
macro_rules! functions {
    ($($variant:ident {
        name: $name:literal,
        params: [$($param:ident),*],
        min_window: $min:literal,
        requires: $requires:ident,
        range: $range:expr,
        build: $build:expr,
        doc: $doc:literal $(,)?
    })*) => {
        /// A function of a pipeline. Every call takes an expression as its
        /// first argument, then its [`Function::params`] as number literals.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Function {
            $(#[doc = $doc] $variant,)*
        }

        impl Function {
            /// Every function, in declaration order.
            pub const ALL: &'static [Function] = &[$(Function::$variant,)*];

            /// The name the function is called by.
            pub fn name(self) -> &'static str {
                match self { $(Function::$variant => $name,)* }
            }

            /// The parameters after the expression argument.
            pub fn params(self) -> &'static [Param] {
                match self { $(Function::$variant => &[$(Param::$param),*],)* }
            }

            /// The smallest window length the function accepts: shorter
            /// windows are degenerate (e.g. an RSI over one value).
            pub fn min_window(self) -> usize {
                match self { $(Function::$variant => $min,)* }
            }

            /// What the function needs of its argument.
            pub fn requires(self) -> Requirement {
                match self { $(Function::$variant => Requirement::$requires,)* }
            }

            /// What the function computes, one line.
            pub fn description(self) -> &'static str {
                match self { $(Function::$variant => $doc,)* }
            }

            fn range(self, arg: Domain, params: Params<'_>) -> Domain {
                let range: RangeFn = match self { $(Function::$variant => $range,)* };
                range(arg, params)
            }

            /// What a call with checked `params` becomes at runtime.
            pub(super) fn instantiate(self, params: Params<'_>) -> Kind {
                let build: BuildFn = match self { $(Function::$variant => $build,)* };
                build(params)
            }
        }
    };
}

functions! {
    Abs {
        name: "abs", params: [], min_window: 1, requires: Any,
        range: |a, _| a.abs(),
        build: |_| Kind::Pure(Pure::Abs),
        doc: "Absolute value.",
    }
    Sign {
        name: "sign", params: [], min_window: 1, requires: Any,
        range: |a, _| a.sign(),
        build: |_| Kind::Pure(Pure::Sign),
        doc: "Sign: -1, 0 or 1.",
    }
    Sqrt {
        name: "sqrt", params: [], min_window: 1, requires: NonNegative,
        range: |a, _| a.map_increasing(f64::sqrt),
        build: |_| Kind::Pure(Pure::Sqrt),
        doc: "Square root.",
    }
    Square {
        name: "square", params: [], min_window: 1, requires: Any,
        range: |a, _| a.abs().map_increasing(|v| v * v),
        build: |_| Kind::Pure(Pure::Square),
        doc: "The argument times itself.",
    }
    Exp {
        name: "exp", params: [], min_window: 1, requires: Any,
        range: |a, _| a.map_increasing(f64::exp),
        build: |_| Kind::Pure(Pure::Exp),
        doc: "Exponential function; not ready where it overflows.",
    }
    Ln {
        name: "ln", params: [], min_window: 1, requires: Positive,
        range: |a, _| a.map_increasing(f64::ln),
        build: |_| Kind::Pure(Pure::Ln),
        doc: "Natural logarithm.",
    }
    Tanh {
        name: "tanh", params: [], min_window: 1, requires: Any,
        range: |a, _| a.map_increasing(f64::tanh),
        build: |_| Kind::Pure(Pure::Tanh),
        doc: "Hyperbolic tangent, squashes into (-1, 1).",
    }
    Sigmoid {
        name: "sigmoid", params: [], min_window: 1, requires: Any,
        range: |a, _| a.map_increasing(sigmoid),
        build: |_| Kind::Pure(Pure::Sigmoid),
        doc: "Logistic function 1 / (1 + exp(-a)), squashes into (0, 1).",
    }
    Asinh {
        name: "asinh", params: [], min_window: 1, requires: Any,
        range: |a, _| a.map_increasing(f64::asinh),
        build: |_| Kind::Pure(Pure::Asinh),
        doc: "Inverse hyperbolic sine: like a signed logarithm, compresses large magnitudes.",
    }
    Gte {
        name: "gte", params: [Real], min_window: 1, requires: Any,
        range: |a, p| a.max_with(p.real(0)),
        build: |p| Kind::Pure(Pure::Gte(p.real(0))),
        doc: "Clips values below the real to it: max(a, real).",
    }
    Lte {
        name: "lte", params: [Real], min_window: 1, requires: Any,
        range: |a, p| a.min_with(p.real(0)),
        build: |p| Kind::Pure(Pure::Lte(p.real(0))),
        doc: "Clips values above the real to it: min(a, real).",
    }
    LnReturn {
        name: "ln_return", params: [], min_window: 1, requires: Positive,
        range: |_, _| Domain::REAL,
        build: |_| LnReturn::new(Echo::new()).into(),
        doc: "Log return between consecutive updates.",
    }
    Drawdown {
        name: "drawdown", params: [], min_window: 1, requires: Positive,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |_| Drawdown::new(Echo::new()).into(),
        doc: "Largest relative peak-to-valley decline so far, in [0, 1].",
    }
    Sma {
        name: "sma", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Sma::new(Echo::new(), p.window(0)).into(),
        doc: "Simple moving average.",
    }
    Ema {
        name: "ema", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Ema::new(Echo::new(), p.window(0)).into(),
        doc: "Exponential moving average, alpha = 2 / (window + 1).",
    }
    Alma {
        name: "alma", params: [Window], min_window: 2, requires: Any,
        range: |a, _| a,
        build: |p| Alma::new(Echo::new(), p.window(0)).into(),
        doc: "Arnaud Legoux moving average (sigma 6, offset 0.85).",
    }
    Roc {
        name: "roc", params: [Window], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Roc::new(Echo::new(), p.window(0)).into(),
        doc: "Rate of change in percent against the value `window` updates ago.",
    }
    Lag {
        name: "lag", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Lag::new(Echo::new(), p.window(0)).into(),
        doc: "The value `window` updates ago.",
    }
    Min {
        name: "min", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Min::new(Echo::new(), p.window(0)).into(),
        doc: "Minimum over the window.",
    }
    Max {
        name: "max", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Max::new(Echo::new(), p.window(0)).into(),
        doc: "Maximum over the window.",
    }
    MinMax {
        name: "min_max", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| MinMaxNormalizer::new(Echo::new(), p.window(0)).into(),
        doc: "Position in the [min, max] range of the previous window, scaled to [-1, 1]; new extremes give -1 or 1.",
    }
    Mad {
        name: "mad", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| MadScaler::new(Echo::new(), p.window(0)).into(),
        doc: "Robust z-score: (a - median) / (1.4826 MAD) of the previous window.",
    }
    ZScore {
        name: "zscore", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| ZScoreStandardization::new(Echo::new(), p.window(0)).into(),
        doc: "Z-score against the mean and standard deviation of the previous window.",
    }
    Std {
        name: "std", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::non_negative(),
        build: |p| WelfordOnline::new(Echo::new(), p.window(0)).into(),
        doc: "Standard deviation over the window (Welford).",
    }
    Cumulative {
        name: "cumulative", params: [Window], min_window: 1, requires: Any,
        range: |a, p| a.mul(Domain::closed(1.0, window_f64(p.window(0)))),
        build: |p| Cumulative::new(Echo::new(), p.window(0)).into(),
        doc: "Sum over the window.",
    }
    Rsi {
        name: "rsi", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| Rsi::new(Echo::new(), p.window(0)).into(),
        doc: "Relative strength index, in [0, 100].",
    }
    MyRsi {
        name: "my_rsi", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| MyRSI::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' MyRSI, in [-1, 1].",
    }
    LaguerreRsi {
        name: "laguerre_rsi", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| LaguerreRSI::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' Laguerre RSI, in [0, 1].",
    }
    LaguerreFilter {
        name: "laguerre_filter", params: [Fraction], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| LaguerreFilter::new(Echo::new(), p.real(0)).into(),
        doc: "John Ehlers' Laguerre filter with damping gamma (the fraction).",
    }
    SuperSmoother {
        name: "super_smoother", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| SuperSmoother::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' two-pole super smoother low-pass filter.",
    }
    RoofingFilter {
        name: "roofing_filter", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| RoofingFilter::new(Echo::new(), p.window(0), p.window(1)).into(),
        doc: "John Ehlers' roofing filter (low-pass window, high-pass window): a band-pass.",
    }
    ReFlex {
        name: "re_flex", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| ReFlex::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' ReFlex zero-lag cycle indicator.",
    }
    TrendFlex {
        name: "trend_flex", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| TrendFlex::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' TrendFlex zero-lag trend indicator.",
    }
    CyberCycle {
        name: "cyber_cycle", params: [Window], min_window: 3, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| CyberCycle::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' cyber cycle.",
    }
    CenterOfGravity {
        name: "center_of_gravity", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| CenterOfGravity::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' center of gravity oscillator.",
    }
    Cti {
        name: "cti", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| CorrelationTrendIndicator::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' correlation trend indicator: correlation with a straight line, in [-1, 1].",
    }
    Pfe {
        name: "pfe", params: [Window, Window], min_window: 3, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| PolarizedFractalEfficiency::new(Echo::new(), ema(p.window(1)), p.window(0)).into(),
        doc: "Polarized fractal efficiency (window, EMA smoothing window), in [-1, 1]: near 1 in a straight up-trend, near -1 down.",
    }
    Fisher {
        name: "fisher", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| EhlersFisherTransform::new(Echo::new(), ema(p.window(1)), p.window(0)).into(),
        doc: "John Ehlers' Fisher transform (window, EMA smoothing window).",
    }
    BinaryEntropy {
        name: "binary_entropy", params: [Window], min_window: 2, requires: BothSigns,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| BinaryEntropy::new(Echo::new(), p.window(0)).into(),
        doc: "Shannon entropy of the signs over the window, in [0, 1].",
    }
    Vst {
        name: "vst", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Vst::new(Echo::new(), p.window(0)).into(),
        doc: "Variance stabilizing transform: change scaled by the rolling standard deviation.",
    }
    Net {
        name: "net", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| NoiseEliminationTechnology::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' noise elimination technology (Kendall correlation), in [-1, 1].",
    }
}

/// The EMA that smooths `pfe` and `fisher`.
fn ema(window: NonZeroUsize) -> Ema<f64, Echo<f64>> {
    Ema::new(Echo::new(), window)
}

/// The logistic function.
pub(super) fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// A window length as `f64`, exact for checked windows (at most 2^53).
fn window_f64(window: NonZeroUsize) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "checked windows are at most 2^53, exact as f64"
    )]
    let value = window.get() as f64;
    value
}

impl Function {
    /// The function called `name`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// The call signature, e.g. `sma(a, window)`.
    pub fn signature(self) -> String {
        let mut out = format!("{}(a", self.name());
        for param in self.params() {
            let _ = write!(out, ", {}", param.name());
        }
        out.push(')');
        out
    }

    /// Check parameters against the signature and `max_window`; the error
    /// says what is wrong.
    pub fn check_params(self, params: &[f64], max_window: usize) -> Result<(), String> {
        let expected = self.params();
        if params.len() != expected.len() {
            return Err(format!(
                "{} takes {} parameter(s) after its argument, got {}",
                self.signature(),
                expected.len(),
                params.len()
            ));
        }
        for (kind, value) in expected.iter().zip(params) {
            let valid = match kind {
                Param::Window => window_len(*value)
                    .is_some_and(|w| w.get() >= self.min_window() && w.get() <= max_window),
                Param::Real => value.is_finite(),
                Param::Fraction => *value > 0.0 && *value < 1.0,
            };
            if !valid {
                let expected = match kind {
                    Param::Window => {
                        format!("an integer in [{}, {max_window}]", self.min_window())
                    }
                    Param::Real => "a finite number".to_string(),
                    Param::Fraction => "a number strictly between 0 and 1".to_string(),
                };
                return Err(format!(
                    "{}: the {} parameter {value} must be {expected}",
                    self.signature(),
                    kind.name()
                ));
            }
        }
        Ok(())
    }

    /// The domain of the call's output for an argument in `arg`, `None` if
    /// no value of `arg` meets [`Function::requires`] (the call is never
    /// ready).
    ///
    /// # Panics
    ///
    /// If `params` do not pass [`Function::check_params`].
    pub fn output_domain(self, arg: Domain, params: &[f64]) -> Option<Domain> {
        let admitted = arg.intersect(self.requires().domain())?;
        Some(self.range(admitted, Params::new(self, params)))
    }

    /// An estimate of the admitted updates of its argument a call needs
    /// before it is ready: the sum of its windows (a log return needs one
    /// more value). Pipelines sum it along their longest chain.
    ///
    /// # Panics
    ///
    /// If `params` do not pass [`Function::check_params`].
    pub fn warm_up(self, params: &[f64]) -> usize {
        let params = Params::new(self, params);
        let windows: usize = self
            .params()
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Param::Window)
            .map(|(i, _)| params.window(i).get())
            .sum();
        windows + usize::from(self == Function::LnReturn)
    }

    /// A markdown table of every function: signature, what the argument
    /// must be, and description. Meant for documentation and for prompting
    /// an agent that writes pipelines.
    pub fn reference() -> String {
        let mut out = String::from("| function | argument | description |\n|---|---|---|\n");
        for function in Self::ALL {
            let _ = writeln!(
                out,
                "| `{}` | {} | {} |",
                function.signature(),
                function.requires().describe(),
                function.description()
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_round_trip() {
        for function in Function::ALL {
            assert_eq!(Function::from_name(function.name()), Some(*function));
            assert!(function.min_window() >= 1);
        }
        let mut names = Vec::from_iter(Function::ALL.iter().map(|f| f.name()));
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Function::ALL.len());
        assert_eq!(Function::from_name("nope"), None);
    }

    #[test]
    fn the_reference_lists_every_signature() {
        let reference = Function::reference();
        assert!(reference.contains("| `pfe(a, window, window)` | any |"));
        assert!(reference.contains("| `ln(a)` | positive |"));
        assert!(reference.contains("| `laguerre_filter(a, fraction)` |"));
        assert_eq!(reference.lines().count(), Function::ALL.len() + 2);
    }

    #[test]
    fn window_lengths() {
        assert_eq!(window_len(3.0).map(NonZeroUsize::get), Some(3));
        for bad in [0.0, -1.0, 2.5, f64::NAN, f64::INFINITY, 1e300] {
            assert_eq!(window_len(bad), None, "{bad}");
        }
    }

    #[test]
    fn parameter_checks() {
        assert!(Function::Pfe.check_params(&[3.0, 3.0], 100).is_ok());
        for (function, params) in [
            (Function::Sma, &[0.0][..]),
            (Function::Sma, &[2.5]),
            (Function::Sma, &[]),
            (Function::Sma, &[101.0]),
            (Function::Pfe, &[2.0, 3.0]),
            (Function::LaguerreFilter, &[1.0]),
            (Function::LaguerreFilter, &[0.0]),
            (Function::Gte, &[f64::NAN]),
            (Function::Abs, &[1.0]),
        ] {
            let err = function.check_params(params, 100).expect_err("invalid");
            assert!(err.starts_with(&function.signature()), "{err}");
        }
        let err = Function::Sma.check_params(&[0.0], 100).unwrap_err();
        assert_eq!(
            err,
            "sma(a, window): the window parameter 0 must be an integer in [1, 100]"
        );
    }

    #[test]
    fn output_domains() {
        let real = Domain::REAL;
        let pos = Domain::positive();
        assert_eq!(Function::Ln.output_domain(pos, &[]), Some(real));
        assert_eq!(Function::Ln.output_domain(real, &[]), Some(real));
        assert_eq!(Function::Ln.output_domain(pos.neg(), &[]), None);
        assert_eq!(
            Function::Sqrt.output_domain(real, &[]),
            Some(Domain::non_negative())
        );
        assert_eq!(
            Function::Sma.output_domain(Domain::closed(1.0, 2.0), &[5.0]),
            Some(Domain::closed(1.0, 2.0))
        );
        assert_eq!(
            Function::Cumulative.output_domain(Domain::closed(1.0, 2.0), &[5.0]),
            Some(Domain::closed(1.0, 10.0))
        );
        assert_eq!(
            Function::Gte.output_domain(real, &[0.0]),
            Some(Domain::non_negative())
        );
        assert_eq!(
            Function::Sigmoid.output_domain(real, &[]),
            Some(Domain::closed(0.0, 1.0)),
            "saturates in floating point"
        );
        assert_eq!(
            Function::Rsi.output_domain(real, &[14.0]),
            Some(Domain::closed(0.0, 100.0))
        );
    }

    #[test]
    fn binary_op_domains() {
        let pos = Domain::positive();
        assert_eq!(BinaryOp::Add.output_domain(pos, pos), Some(pos));
        assert_eq!(BinaryOp::Div.output_domain(pos, pos), Some(pos));
        assert_eq!(
            BinaryOp::Div.output_domain(pos, Domain::REAL),
            Some(Domain::REAL)
        );
        assert_eq!(
            BinaryOp::Div.output_domain(pos, Domain::point(0.0)),
            None,
            "dividing by zero is never finite"
        );
        assert_eq!(BinaryOp::Sub.symbol(), '-');
    }
}

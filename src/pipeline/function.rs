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
        AdaptiveLaguerreFilter,
        AdaptiveRsi,
        AdaptiveSuperSmoother,
        Alma,
        AutoTuneFilter,
        AutocorrelationPeriodogram,
        BandPass,
        BinaryEntropy,
        Butterworth,
        ButterworthPoles,
        CenterOfGravity,
        ContinuationIndex,
        CorrelationTrendIndicator,
        Cumulative,
        CyberCycle,
        Decycler,
        DecyclerOscillator,
        Dsma,
        EhlersFilter,
        EhlersFisherTransform,
        Ema,
        EmpiricalModeDecomposition,
        EvenBetterSinewave,
        GaussianFilter,
        GaussianPoles,
        HighPass,
        HighPassPoles,
        HilbertPhasor,
        HilbertTrendMode,
        HomodyneDominantCycle,
        InstantaneousTrendline,
        Lag,
        LaguerreFilter,
        LaguerreRSI,
        MadScaler,
        Mama,
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
        SineWave,
        Sma,
        SuperSmoother,
        TrendFlex,
        UltimateBands,
        UltimateSmoother,
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
    /// A finite real number above 0, e.g. a scale.
    Positive,
    /// A small integer choice in `[min, max]`, e.g. the number of poles of
    /// a filter. Unlike a window it does not count updates, so it adds
    /// nothing to [`Function::warm_up`].
    Integer {
        /// The smallest admitted value.
        min: u8,
        /// The largest admitted value.
        max: u8,
    },
}

impl Param {
    /// The name of the parameter kind in signatures.
    pub fn name(self) -> &'static str {
        match self {
            Param::Window => "window",
            Param::Real => "real",
            Param::Fraction => "fraction",
            Param::Positive => "positive",
            Param::Integer { .. } => "integer",
        }
    }

    /// Whether `value` is a valid parameter of this kind for a function
    /// whose windows lie in `[min_window, max_window]`.
    fn admits(self, value: f64, min_window: usize, max_window: usize) -> bool {
        match self {
            Param::Window => {
                window_len(value).is_some_and(|w| w.get() >= min_window && w.get() <= max_window)
            }
            Param::Real => value.is_finite(),
            Param::Fraction => value > 0.0 && value < 1.0,
            Param::Positive => value.is_finite() && value > 0.0,
            Param::Integer { min, max } => {
                value.fract() == 0.0 && value >= f64::from(min) && value <= f64::from(max)
            }
        }
    }

    /// What [`Param::admits`] admits, for error messages.
    fn expected(self, min_window: usize, max_window: usize) -> String {
        match self {
            Param::Window => format!("an integer in [{min_window}, {max_window}]"),
            Param::Real => "a finite number".to_string(),
            Param::Fraction => "a number strictly between 0 and 1".to_string(),
            Param::Positive => "a finite number above 0".to_string(),
            Param::Integer { min, max } => format!("an integer in [{min}, {max}]"),
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

/// The checked parameters of a call. The accessors assert that the table
/// reads each parameter as the kind it declares, so a wiring mistake fails
/// on the first call instead of misreading a value.
#[derive(Debug, Clone, Copy)]
pub(super) struct Params<'a> {
    kinds: &'static [Param],
    values: &'a [f64],
}

impl<'a> Params<'a> {
    /// Wrap parameters that passed the per-kind checks of
    /// [`Function::check_params`].
    pub(super) fn new(function: Function, values: &'a [f64]) -> Self {
        let kinds = function.params();
        assert_eq!(values.len(), kinds.len(), "checked parameters");
        Self { kinds, values }
    }

    /// The `i`-th parameter as a window length.
    pub(super) fn window(self, i: usize) -> NonZeroUsize {
        assert_eq!(self.kinds[i], Param::Window, "parameter {i} is a window");
        window_len(self.values[i]).expect("checked window")
    }

    /// The `i`-th parameter, a [`Param::Real`], [`Param::Fraction`] or
    /// [`Param::Positive`], as a real number.
    pub(super) fn real(self, i: usize) -> f64 {
        let kind = self.kinds[i];
        assert!(
            matches!(kind, Param::Real | Param::Fraction | Param::Positive),
            "parameter {i} is real, not {kind:?}"
        );
        let value = self.values[i];
        assert!(value.is_finite(), "checked parameter");
        value
    }

    /// The `i`-th parameter, a [`Param::Integer`], as an integer.
    pub(super) fn integer(self, i: usize) -> usize {
        let Param::Integer { min, max } = self.kinds[i] else {
            panic!("parameter {i} is {:?}, not an integer", self.kinds[i]);
        };
        let value = self.values[i];
        assert!(value.fract() == 0.0, "checked integer");
        assert!(value >= f64::from(min), "checked integer");
        assert!(value <= f64::from(max), "checked integer");
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a checked integer lies in [0, 255]"
        )]
        let integer = value as u8;
        usize::from(integer)
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

/// A constraint on parameters that passed their per-kind checks; the error
/// says what is wrong, without the signature.
type CheckFn = fn(Params<'_>) -> Result<(), String>;

/// The warm-up of a call given its checked parameters.
type WarmUpFn = fn(Params<'_>) -> usize;

/// Declares [`Function`] and its metadata from one table. Each entry has:
///
/// - `name`: what the text form calls it, snake_case and unique.
/// - `params`: the [`Param`] kinds after the expression argument, e.g.
///   `[Window, Fraction, Integer { min: 1, max: 2 }]`.
/// - `min_window`: the smallest value of every window parameter.
/// - `requires`: the [`Requirement`] on the argument.
/// - `range`: `|arg: Domain, p: Params<'_>| -> Domain`, the output domain for
///   an admitted argument in `arg`. It must be sound: the property tests
///   feed adversarial input and check every output against it.
/// - `build`: `|p: Params<'_>| -> Kind`, the runtime form. Its constructor
///   must not panic for any parameters that pass [`Function::check_params`].
/// - `check` (optional): `|p: Params<'_>| -> Result<(), String>`, the
///   constraints the per-kind checks cannot express: between parameters
///   (e.g. a bandwidth below a quarter of the period) or caps that keep a
///   constructor's asserts unreachable. The error is prefixed with the
///   signature.
/// - `warm_up` (optional): `|p: Params<'_>| -> usize`, the updates a call
///   needs before it is ready, when that exceeds the default (the sum of
///   the windows), e.g. for fixed warm-ups or chained stages.
/// - `doc`: one line naming every parameter in order and the output range.
macro_rules! functions {
    (@optional) => { None };
    (@optional $f:expr) => { Some($f) };
    ($($variant:ident {
        name: $name:literal,
        params: [$($param:ident $({ $($field:ident: $value:literal),* })?),*],
        min_window: $min:literal,
        requires: $requires:ident,
        range: $range:expr,
        build: $build:expr,
        $(check: $check:expr,)?
        $(warm_up: $warm_up:expr,)?
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
                match self {
                    $(Function::$variant => &[$(Param::$param $({ $($field: $value),* })?),*],)*
                }
            }

            /// The table's constraint between checked parameters, if any.
            fn check_fn(self) -> Option<CheckFn> {
                match self { $(Function::$variant => functions!(@optional $($check)?),)* }
            }

            /// The table's warm-up, if it overrides the default.
            fn warm_up_fn(self) -> Option<WarmUpFn> {
                match self { $(Function::$variant => functions!(@optional $($warm_up)?),)* }
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
    // John Ehlers' filters. Pole counts are `Integer` parameters mapped to
    // the views' enums; other settings use the views' documented defaults.
    HighPass {
        name: "high_pass", params: [Window, Integer { min: 1, max: 2 }], min_window: 5, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| HighPass::new(Echo::new(), p.window(0), high_pass_poles(p.integer(1))).into(),
        doc: "John Ehlers' high-pass filter (cutoff period window, poles 1 or 2): removes the trend, oscillates around 0.",
    }
    BandPass {
        name: "band_pass", params: [Window, Fraction], min_window: 3, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| BandPass::new(Echo::new(), p.window(0), p.real(1)).into(),
        check: |p| check_band_pass(p.window(0), p.real(1)),
        doc: "John Ehlers' two-pole band-pass filter (center period window at most 100000, relative bandwidth fraction from 0.01 to below window / 4): keeps cycles near the period, oscillates around 0.",
    }
    UltimateSmoother {
        name: "ultimate_smoother", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| UltimateSmoother::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' UltimateSmoother (critical period window): a low-pass with almost no lag that can overshoot the input.",
    }
    UltimateBands {
        name: "ultimate_bands", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| UltimateBands::new(Echo::new(), p.window(0), 1.0).into(),
        doc: "Center line of John Ehlers' Ultimate Bands (period window): the values of `ultimate_smoother`; the bands are not exposed.",
    }
    Decycler {
        name: "decycler", params: [Window, Integer { min: 1, max: 2 }], min_window: 5, requires: Any,
        // With one pole it is an EMA-like average of `(x + x[1]) / 2`.
        range: |a, p| if p.integer(1) == 1 { a } else { Domain::REAL },
        build: |p| Decycler::new(Echo::new(), p.window(0), high_pass_poles(p.integer(1))).into(),
        doc: "John Ehlers' decycler (cutoff period window, poles 1 or 2): the input minus its high-pass, a low-lag trend line; within the input range for 1 pole.",
    }
    DecyclerOscillator {
        name: "decycler_oscillator", params: [Window], min_window: 6, requires: Positive,
        range: |_, _| Domain::REAL,
        build: |p| DecyclerOscillator::new(Echo::new(), p.window(0), 1.0).into(),
        doc: "John Ehlers' decycler oscillator (high-pass period window, K = 1): the cycles between window / 2 and window in percent of the input, oscillates around 0.",
    }
    GaussianFilter {
        name: "gaussian_filter", params: [Window, Integer { min: 1, max: 4 }], min_window: 2, requires: Any,
        // A cascade of EMAs.
        range: |a, _| a,
        build: |p| GaussianFilter::new(Echo::new(), p.window(0), gaussian_poles(p.integer(1))).into(),
        doc: "John Ehlers' Gaussian low-pass filter (critical period window, poles 1 to 4): cascaded EMAs, within the input range.",
    }
    Butterworth {
        name: "butterworth", params: [Window, Integer { min: 2, max: 3 }], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Butterworth::new(Echo::new(), p.window(0), butterworth_poles(p.integer(1))).into(),
        check: |p| check_iir_period(p.window(0)),
        doc: "John Ehlers' Butterworth low-pass filter (critical period window at most 100000, poles 2 or 3): can overshoot the input.",
    }
    EhlersFilter {
        name: "ehlers_filter", params: [Window], min_window: 2, requires: Any,
        range: |a, _| a,
        build: |p| EhlersFilter::new(Echo::new(), p.window(0)).into(),
        // A 4-value smoother, then `window` smoothed values for each of the
        // `window` weights.
        warm_up: |p| 2 * p.window(0).get() + 3,
        doc: "John Ehlers' nonlinear filter (window): weights smoothed values by their distance to the recent past, within the input range; costs O(window) per update.",
    }
    Dsma {
        name: "dsma", params: [Window], min_window: 4, requires: Any,
        range: |a, _| a,
        build: |p| Dsma::new(Echo::new(), p.window(0)).into(),
        doc: "John Ehlers' deviation scaled moving average (period window): an EMA that speeds up on large moves, within the input range.",
    }
    ContinuationIndex {
        name: "continuation_index", params: [Window], min_window: 4, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| Box::new(ContinuationIndex::new(Echo::new(), 0.8, 8, p.window(0))).into(),
        doc: "John Ehlers' continuation index (length window; Laguerre gamma 0.8, order 8), in [-1, 1]: near 1 in an up-trend, near -1 in a down-trend.",
    }
    Emd {
        name: "emd", params: [Window, Positive], min_window: 5, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(EmpiricalModeDecomposition::new(Echo::new(), p.window(0), p.real(1), 0.1)).into(),
        check: check_emd,
        // The view also averages the band-pass peaks over 50 updates.
        warm_up: |p| (2 * p.window(0).get()).max(50),
        doc: "Trend of John Ehlers' empirical mode decomposition (band-pass period window at most 100000, half-bandwidth delta positive from 0.005 to below window / 8): the band-pass averaged over 2 windows, oscillates around 0.",
    }
    // John Ehlers' cycle measurements and adaptive indicators. The Hilbert
    // transform views and the autocorrelation periodogram have fixed
    // internals, so they take no parameters and override the warm-up.
    HomodyneDominantCycle {
        name: "homodyne_dominant_cycle", params: [], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(6.0, 50.0),
        build: |_| Box::new(HomodyneDominantCycle::new(Echo::new())).into(),
        warm_up: |_| HILBERT_PERIOD_WARM_UP,
        doc: "John Ehlers' dominant cycle period from the Hilbert transform homodyne discriminator (TA-Lib's HT_DCPERIOD), in [6, 50] bars; ready after 50 updates.",
    }
    HilbertPhasor {
        name: "hilbert_phasor", params: [], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(-180.0, 180.0),
        build: |_| Box::new(HilbertPhasor::new(Echo::new())).into(),
        warm_up: |_| HILBERT_PERIOD_WARM_UP,
        doc: "Angle of John Ehlers' Hilbert transform phasor (TA-Lib's HT_PHASOR) in degrees, in [-180, 180]: decreases by 360 / period per bar in a cycle; ready after 50 updates.",
    }
    SineWave {
        name: "sine_wave", params: [], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |_| Box::new(SineWave::new(Echo::new())).into(),
        warm_up: |_| HILBERT_CYCLE_WARM_UP,
        doc: "John Ehlers' Hilbert sine wave (TA-Lib's HT_SINE): the sine of the dominant cycle phase, in [-1, 1]; the lead sine is not exposed; ready after 63 updates.",
    }
    InstantaneousTrendline {
        name: "instantaneous_trendline", params: [], min_window: 1, requires: Any,
        // A WMA of averages of past inputs over one dominant cycle.
        range: |a, _| a,
        build: |_| Box::new(InstantaneousTrendline::new(Echo::new())).into(),
        warm_up: |_| HILBERT_CYCLE_WARM_UP,
        doc: "John Ehlers' instantaneous trendline (TA-Lib's HT_TRENDLINE): the input averaged over one measured dominant cycle, within the input range; ready after 63 updates.",
    }
    HilbertTrendMode {
        // Its 1.5% deviation rule is relative to the trendline, which is
        // only meaningful for positive prices: near zero it always says trend.
        name: "hilbert_trend_mode", params: [], min_window: 1, requires: Positive,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |_| Box::new(HilbertTrendMode::new(Echo::new())).into(),
        warm_up: |_| HILBERT_CYCLE_WARM_UP,
        doc: "John Ehlers' Hilbert transform trend mode (TA-Lib's HT_TRENDMODE) of a positive price: 1 in trend mode, 0 in cycle mode; ready after 63 updates.",
    }
    Mama {
        name: "mama", params: [Fraction, Fraction], min_window: 1, requires: Any,
        // An EMA with a smoothing factor in [slow, fast] that starts at the
        // first input.
        range: |a, _| a,
        build: |p| Box::new(Mama::with_limits(Echo::new(), p.real(0), p.real(1))).into(),
        check: |p| check_mama(p.real(0), p.real(1)),
        warm_up: |_| MAMA_WARM_UP,
        doc: "John Ehlers' MESA adaptive moving average (fast limit fraction, slow limit fraction at most the fast limit; Ehlers uses 0.5 and 0.05), within the input range; FAMA is not exposed; ready after 32 updates.",
    }
    AutocorrelationPeriodogram {
        name: "autocorrelation_periodogram", params: [], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(PERIODOGRAM_MIN_PERIOD, PERIODOGRAM_MAX_PERIOD),
        build: |_| Box::new(AutocorrelationPeriodogram::new(Echo::new())).into(),
        warm_up: |_| PERIODOGRAM_WARM_UP,
        doc: "John Ehlers' autocorrelation periodogram dominant cycle with the book's settings (periods 10 to 48, averaging 3 bars), in [10, 48] bars; ready after 109 updates.",
    }
    EvenBetterSinewave {
        name: "even_better_sinewave", params: [Window], min_window: 5, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| Box::new(EvenBetterSinewave::new(Echo::new(), p.window(0))).into(),
        // The high-pass, the 10-bar super smoother, then 3 filtered values.
        warm_up: |p| p.window(0).get() + 13,
        doc: "John Ehlers' even better sinewave (high-pass cutoff period window, 40 in the book; smoother 10): in [-1, 1], swings between -1 and 1 in cycles and stays near 1 (-1) in up (down) trends.",
    }
    AdaptiveRsi {
        name: "adaptive_rsi", params: [], min_window: 1, requires: Any,
        // The final super smoother can overshoot [0, 1].
        range: |_, _| Domain::REAL,
        build: |_| Box::new(AdaptiveRsi::new(Echo::new())).into(),
        // The smoother needs 10 more bars after the periodogram.
        warm_up: |_| PERIODOGRAM_WARM_UP + 10,
        doc: "John Ehlers' adaptive RSI over half the autocorrelation periodogram's dominant cycle (book settings): roughly in [0, 1] (can overshoot slightly), reference lines 0.3 and 0.7; ready after 119 updates.",
    }
    AdaptiveLaguerreFilter {
        name: "adaptive_laguerre_filter", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| AdaptiveLaguerreFilter::new(Echo::new(), p.window(0)).into(),
        // A median of 5 normalized tracking errors after the first window.
        warm_up: |p| p.window(0).get() + 4,
        doc: "John Ehlers' adaptive Laguerre filter (tracking error length window, 20 in the book): smooths heavily in ranges and follows trends; costs O(window) per update.",
    }
    AutoTuneFilter {
        name: "auto_tune_filter", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| AutoTuneFilter::new(Echo::new(), p.window(0)).into(),
        check: |p| check_quadratic_window(p.window(0)),
        // 2 windows of high-pass history, then the dominant cycle ramps up
        // by 2 bars per bar.
        warm_up: |p| 3 * p.window(0).get(),
        doc: "John Ehlers' AutoTune filter (correlation window at most 100, 20 in the paper): a band-pass tuned to twice the lag of the minimum autocorrelation, oscillates around 0; costs O(window^2) per update.",
    }
    AdaptiveSuperSmoother {
        name: "adaptive_super_smoother", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| AdaptiveSuperSmoother::new(Echo::new(), p.window(0), ADAPTIVE_RMS_LEN).into(),
        // Ready once the RMS of the rate of change covers its 81 values.
        warm_up: |_| ADAPTIVE_RMS_LEN.get(),
        doc: "John Ehlers' adaptive super smoother (longest period window, 20 in the paper; RMS length 81): shortens its period on fast moves, can overshoot the input; ready after 81 updates.",
    }
}

/// Updates before the Hilbert transform cycle period settles (see
/// [`HomodyneDominantCycle`]).
const HILBERT_PERIOD_WARM_UP: usize = 50;

/// Updates before the Hilbert transform views that also average over one
/// dominant cycle report: TA-Lib's lookback, see [`SineWave`].
const HILBERT_CYCLE_WARM_UP: usize = 63;

/// Updates before [`Mama`] reports: TA-Lib's lookback.
const MAMA_WARM_UP: usize = 32;

/// The book's period range of [`AutocorrelationPeriodogram::new`].
const PERIODOGRAM_MIN_PERIOD: f64 = 10.0;
const PERIODOGRAM_MAX_PERIOD: f64 = 48.0;

/// Updates before [`AutocorrelationPeriodogram::new`] reports: its 48-bar
/// high-pass, its 10-bar super smoother, then a filtered history of 48
/// lags plus 3 averaged bars.
const PERIODOGRAM_WARM_UP: usize = 48 + 10 + 48 + 3;

/// The RMS length of [`AdaptiveSuperSmoother`]'s rate of change, the
/// paper's value.
const ADAPTIVE_RMS_LEN: NonZeroUsize = NonZeroUsize::new(81).expect("81 is not zero");

/// The longest window of views that cost O(window^2) per update: 100
/// keeps an update at about 10^4 multiply-adds, while the maximum window
/// of a pipeline (100000) would cost 10^10.
const MAX_QUADRATIC_WINDOW: usize = 100;

/// `Err` if the `window` of a view that costs O(window^2) per update
/// exceeds [`MAX_QUADRATIC_WINDOW`].
fn check_quadratic_window(window: NonZeroUsize) -> Result<(), String> {
    if window.get() > MAX_QUADRATIC_WINDOW {
        return Err(format!(
            "the window {window} must be at most {MAX_QUADRATIC_WINDOW}, the cost per update grows with its square"
        ));
    }
    Ok(())
}

/// The constraint of [`Mama::with_limits`] the fraction checks do not
/// cover: a slow limit of at most the fast limit.
fn check_mama(fast_limit: f64, slow_limit: f64) -> Result<(), String> {
    if slow_limit > fast_limit {
        return Err(format!(
            "the slow limit {slow_limit} must be at most the fast limit {fast_limit}"
        ));
    }
    Ok(())
}

/// The longest period of the band-pass and Butterworth filters: their
/// recursions lose accuracy to rounding at long periods. The three-pole
/// Butterworth's gain is off by 0.1% at 1e5 and it diverges by 1e7, and a
/// band-pass pole radius rounds to 1 (unstable, which its constructor
/// rejects) once `bandwidth / period` is below about 1e-17.
const MAX_IIR_PERIOD: usize = 100_000;

/// The narrowest relative bandwidth of the band-pass filters: narrower
/// bands ring for more than 30 periods.
const MIN_BANDWIDTH: f64 = 0.01;

/// `Err` if a recursive filter's `period` exceeds [`MAX_IIR_PERIOD`].
fn check_iir_period(period: NonZeroUsize) -> Result<(), String> {
    if period.get() > MAX_IIR_PERIOD {
        return Err(format!(
            "the window {period} must be at most {MAX_IIR_PERIOD}, longer periods round the filter coefficients"
        ));
    }
    Ok(())
}

/// The constraints of [`BandPass`]: a bandwidth from [`MIN_BANDWIDTH`] to
/// below `period / 4` (so that `cos(2pi bandwidth / period) > 0` for a
/// stable filter), and a period of at most [`MAX_IIR_PERIOD`].
fn check_band_pass(period: NonZeroUsize, bandwidth: f64) -> Result<(), String> {
    if bandwidth < MIN_BANDWIDTH {
        return Err(format!(
            "the bandwidth {bandwidth} must be at least {MIN_BANDWIDTH}"
        ));
    }
    if 4.0 * bandwidth >= window_f64(period) {
        return Err(format!(
            "the bandwidth {bandwidth} must be below window / 4 = {}",
            window_f64(period) / 4.0
        ));
    }
    check_iir_period(period)
}

/// The constraints of [`EmpiricalModeDecomposition`], whose band-pass has
/// the bandwidth `2 delta`: those of [`check_band_pass`], except that the
/// view admits bandwidths of 1 and more.
fn check_emd(p: Params<'_>) -> Result<(), String> {
    let (period, delta) = (p.window(0), p.real(1));
    if delta < MIN_BANDWIDTH / 2.0 {
        return Err(format!(
            "the delta {delta} must be at least {}",
            MIN_BANDWIDTH / 2.0
        ));
    }
    if 8.0 * delta >= window_f64(period) {
        return Err(format!(
            "the delta {delta} must be below window / 8 = {}",
            window_f64(period) / 8.0
        ));
    }
    check_iir_period(period)
}

/// The [`HighPassPoles`] of a checked `Integer { min: 1, max: 2 }`.
fn high_pass_poles(poles: usize) -> HighPassPoles {
    match poles {
        1 => HighPassPoles::One,
        2 => HighPassPoles::Two,
        _ => unreachable!("checked poles {poles}"),
    }
}

/// The [`GaussianPoles`] of a checked `Integer { min: 1, max: 4 }`.
fn gaussian_poles(poles: usize) -> GaussianPoles {
    match poles {
        1 => GaussianPoles::One,
        2 => GaussianPoles::Two,
        3 => GaussianPoles::Three,
        4 => GaussianPoles::Four,
        _ => unreachable!("checked poles {poles}"),
    }
}

/// The [`ButterworthPoles`] of a checked `Integer { min: 2, max: 3 }`.
fn butterworth_poles(poles: usize) -> ButterworthPoles {
    match poles {
        2 => ButterworthPoles::Two,
        3 => ButterworthPoles::Three,
        _ => unreachable!("checked poles {poles}"),
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

    /// Check parameters against the signature, `max_window` and the
    /// function's own constraints; the error says what is wrong. Parameters
    /// that pass never make the runtime form panic.
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
        let min_window = self.min_window();
        for (kind, value) in expected.iter().zip(params) {
            if !kind.admits(*value, min_window, max_window) {
                return Err(format!(
                    "{}: the {} parameter {value} must be {}",
                    self.signature(),
                    kind.name(),
                    kind.expected(min_window, max_window)
                ));
            }
        }
        if let Some(check) = self.check_fn() {
            check(Params::new(self, params))
                .map_err(|message| format!("{}: {message}", self.signature()))?;
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
    /// before it is ready: by default the sum of its windows (a log return
    /// needs one more value), or the function's own estimate. Pipelines sum
    /// it along their longest chain.
    ///
    /// # Panics
    ///
    /// If `params` do not pass [`Function::check_params`].
    pub fn warm_up(self, params: &[f64]) -> usize {
        let params = Params::new(self, params);
        if let Some(warm_up) = self.warm_up_fn() {
            return warm_up(params);
        }
        let windows: usize = self
            .params()
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Param::Window)
            .map(|(i, _)| params.window(i).get())
            .sum();
        windows + usize::from(self == Function::LnReturn)
    }

    /// Parameters that pass the checks, at the small end: the `i`-th window
    /// is `min_window + i` (so "first window below the second" constraints
    /// hold), integers are at their minimum, reals are 0.5.
    #[cfg(test)]
    pub(super) fn example_params(self) -> Vec<f64> {
        self.test_params(self.min_window(), |min, _| min)
    }

    /// Parameters that pass the checks, larger than
    /// [`Function::example_params`]: the `i`-th window is `10 + i`, integers
    /// are at their maximum, reals are 0.5.
    #[cfg(test)]
    pub(super) fn larger_params(self) -> Vec<f64> {
        self.test_params(10.max(self.min_window()), |_, max| max)
    }

    #[cfg(test)]
    fn test_params(self, first_window: usize, integer: fn(u8, u8) -> u8) -> Vec<f64> {
        let mut next_window = first_window;
        Vec::from_iter(self.params().iter().map(|kind| match *kind {
            Param::Window => {
                let window = next_window;
                next_window += 1;
                f64::from(u32::try_from(window).expect("small test window"))
            }
            Param::Real | Param::Fraction | Param::Positive => 0.5,
            Param::Integer { min, max } => f64::from(integer(min, max)),
        }))
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
        assert!(reference.contains("| `high_pass(a, window, integer)` | any |"));
        assert!(reference.contains("| `emd(a, window, positive)` | any |"));
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
        assert!(Function::HighPass.check_params(&[5.0, 2.0], 100).is_ok());
        assert!(Function::Emd.check_params(&[5.0, 0.6], 100).is_ok());
        assert!(Function::BandPass.check_params(&[3.0, 0.7], 100).is_ok());
        assert!(Function::Mama.check_params(&[0.5, 0.5], 100).is_ok());
        assert!(
            Function::AutoTuneFilter
                .check_params(&[100.0], 1000)
                .is_ok()
        );
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
            // Integers: outside the range, or not whole.
            (Function::HighPass, &[5.0, 0.0]),
            (Function::HighPass, &[5.0, 3.0]),
            (Function::HighPass, &[5.0, 1.5]),
            (Function::Butterworth, &[5.0, 1.0]),
            // Positive reals.
            (Function::Emd, &[20.0, 0.0]),
            (Function::Emd, &[20.0, -1.0]),
            (Function::Emd, &[20.0, f64::INFINITY]),
            // The table's own checks, just past their boundaries.
            (Function::BandPass, &[3.0, 0.75]),
            (Function::BandPass, &[20.0, 0.009]),
            (Function::Emd, &[5.0, 0.625]),
            (Function::Emd, &[20.0, 0.004]),
            (Function::Mama, &[0.5, 0.51]),
            (Function::AutoTuneFilter, &[101.0]),
        ] {
            let Err(err) = function.check_params(params, 100) else {
                panic!("{function:?}{params:?} passed");
            };
            assert!(err.starts_with(&function.signature()), "{err}");
        }
        let err = Function::Sma.check_params(&[0.0], 100).unwrap_err();
        assert_eq!(
            err,
            "sma(a, window): the window parameter 0 must be an integer in [1, 100]"
        );
        let err = Function::HighPass
            .check_params(&[5.0, 3.0], 100)
            .unwrap_err();
        assert_eq!(
            err,
            "high_pass(a, window, integer): the integer parameter 3 must be an integer in [1, 2]"
        );
        let err = Function::Emd.check_params(&[5.0, 0.0], 100).unwrap_err();
        assert_eq!(
            err,
            "emd(a, window, positive): the positive parameter 0 must be a finite number above 0"
        );
        let err = Function::BandPass
            .check_params(&[3.0, 0.75], 100)
            .unwrap_err();
        assert_eq!(
            err,
            "band_pass(a, window, fraction): the bandwidth 0.75 must be below window / 4 = 0.75"
        );
        assert!(
            Function::Butterworth
                .check_params(&[100_000.0, 3.0], 1_000_000)
                .is_ok()
        );
        let err = Function::AutoTuneFilter
            .check_params(&[101.0], 1000)
            .unwrap_err();
        assert_eq!(
            err,
            "auto_tune_filter(a, window): the window 101 must be at most 100, \
             the cost per update grows with its square"
        );
        let err = Function::Mama.check_params(&[0.5, 0.51], 100).unwrap_err();
        assert_eq!(
            err,
            "mama(a, fraction, fraction): the slow limit 0.51 must be at most the fast limit 0.5"
        );
        let err = Function::Butterworth
            .check_params(&[100_001.0, 2.0], 1_000_000)
            .unwrap_err();
        assert_eq!(
            err,
            "butterworth(a, window, integer): the window 100001 must be at most 100000, \
             longer periods round the filter coefficients"
        );
    }

    #[test]
    fn example_params_pass_the_checks() {
        for function in Function::ALL {
            for params in [function.example_params(), function.larger_params()] {
                function
                    .check_params(&params, 100)
                    .unwrap_or_else(|e| panic!("{params:?}: {e}"));
            }
        }
        assert_eq!(Function::Pfe.example_params(), [3.0, 4.0]);
        assert_eq!(Function::Butterworth.example_params(), [2.0, 2.0]);
        assert_eq!(Function::Butterworth.larger_params(), [10.0, 3.0]);
        assert_eq!(Function::Emd.larger_params(), [10.0, 0.5]);
    }

    /// The constructors' asserts stay unreachable at the extremes of what
    /// passes the checks with unbounded windows. Only views with O(1)
    /// memory: the others would allocate their window.
    #[test]
    fn checked_extreme_parameters_construct() {
        let huge = 9_007_199_254_740_992.0;
        let max_iir = 100_000.0;
        for (function, params) in [
            (Function::HighPass, &[huge, 1.0][..]),
            (Function::HighPass, &[huge, 2.0]),
            (Function::HighPass, &[5.0, 1.0]),
            (Function::BandPass, &[max_iir, 0.01]),
            (Function::BandPass, &[max_iir, 0.999_999]),
            (Function::BandPass, &[3.0, 0.749_999]),
            (Function::BandPass, &[3.0, 0.01]),
            (Function::UltimateSmoother, &[huge]),
            (Function::Decycler, &[huge, 1.0]),
            (Function::Decycler, &[huge, 2.0]),
            (Function::DecyclerOscillator, &[huge]),
            (Function::GaussianFilter, &[huge, 1.0]),
            (Function::GaussianFilter, &[huge, 4.0]),
            (Function::GaussianFilter, &[2.0, 4.0]),
            (Function::Butterworth, &[max_iir, 2.0]),
            (Function::Butterworth, &[max_iir, 3.0]),
            (Function::Mama, &[0.999_999, 1e-9]),
            (Function::Mama, &[1e-9, 1e-9]),
            (Function::EvenBetterSinewave, &[huge]),
            (Function::EvenBetterSinewave, &[5.0]),
            (Function::AutoTuneFilter, &[2.0]),
            (Function::AutoTuneFilter, &[100.0]),
            (Function::AdaptiveSuperSmoother, &[huge]),
            (Function::AdaptiveSuperSmoother, &[2.0]),
        ] {
            function
                .check_params(params, usize::MAX)
                .unwrap_or_else(|e| panic!("{e}"));
            let Kind::Stateful(mut state) = function.instantiate(Params::new(function, params))
            else {
                panic!("{function:?} is a view");
            };
            for v in [1.0, 2.0, 1.5] {
                assert!(!state.update(v).is_infinite(), "{function:?}{params:?}");
            }
        }
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

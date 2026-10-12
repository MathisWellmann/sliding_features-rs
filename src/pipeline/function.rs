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
        AlphaBetaFilter,
        Bocpd,
        Cusum,
        Drawdown,
        HuberEma,
        Kalman1d,
        LnReturn,
        OneEuroFilter,
        PageHinkley,
        PageHinkleyDirection,
    },
    sliding_windows::{
        AdaptiveLaguerreFilter,
        AdaptiveRsi,
        AdaptiveSuperSmoother,
        Adwin,
        Alma,
        AutoTuneFilter,
        Autocorrelation,
        AutocorrelationPeriodogram,
        BandPass,
        BinaryEntropy,
        Butterworth,
        ButterworthPoles,
        CenterOfGravity,
        Cmo,
        CoefficientOfVariation,
        ConditionalValueAtRisk,
        ConnorsRsi,
        ContinuationIndex,
        Coppock,
        CorrelationTrendIndicator,
        Cumulative,
        CyberCycle,
        Decycler,
        DecyclerOscillator,
        Dema,
        Dpo,
        Dsma,
        EhlersFilter,
        EhlersFisherTransform,
        EhlersStochastic,
        Ehma,
        Ema,
        EmpiricalModeDecomposition,
        EvenBetterSinewave,
        EwmaVariance,
        GaussianFilter,
        GaussianPoles,
        HampelFilter,
        HighPass,
        HighPassPoles,
        HilbertPhasor,
        HilbertTrendMode,
        Hma,
        HomodyneDominantCycle,
        HurstExponent,
        InstantaneousTrendline,
        Iqr,
        Jma,
        Kama,
        KaufmanEfficiencyRatio,
        Kst,
        Kurtosis,
        Lag,
        LaguerreFilter,
        LaguerreRSI,
        LinRegAngle,
        LinRegIntercept,
        LinRegSlope,
        Macd,
        MadScaler,
        Mama,
        Max,
        McGinleyDynamic,
        Min,
        MinMaxNormalizer,
        MyRSI,
        NlmsFilter,
        NoiseEliminationTechnology,
        OuHalfLife,
        PercentileRank,
        PermutationEntropy,
        PolarizedFractalEfficiency,
        Ppo,
        QuantileNormalizer,
        RSquared,
        ReFlex,
        ReversionIndex,
        RlsFilter,
        Rma,
        Roc,
        RollingMedian,
        RollingQuantile,
        RoofingFilter,
        Rsi,
        Rsx,
        SampleEntropy,
        SchaffTrendCycle,
        ShannonEntropy,
        SharpeRatio,
        SineWave,
        Skewness,
        Sma,
        SortinoRatio,
        StandardError,
        StochRsi,
        SuperSmoother,
        T3,
        Tema,
        TrendFlex,
        Trima,
        Trix,
        Tsf,
        Tsi,
        UlcerIndex,
        UltimateSmoother,
        UniversalOscillator,
        ValueAtRisk,
        VarianceRatio,
        Vhf,
        Vidya,
        Vst,
        WelfordOnline,
        Wma,
        ZScoreStandardization,
        Zlema,
        inverse_normal_cdf,
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
        doc: "John Ehlers' roofing filter (high-pass cutoff period window, super smoother period window): a band-pass, e.g. roofing_filter(a, 48, 10).",
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
    // John Ehlers' oscillators.
    InverseFisher {
        name: "inverse_fisher", params: [Positive], min_window: 1, requires: Any,
        range: |a, p| {
            let gain = p.real(0);
            a.map_increasing(|v| (gain * v).tanh())
        },
        build: |p| Kind::Pure(Pure::InverseFisher(p.real(0))),
        doc: "John Ehlers' inverse Fisher transform tanh(gain * a) (gain positive): squashes an oscillator into [-1, 1], choose the gain so its typical range maps to about [-2, 2].",
    }
    EhlersStochastic {
        name: "ehlers_stochastic", params: [Window, Window, Window], min_window: 2, requires: Any,
        // The final super smoother can overshoot [0, 1].
        range: |_, _| Domain::REAL,
        build: |p| Box::new(EhlersStochastic::new(Echo::new(), p.window(0), p.window(1), p.window(2))).into(),
        // The roofing filter, then a full stochastic window, then its
        // super smoother of the same length.
        warm_up: |p| p.window(0).get() + p.window(1).get() + 2 * p.window(2).get(),
        doc: "John Ehlers' stochastic (roofing high-pass window, roofing smoother window, stochastic window): the roofing filtered input's position in its range over the stochastic window, super smoothed; roughly in [0, 1] (can overshoot slightly); costs O(stochastic window) per update.",
    }
    UniversalOscillator {
        name: "universal_oscillator", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| UniversalOscillator::new(Echo::new(), p.window(0)).into(),
        // Two inputs for the first momentum, then the super smoother.
        warm_up: |p| p.window(0).get() + 2,
        doc: "John Ehlers' universal oscillator (band edge window): super smoothed momentum divided by its decaying peak, in [-1, 1].",
    }
    ReversionIndex {
        name: "reversion_index", params: [Window, Window], min_window: 2, requires: Any,
        // The super smoother of a ratio in [-1, 1] can overshoot it.
        range: |_, _| Domain::REAL,
        build: |p| ReversionIndex::new(Echo::new(), p.window(0), p.window(1)).into(),
        doc: "John Ehlers' reversion index (length window, about half the dominant cycle; smoother window, 4 or 8 in the article): sum of changes over the sum of their absolute values, super smoothed; roughly in [-1, 1] (can overshoot slightly).",
    }
    // Moving averages. The convex combinations of past inputs stay within
    // the input range; the lag-reducing ones overshoot it on reversals.
    Wma {
        name: "wma", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Wma::new(Echo::new(), p.window(0)).into(),
        doc: "Linearly weighted moving average (window), within the input range.",
    }
    Rma {
        name: "rma", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Rma::new(Echo::new(), p.window(0)).into(),
        doc: "Wilder's moving average (window): an EMA with alpha = 1 / window seeded with the SMA, within the input range.",
    }
    Dema {
        name: "dema", params: [Window], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Dema::new(Echo::new(), p.window(0)).into(),
        // The outer EMA starts once the inner one is ready.
        warm_up: |p| 2 * p.window(0).get() - 1,
        doc: "Double exponential moving average 2 EMA - EMA(EMA) (window): less lag than an EMA, can overshoot the input.",
    }
    Tema {
        name: "tema", params: [Window], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(Tema::new(Echo::new(), p.window(0))).into(),
        warm_up: |p| 3 * p.window(0).get() - 2,
        doc: "Triple exponential moving average 3 EMA - 3 EMA(EMA) + EMA(EMA(EMA)) (window): less lag than DEMA, can overshoot the input.",
    }
    T3 {
        name: "t3", params: [Window, Fraction], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(T3::with_volume_factor(Echo::new(), p.window(0), p.real(1))).into(),
        // Six chained EMAs.
        warm_up: |p| 6 * p.window(0).get() - 5,
        doc: "Tillson's T3 (window, volume factor fraction, 0.7 classic): six chained EMAs, smoother than TEMA, can overshoot the input.",
    }
    Zlema {
        name: "zlema", params: [Window], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Zlema::new(Echo::new(), p.window(0)).into(),
        // The de-lagging reaches back (window - 1) / 2 updates.
        warm_up: |p| p.window(0).get() + (p.window(0).get() - 1) / 2,
        doc: "Zero-lag EMA (window): an EMA of 2 a - a[(window - 1) / 2], can overshoot the input.",
    }
    Hma {
        name: "hma", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(Hma::new(Echo::new(), p.window(0))).into(),
        warm_up: |p| hull_warm_up(p.window(0)),
        doc: "Hull moving average WMA(2 WMA(window / 2) - WMA(window), sqrt(window)) (window): low lag, can overshoot the input.",
    }
    Ehma {
        name: "ehma", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(Ehma::new(Echo::new(), p.window(0))).into(),
        warm_up: |p| hull_warm_up(p.window(0)),
        doc: "Exponential Hull moving average: `hma` with EMAs instead of WMAs (window), can overshoot the input.",
    }
    Trima {
        name: "trima", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Trima::new(Echo::new(), p.window(0)).into(),
        doc: "Triangular moving average (window): an SMA of an SMA, weights peak at the window center, within the input range.",
    }
    McGinleyDynamic {
        // Its speed adapts to the ratio of the input to the average.
        name: "mc_ginley_dynamic", params: [Window], min_window: 1, requires: Positive,
        range: |a, _| a,
        build: |p| McGinleyDynamic::new(Echo::new(), p.window(0)).into(),
        doc: "McGinley dynamic of a positive input (window; k 0.6): a moving average that speeds up when the input falls away from it, within the input range.",
    }
    Kama {
        name: "kama", params: [Window], min_window: 1, requires: Any,
        // An EMA whose smoothing factor stays in (0, 1).
        range: |a, _| a,
        build: |p| Kama::new(Echo::new(), p.window(0)).into(),
        // The efficiency ratio needs window changes.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Kaufman's adaptive moving average (efficiency ratio window; fast 2, slow 30): fast in clean trends, slow in noise, within the input range.",
    }
    Vidya {
        name: "vidya", params: [Window], min_window: 1, requires: Any,
        range: |a, _| a,
        build: |p| Vidya::new(Echo::new(), p.window(0)).into(),
        // The CMO needs window changes.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Chande's variable index dynamic average (window): an EMA scaled by |CMO| / 100 over the window, within the input range.",
    }
    Jma {
        name: "jma", params: [Window, Real], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Jma::with_params(Echo::new(), p.window(0), p.real(1), JMA_POWER).into(),
        check: |p| check_jma_phase(p.real(1)),
        doc: "Jurik moving average, the open reconstruction (period window, phase real in [-100, 100], 0 balanced, higher overshoots more; power 2): low lag, can overshoot the input.",
    }
    // Oscillators. Multi-output views expose their main line; signal lines
    // can be composed, e.g. `ema(macd(x, 12, 26), 9)`.
    Cmo {
        name: "cmo", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-100.0, 100.0),
        build: |p| Cmo::new(Echo::new(), p.window(0)).into(),
        // `window` changes need one more value.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Chande momentum oscillator (window): 100 (up - down) / (up + down) of the changes over the window, in [-100, 100].",
    }
    Macd {
        name: "macd", params: [Window, Window], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Box::new(Macd::with_periods(Echo::new(), p.window(0), p.window(1), MACD_SIGNAL)).into(),
        check: |p| check_fast_slow(p.window(0), p.window(1)),
        // Both EMAs run in parallel.
        warm_up: |p| p.window(1).get(),
        doc: "MACD line EMA(fast) - EMA(slow) (fast window below slow window, classic 12 and 26), TA-Lib's APO; in input units.",
    }
    Ppo {
        // A percentage of the slow EMA, which only has a meaning (and a
        // fixed sign) for positive prices.
        name: "ppo", params: [Window, Window], min_window: 1, requires: Positive,
        // The fast EMA of positive values is positive.
        range: |_, _| Domain::closed(-100.0, f64::INFINITY),
        build: |p| Box::new(Ppo::with_periods(Echo::new(), p.window(0), p.window(1), MACD_SIGNAL)).into(),
        check: |p| check_fast_slow(p.window(0), p.window(1)),
        warm_up: |p| p.window(1).get(),
        doc: "Percentage price oscillator of a positive price (fast window below slow window, classic 12 and 26): 100 (EMA(fast) - EMA(slow)) / EMA(slow), above -100.",
    }
    Trix {
        name: "trix", params: [Window], min_window: 2, requires: Positive,
        // The rate of change of a positive triple EMA.
        range: |_, _| Domain::closed(-100.0, f64::INFINITY),
        build: |p| Box::new(Trix::new(Echo::new(), p.window(0))).into(),
        // Three chained EMAs, then one more value for the rate of change.
        warm_up: |p| 3 * p.window(0).get() - 1,
        doc: "TRIX of a positive price (window): the 1-bar rate of change in percent of a triple EMA, above -100, oscillates around 0.",
    }
    Tsi {
        name: "tsi", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-100.0, 100.0),
        build: |p| Box::new(Tsi::with_periods(Echo::new(), p.window(0), p.window(1), TSI_SIGNAL)).into(),
        doc: "True strength index (long window, classic 25; short window, classic 13): double smoothed momentum over double smoothed absolute momentum, in [-100, 100].",
    }
    StochRsi {
        name: "stoch_rsi", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| Box::new(StochRsi::new(Echo::new(), p.window(0), p.window(1), STOCH_RSI_D)).into(),
        doc: "Stochastic RSI %K (RSI window, stochastic window; classic 14 and 14): the RSI's position in its range over the stochastic window, in [0, 1] (0 when flat).",
    }
    Rsx {
        name: "rsx", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| Rsx::new(Echo::new(), p.window(0)).into(),
        doc: "Jurik's RSX (window): a smoother RSI from triple smoothed momentum, in [0, 100]; reports 50 during its short warm-up.",
    }
    SchaffTrendCycle {
        name: "schaff_trend_cycle", params: [Window, Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| Box::new(SchaffTrendCycle::with_periods(Echo::new(), p.window(0), p.window(1), p.window(2))).into(),
        check: |p| check_fast_slow(p.window(0), p.window(1)),
        // The MACD, then two stochastic stages over the cycle.
        warm_up: |p| p.window(1).get() + 2 * p.window(2).get(),
        doc: "Schaff trend cycle (fast window below slow window, cycle window; classic 23, 50, 10): a double smoothed stochastic of the MACD, in [0, 100].",
    }
    ConnorsRsi {
        name: "connors_rsi", params: [Window, Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| Box::new(ConnorsRsi::with_params(Echo::new(), p.window(0), p.window(1), p.window(2))).into(),
        doc: "ConnorsRSI (RSI window, streak RSI window, percent rank window; classic 3, 2, 100): the mean of an RSI, an RSI of the up/down streak and the percent rank of the 1-bar change, in [0, 100]; costs O(log rank window) per update.",
    }
    Dpo {
        name: "dpo", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Dpo::new(Echo::new(), p.window(0)).into(),
        // The SMA, and the value window / 2 + 1 updates ago.
        warm_up: |p| p.window(0).get().max(p.window(0).get() / 2 + 2),
        doc: "Detrended price oscillator (window): the value window / 2 + 1 updates ago minus the SMA over the window, oscillates around 0.",
    }
    Coppock {
        name: "coppock", params: [Window, Window, Window], min_window: 1, requires: Positive,
        // A WMA of two rates of change of a positive price, each above -100.
        range: |_, _| Domain::closed(-200.0, f64::INFINITY),
        build: |p| Box::new(Coppock::with_periods(Echo::new(), p.window(0), p.window(1), p.window(2))).into(),
        // The view reports once its WMA is full, but its rates of change
        // compare against the first value until their windows are full.
        warm_up: |p| p.window(0).get().max(p.window(1).get()) + p.window(2).get(),
        doc: "Coppock curve of a positive price (long ROC window, short ROC window, WMA window; classic 14, 11, 10): a WMA of the sum of two rates of change in percent, above -200.",
    }
    Kst {
        name: "kst", params: [], min_window: 1, requires: Positive,
        // Weights 1 to 4 of SMAs of rates of change, each above -100.
        range: |_, _| Domain::closed(-1000.0, f64::INFINITY),
        build: |_| Box::new(Kst::new(Echo::new())).into(),
        // The slowest leg: ROC(30), then its SMA(15).
        warm_up: |_| KST_WARM_UP,
        doc: "Pring's know sure thing of a positive price with the classic settings (ROC 10, 15, 20, 30 smoothed by SMA 10, 10, 10, 15, weighted 1 to 4): above -1000, oscillates around 0; ready after 45 updates.",
    }
    // Trend strength and linear regression over a sliding window, with
    // x = 0 at the oldest value of the window.
    EfficiencyRatio {
        name: "efficiency_ratio", params: [Window], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| KaufmanEfficiencyRatio::new(Echo::new(), p.window(0)).into(),
        // `window` changes need one more value.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Kaufman's efficiency ratio (window): the net change over the window divided by the sum of the absolute changes, in [0, 1]: 1 in a straight trend, near 0 in noise.",
    }
    LinRegSlope {
        name: "lin_reg_slope", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| LinRegSlope::new(Echo::new(), p.window(0)).into(),
        doc: "Slope of the least-squares line over the window (window), TA-Lib's LINEARREG_SLOPE: input units per update.",
    }
    LinRegIntercept {
        name: "lin_reg_intercept", params: [Window], min_window: 2, requires: Any,
        // The fitted value at the oldest update can lie outside the input
        // range.
        range: |_, _| Domain::REAL,
        build: |p| LinRegIntercept::new(Echo::new(), p.window(0)).into(),
        doc: "Intercept of the least-squares line over the window (window), TA-Lib's LINEARREG_INTERCEPT: the fitted value at the oldest update of the window.",
    }
    LinRegAngle {
        name: "lin_reg_angle", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(-90.0, 90.0),
        build: |p| LinRegAngle::new(Echo::new(), p.window(0)).into(),
        doc: "Angle atan(slope) in degrees of the least-squares line over the window (window), TA-Lib's LINEARREG_ANGLE, in [-90, 90]; depends on the input scale.",
    }
    RSquared {
        name: "r_squared", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| RSquared::new(Echo::new(), p.window(0)).into(),
        doc: "Coefficient of determination R^2 of the least-squares line over the window (window), in [0, 1]: 1 when a line explains the window (or it is flat), 0 when no better than the mean.",
    }
    Tsf {
        name: "tsf", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| Tsf::new(Echo::new(), p.window(0)).into(),
        doc: "Time series forecast (window), TA-Lib's TSF: the least-squares line over the window extended one update past the newest value; can overshoot the input.",
    }
    StandardError {
        name: "standard_error", params: [Window], min_window: 3, requires: Any,
        range: |_, _| Domain::non_negative(),
        build: |p| StandardError::new(Echo::new(), p.window(0)).into(),
        doc: "Standard error of the least-squares line over the window (window): sqrt(SSE / (window - 2)) of the residuals, non-negative, in input units.",
    }
    Vhf {
        name: "vhf", params: [Window], min_window: 2, requires: Any,
        // The range of the window is at most the path between its extremes,
        // which the sum of the absolute changes covers.
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| Vhf::new(Echo::new(), p.window(0)).into(),
        // The oldest change needs one value before the window.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Vertical horizontal filter (window): the range of the window over the sum of its absolute changes, in [0, 1]: high in trends, low in ranges; costs O(window) per update.",
    }
    // Rolling statistics. The order statistics keep a sorted copy of the
    // window, whose insert and removal move O(window) values per update.
    Quantile {
        name: "quantile", params: [Window, Fraction], min_window: 2, requires: Any,
        // An interpolation between two values of the window.
        range: |a, _| a,
        build: |p| RollingQuantile::new(Echo::new(), p.window(0), p.real(1)).into(),
        doc: "Quantile of the window (window, q fraction, e.g. 0.25), interpolated linearly between order statistics as numpy and R do, within the input range; `min` and `max` give q = 0 and 1; costs O(window) per update.",
    }
    Median {
        name: "median", params: [Window], min_window: 2, requires: Any,
        range: |a, _| a,
        build: |p| RollingMedian::new(Echo::new(), p.window(0)).into(),
        doc: "Median of the window (window; the mean of the two middle values for an even window), robust to outliers, within the input range; costs O(window) per update.",
    }
    Iqr {
        name: "iqr", params: [Window], min_window: 2, requires: Any,
        // Q3 - Q1 is at most the range of the window.
        range: |a, _| Domain::closed(0.0, a.hi() - a.lo()),
        build: |p| Iqr::new(Echo::new(), p.window(0)).into(),
        doc: "Interquartile range Q3 - Q1 of the window (window; quantiles as `quantile`): a robust dispersion in input units, non-negative and at most the range of the input; costs O(window) per update.",
    }
    PercentileRank {
        name: "percentile_rank", params: [Window], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| PercentileRank::new(Echo::new(), p.window(0)).into(),
        doc: "Percentile rank of the newest value in the window that includes it (window), in [0, 100]: the mean of the shares of values below it and at most it (scipy's `mean` kind), 50 for a flat window; costs O(window) per update.",
    }
    GaussianRank {
        name: "gaussian_rank", params: [Window], min_window: 2, requires: Any,
        range: |_, p| gaussian_rank_range(p.window(0)),
        build: |p| QuantileNormalizer::new(Echo::new(), p.window(0)).into(),
        doc: "Van der Waerden normal score of the newest value in the window that includes it (window): the standard normal quantile of its mid-rank / (window + 1), a robust z-score that is roughly standard normal, within +-Phi^-1(window / (window + 1)) (+-1.7 for 20); costs O(window) per update.",
    }
    Skewness {
        name: "skewness", params: [Window], min_window: 3, requires: Any,
        // |G1| <= sqrt(window) holds in exact arithmetic only: a window that
        // is flat up to rounding gives a ratio of round-off errors.
        range: |_, _| Domain::REAL,
        build: |p| Skewness::new(Echo::new(), p.window(0)).into(),
        doc: "Sample skewness G1 of the window (window; bias adjusted, as Excel's SKEW and pandas): 0 for symmetric or flat windows, positive for a long right tail.",
    }
    Kurtosis {
        name: "kurtosis", params: [Window], min_window: 4, requires: Any,
        // Bounded in exact arithmetic only, like the skewness.
        range: |_, _| Domain::REAL,
        build: |p| Kurtosis::new(Echo::new(), p.window(0)).into(),
        doc: "Sample excess kurtosis G2 of the window (window; bias adjusted, as Excel's KURT and pandas): 0 for normal data and flat windows, -1.2 for uniform, positive for heavy tails.",
    }
    CoefficientOfVariation {
        // A dispersion relative to the level, which only has a meaning on a
        // ratio scale: near a zero mean it explodes.
        name: "coefficient_of_variation", params: [Window], min_window: 2, requires: Positive,
        range: |_, _| Domain::non_negative(),
        build: |p| CoefficientOfVariation::new(Echo::new(), p.window(0)).into(),
        doc: "Coefficient of variation of a positive input over the window (window): the sample standard deviation divided by the mean, non-negative and unitless.",
    }
    EwmaVariance {
        name: "ewma_variance", params: [Window], min_window: 2, requires: Any,
        // A weighted variance of past inputs, at most (range / 2)^2 by
        // Popoviciu's inequality.
        range: |a, _| {
            let half_width = (a.hi() - a.lo()) / 2.0;
            Domain::closed(0.0, half_width * half_width)
        },
        build: |p| EwmaVariance::new(Echo::new(), p.window(0)).into(),
        doc: "Exponentially weighted variance (window; alpha = 2 / (window + 1) as `ema`, without bias correction, RiskMetrics style): non-negative, in squared input units; ready after window updates.",
    }
    // Power transforms: stateless and increasing in the argument for every
    // lambda.
    BoxCox {
        name: "box_cox", params: [Real], min_window: 1, requires: Positive,
        range: |a, p| {
            let lambda = p.real(0);
            a.map_increasing(|v| box_cox(v, lambda))
        },
        build: |p| Kind::Pure(Pure::BoxCox(p.real(0))),
        doc: "Box-Cox transform (a^lambda - 1) / lambda of a positive input (lambda real, usually in [-2, 2]; ln(a) at 0, a - 1 at 1): makes right-skewed data more normal; not ready where the power overflows.",
    }
    YeoJohnson {
        name: "yeo_johnson", params: [Real], min_window: 1, requires: Any,
        range: |a, p| {
            let lambda = p.real(0);
            a.map_increasing(|v| yeo_johnson(v, lambda))
        },
        build: |p| Kind::Pure(Pure::YeoJohnson(p.real(0))),
        doc: "Yeo-Johnson transform (lambda real, usually in [-2, 2]; the identity at 1): `box_cox(a + 1, lambda)` for a >= 0 and its mirror image with 2 - lambda below 0, so it takes signed input such as returns; not ready where the power overflows.",
    }
    // Complexity, memory and dependence measures over a sliding window.
    // Horizons that must stay below the window (a lag, an aggregation
    // period) come first, so that the shorter horizon reads first.
    ShannonEntropy {
        name: "shannon_entropy", params: [Window, Integer { min: 2, max: 100 }, Real, Positive], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        // The range is `lo` and a width rather than `lo` and `hi`, so that
        // every pair of finite parameters orders its ends.
        build: |p| ShannonEntropy::new(Echo::new(), p.window(0), count(p.integer(1)), p.real(2), p.real(2) + p.real(3)).into(),
        check: |p| check_bin_range(p.real(2), p.real(3)),
        doc: "Normalized Shannon entropy of the window (window, bins integer from 2 to 100, lo real, width positive) binned into equal bins over the fixed range [lo, lo + width], values outside clamped into the end bins, in [0, 1]: 0 when all values share a bin, 1 when spread evenly; normalize the input first, e.g. `shannon_entropy(min_max(x, 50), 100, 10, -1, 2)`.",
    }
    PermutationEntropy {
        name: "permutation_entropy", params: [Window, Integer { min: 2, max: 7 }], min_window: 2, requires: Any,
        range: |_, _| Domain::closed(0.0, 1.0),
        build: |p| PermutationEntropy::new(Echo::new(), p.window(0), p.integer(1), PERMUTATION_DELAY).into(),
        // The first pattern needs `order` values.
        warm_up: |p| p.window(0).get() + p.integer(1) - 1,
        doc: "Bandt and Pompe's normalized permutation entropy (window of patterns, order integer from 2 to 7; delay 1): the entropy of the ordinal patterns of `order` consecutive values over the window, in [0, 1]: 0 for a monotone (or flat) series, near 1 for noise; wants a window well above order!.",
    }
    SampleEntropy {
        // Shorter windows have so few template pairs that the entropy is
        // mostly undefined (no pair matches) or 0. 20 also covers the
        // constructor's `window >= m + 2`.
        name: "sample_entropy", params: [Window, Integer { min: 1, max: 5 }, Positive], min_window: 20, requires: Any,
        range: |_, p| Domain::closed(0.0, sample_entropy_max(p.window(0), p.integer(1))),
        build: |p| SampleEntropy::new(Echo::new(), p.window(0), count(p.integer(1)), p.real(2)).into(),
        check: |p| check_quadratic_window(p.window(0)),
        doc: "Richman and Moorman's sample entropy -ln(A / B) of the window (window from 20 to 100, template length m integer from 1 to 5, usually 2; tolerance r positive times the window's standard deviation, usually 0.2): low for regular series; in [0, ln(P)] for the P pairs of templates; not ready while no template pair matches at length m + 1; costs O(window^2 m) per update.",
    }
    Hurst {
        name: "hurst", params: [Window], min_window: 32, requires: Any,
        // The regression slope is not clamped.
        range: |_, _| Domain::REAL,
        build: |p| HurstExponent::new(Echo::new(), p.window(0)).into(),
        check: |p| check_log_linear_window(p.window(0)),
        doc: "Hurst exponent of returns, not levels (window from 32 to 1000), by rescaled range analysis over chunks of 8, 16, 32, ...: about 0.5 for uncorrelated returns, above for trending, below for mean-reverting; typically in [0, 1] but not clamped; not ready for a flat window; costs O(window log window) per update.",
    }
    VarianceRatio {
        name: "variance_ratio", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, _| Domain::non_negative(),
        build: |p| VarianceRatio::new(Echo::new(), p.window(1), p.window(0).get()).into(),
        check: |p| check_horizon("period", p.window(0), p.window(1)),
        // The period's sums lie within the window.
        warm_up: |p| p.window(1).get(),
        doc: "Lo and MacKinlay's variance ratio of returns, not levels (period q window, window above q): the variance of overlapping q-period returns over q times that of 1-period returns, non-negative: about 1 for a random walk, above for trending, below for mean-reverting; not ready for flat returns.",
    }
    Autocorrelation {
        name: "autocorrelation", params: [Window, Window], min_window: 1, requires: Any,
        range: |_, _| Domain::closed(-1.0, 1.0),
        build: |p| Autocorrelation::new(Echo::new(), p.window(1), p.window(0)).into(),
        check: |p| check_horizon("lag", p.window(0), p.window(1)),
        warm_up: |p| p.window(1).get(),
        doc: "Sample autocorrelation (lag window, window above the lag) against the mean of the whole window (Box-Jenkins, biased towards 0 by (window - lag) / window), in [-1, 1]; 0 for a flat window.",
    }
    OuHalfLife {
        name: "ou_half_life", params: [Window], min_window: 3, requires: Any,
        // The view reports exactly 0 for an AR(1) coefficient of 0.
        range: |_, _| Domain::non_negative(),
        build: |p| OuHalfLife::new(Echo::new(), p.window(0)).into(),
        // `window` changes need one more value.
        warm_up: |p| p.window(0).get() + 1,
        doc: "Mean-reversion half-life in updates of levels, not returns (window of changes): -ln 2 / ln |phi| of the AR(1) coefficient phi fitted by OLS over the window, non-negative; not ready without mean reversion (|phi| >= 1) or for a flat window.",
    }
    // Estimators and robust filters, ready from their first value unless
    // they predict from a tap history.
    OneEuro {
        name: "one_euro", params: [Positive, Real], min_window: 1, requires: Any,
        // An EMA with an adaptive factor in [0, 1]: within the input's hull.
        range: |a, _| a,
        build: |p| OneEuroFilter::new(Echo::new(), p.real(0), p.real(1), ONE_EURO_D_CUTOFF).into(),
        check: |p| check_non_negative("beta", p.real(1)),
        doc: "Casiez's 1 euro filter (min_cutoff positive, beta non-negative real; derivative cutoff 1, one update per time unit): an EMA whose cutoff frequency, in cycles per update, grows from min_cutoff by beta times the smoothed speed, so it lags little in moves and jitters little at rest; within the input's range.",
    }
    AlphaBeta {
        name: "alpha_beta", params: [Fraction, Positive], min_window: 1, requires: Any,
        // The constant-velocity model overshoots turns.
        range: |_, _| Domain::REAL,
        build: |p| AlphaBetaFilter::new(Echo::new(), p.real(0), p.real(1)).into(),
        check: |p| check_alpha_beta(p.real(0), p.real(1)),
        doc: "Alpha-beta (g-h) tracker's position estimate (alpha fraction position gain, beta positive velocity gain, at most 2 and with 4 - 2 alpha - beta > 0 for stability): a fixed-gain constant-velocity Kalman filter; overshoots turns, so any real.",
    }
    Kalman {
        name: "kalman", params: [Positive, Positive], min_window: 1, requires: Any,
        // The gain is in [0, 1]: within the input's hull.
        range: |a, _| a,
        build: |p| Kalman1d::new(Echo::new(), p.real(0), p.real(1)).into(),
        check: |p| check_kalman(p.real(0), p.real(1)),
        doc: "Scalar Kalman filter of the local-level (random walk plus noise) model (q positive process variance, r positive measurement variance, in squared input units; only q / r matters): an EMA whose gain settles where q / r puts it; within the input's range.",
    }
    HuberEma {
        name: "huber_ema", params: [Fraction, Fraction, Positive], min_window: 1, requires: Any,
        // Each step moves the level towards the value by at most alpha of
        // the residual: within the input's hull.
        range: |a, _| a,
        build: |p| HuberEma::new(Echo::new(), p.real(0), p.real(1), p.real(2)).into(),
        doc: "Robust EMA of Gelper, Fried and Croux (level_alpha fraction, scale_alpha fraction, k positive, usually 1 to 3): residuals beyond k times an EMA of absolute residuals are clipped, bounding the pull of outliers; within the input's range.",
    }
    Nlms {
        name: "nlms", params: [Integer { min: 1, max: 64 }, Positive], min_window: 1, requires: Any,
        // A linear predictor extrapolates.
        range: |_, _| Domain::REAL,
        build: |p| NlmsFilter::new(Echo::new(), count(p.integer(0)), p.real(1), NLMS_EPSILON).into(),
        check: |p| check_nlms_mu(p.real(1)),
        // The first weight update needs `order` values before it.
        warm_up: |p| p.integer(0) + 1,
        doc: "Normalized LMS adaptive linear predictor's forecast of the next value (order integer taps from 1 to 64, mu positive step below 2, scale-free; regularizer 1e-12): the weights adapt by a gradient step normalized by the tap energy; any real. Plain LMS is not offered: its step must suit the unknown input power or it diverges.",
    }
    Rls {
        name: "rls", params: [Integer { min: 1, max: 32 }, Fraction], min_window: 1, requires: Any,
        range: |_, _| Domain::REAL,
        build: |p| RlsFilter::new(Echo::new(), count(p.integer(0)), p.real(1), RLS_DELTA).into(),
        warm_up: |p| p.integer(0) + 1,
        doc: "Recursive least squares adaptive linear predictor's forecast of the next value (order integer taps from 1 to 32, lambda fraction forgetting factor, usually 0.95 to 0.999; initial inverse correlation 100 I): exponentially weighted least squares, converging faster than nlms at O(order^2) per update; any real.",
    }
    Hampel {
        name: "hampel", params: [Window, Positive], min_window: 3, requires: Any,
        // The value or the window's median.
        range: |a, _| a,
        build: |p| HampelFilter::new(Echo::new(), p.window(0), p.real(1)).into(),
        check: |p| check_log_linear_window(p.window(0)),
        doc: "Causal Hampel filter (window from 3 to 1000, t positive threshold, usually 3): the value, or the window's median if the value lies more than t scaled MADs (1.4826 MAD) from it; within the input's range; costs O(window log window) per update.",
    }
    // Change detectors and risk measures. The detectors report their
    // statistic, not their alarms: an alarm is the statistic crossing the
    // threshold, which `gte`/`lte` and arithmetic express.
    Cusum {
        name: "cusum", params: [Real, Positive], min_window: 1, requires: Any,
        range: |a, p| cusum_range(a, p.real(0), p.real(1)),
        build: |p| Cusum::new(Echo::new(), 0.0, 1.0, p.real(0), p.real(1)).into(),
        check: |p| check_non_negative("k", p.real(0)),
        doc: "Two-sided tabular CUSUM of standardized input, in-control mean 0 and standard deviation 1 (k non-negative real slack, h positive threshold; Montgomery's k = 0.5, h = 5): the signed statistic S+ if S+ >= S-, else -S-, both reset after one exceeds h, which it still reports; standardize the input first, e.g. `cusum((x - 0.001) / 0.02, 0.5, 5)` or `cusum(zscore(x, 100), 0.5, 5)`.",
    }
    PageHinkley {
        name: "page_hinkley", params: [Window, Real, Positive, Integer { min: 0, max: 2 }], min_window: 1, requires: Any,
        range: |a, p| page_hinkley_range(a, p),
        build: |p| PageHinkley::new(Echo::new(), page_hinkley_direction(p.integer(3)), p.window(0), p.real(1), p.real(2), PAGE_HINKLEY_ALPHA).into(),
        check: |p| check_non_negative("delta", p.real(1)),
        doc: "Page-Hinkley test for shifts of the mean (min_instances window before alarms, delta non-negative real tolerated change, lambda positive threshold, both in input units; direction integer: 0 both, 1 increases, 2 decreases; forgetting 0.9999 as scikit-multiflow): the cumulative deviation from the running mean, positive for increases and negative for decreases, with mean and statistics reset after it exceeds lambda, which it still reports.",
    }
    Bocpd {
        name: "bocpd", params: [Window, Window], min_window: 2, requires: Any,
        range: |_, p| Domain::closed(0.0, window_f64(p.window(1))),
        build: |p| Bocpd::new(Echo::new(), window_f64(p.window(0)), 0.0, 1.0, 1.0, 1.0, p.window(1).get()).into(),
        check: |p| check_max_run_length(p.window(1)),
        // Ready from the first value; neither window is a span of history.
        warm_up: |_| 0,
        doc: "Adams and MacKay's Bayesian online changepoint detection of mean and variance of standardized input (lambda window expected run length, max_run_length window at most 1000; prior mean 0 worth 1 value, variance prior Inverse-Gamma(1, 1)): the most probable number of values since the last changepoint, in [0, max_run_length]; costs O(max_run_length) per update.",
    }
    Adwin {
        name: "adwin", params: [Fraction], min_window: 1, requires: Any,
        // The mean of the retained values.
        range: |a, _| a,
        build: |p| Adwin::new(Echo::new(), p.real(0), ADWIN_MAX_BUCKETS, ADWIN_CLOCK).into(),
        doc: "Bifet and Gavalda's ADWIN2 (delta fraction confidence, MOA's 0.002; 5 buckets per level, cut checks every 32 updates as MOA): the mean of an adaptive window that drops its older part when its mean differs significantly from the newer one, within the input's range.",
    }
    Var {
        name: "var", params: [Window, Fraction], min_window: 2, requires: Any,
        // The negated quantile of the window.
        range: |a, _| a.neg(),
        build: |p| ValueAtRisk::new(Echo::new(), p.window(0), p.real(1)).into(),
        doc: "Historical value at risk of returns, not prices (window, confidence fraction, e.g. 0.95): minus the 1 - confidence quantile of the window (as `quantile`), so a loss is positive; within the negated input range; costs O(window) per update.",
    }
    Cvar {
        name: "cvar", params: [Window, Fraction], min_window: 2, requires: Any,
        // Minus the mean of the lowest values of the window.
        range: |a, _| a.neg(),
        build: |p| ConditionalValueAtRisk::new(Echo::new(), p.window(0), p.real(1)).into(),
        doc: "Historical conditional value at risk (expected shortfall) of returns, not prices (window, confidence fraction, e.g. 0.975): minus the mean of the returns at or below the 1 - confidence quantile of the window, so a loss is positive, at least `var`; within the negated input range; costs O(window) per update.",
    }
    UlcerIndex {
        name: "ulcer_index", params: [Window], min_window: 2, requires: Positive,
        // The root mean square of drawdowns in percent, each in (-100, 0].
        range: |_, _| Domain::closed(0.0, 100.0),
        build: |p| UlcerIndex::new(Echo::new(), p.window(0)).into(),
        // A drawdown needs a full window of prices, the mean a full window
        // of drawdowns.
        warm_up: |p| 2 * p.window(0).get() - 1,
        doc: "Martin's Ulcer Index of prices, not returns (window): the root mean square of the percent drawdowns from the rolling window maximum over the window, in [0, 100]: 0 without drawdowns.",
    }
    Sharpe {
        name: "sharpe", params: [Window, Real], min_window: 2, requires: Any,
        // Unbounded as the standard deviation vanishes.
        range: |_, _| Domain::REAL,
        build: |p| SharpeRatio::new(Echo::new(), p.window(0), p.real(1)).into(),
        check: |p| check_rate("risk_free", p.real(1)),
        doc: "Per-period Sharpe ratio of returns, not prices (window, risk_free real rate per period, at most 1e6 in magnitude): the mean excess return over the sample standard deviation of the window, not annualized; not ready for a flat window.",
    }
    Sortino {
        name: "sortino", params: [Window, Real], min_window: 2, requires: Any,
        // The mean excess return is at least minus the root mean square of
        // its negative parts, the downside deviation.
        range: |_, _| Domain::closed(-1.0, f64::INFINITY),
        build: |p| SortinoRatio::new(Echo::new(), p.window(0), p.real(1)).into(),
        check: |p| check_rate("target", p.real(1)),
        doc: "Per-period Sortino ratio of returns, not prices (window, target real minimum acceptable return per period, e.g. 0, at most 1e6 in magnitude): the mean excess return over the downside deviation of the window (over all its returns), not annualized, at least -1; not ready while no return of the window is below the target.",
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

/// The signal period of [`Macd`] and [`Ppo`], unused by the pipeline
/// but required by their constructors: Appel's 9.
const MACD_SIGNAL: NonZeroUsize = NonZeroUsize::new(9).expect("9 is not zero");

/// The signal period of [`Tsi`], unused by the pipeline: the view's
/// default.
const TSI_SIGNAL: NonZeroUsize = NonZeroUsize::new(7).expect("7 is not zero");

/// The %D period of [`StochRsi`], unused by the pipeline: TA-Lib's
/// default.
const STOCH_RSI_D: NonZeroUsize = NonZeroUsize::new(3).expect("3 is not zero");

/// Updates before [`Kst::new`] compares every rate of change against a
/// full window: its slowest leg, ROC(30) smoothed by SMA(15).
const KST_WARM_UP: usize = 30 + 15;

/// `Err` unless the fast window is below the slow one, which the
/// constructors of the two-EMA oscillators assert.
fn check_fast_slow(fast: NonZeroUsize, slow: NonZeroUsize) -> Result<(), String> {
    if fast >= slow {
        return Err(format!(
            "the fast window {fast} must be below the slow window {slow}"
        ));
    }
    Ok(())
}

/// The power of [`Jma`]'s kernel, its documented default.
const JMA_POWER: i32 = 2;

/// The largest phase magnitude of [`Jma`]: the view clamps beyond it, so a
/// larger value would silently mean the same as 100.
const MAX_JMA_PHASE: f64 = 100.0;

/// `Err` if a [`Jma`] phase lies outside `[-100, 100]`.
fn check_jma_phase(phase: f64) -> Result<(), String> {
    if phase.abs() > MAX_JMA_PHASE {
        return Err(format!(
            "the phase {phase} must be in [-{MAX_JMA_PHASE}, {MAX_JMA_PHASE}]"
        ));
    }
    Ok(())
}

/// Updates before [`Hma`] and [`Ehma`] are ready: the full window, then
/// the final stage over `round(sqrt(window))` values, at most
/// `isqrt(window) + 1`.
fn hull_warm_up(window: NonZeroUsize) -> usize {
    window.get() + window.get().isqrt()
}

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

/// The embedding delay of [`PermutationEntropy`]: consecutive values, as
/// Bandt and Pompe.
const PERMUTATION_DELAY: NonZeroUsize = NonZeroUsize::MIN;

/// `Err` unless `[lo, lo + width]` has finite ends that rounding keeps
/// apart, which [`ShannonEntropy::new`] asserts (a `width` far below the
/// spacing of floats at `lo` would vanish).
fn check_bin_range(lo: f64, width: f64) -> Result<(), String> {
    let hi = lo + width;
    if !hi.is_finite() || !(hi - lo).is_finite() || hi <= lo {
        return Err(format!(
            "the range [{lo}, {lo} + {width}] must have finite, distinct ends"
        ));
    }
    Ok(())
}

/// The largest sample entropy of a `window` with templates of length `m`:
/// `ln` of the number of template pairs, as at least one pair matches.
fn sample_entropy_max(window: NonZeroUsize, m: usize) -> f64 {
    let templates = window_f64(window) - f64::from(u8::try_from(m).expect("checked integer"));
    (templates * (templates - 1.0) / 2.0).ln()
}

/// The longest window of views that cost O(window log window) per update:
/// 1000 keeps an update of [`HurstExponent`] (two passes over the window
/// per chunk size 8, 16, ..., 512) at about 10^4 operations, like
/// [`MAX_QUADRATIC_WINDOW`].
const MAX_LOG_LINEAR_WINDOW: usize = 1000;

/// `Err` if the `window` of a view that costs O(window log window) per
/// update exceeds [`MAX_LOG_LINEAR_WINDOW`].
fn check_log_linear_window(window: NonZeroUsize) -> Result<(), String> {
    if window.get() > MAX_LOG_LINEAR_WINDOW {
        return Err(format!(
            "the window {window} must be at most {MAX_LOG_LINEAR_WINDOW}, the cost per update grows with window log window"
        ));
    }
    Ok(())
}

/// `Err` unless a horizon (a lag, an aggregation period) is below the
/// window that holds it, which the constructors assert.
fn check_horizon(what: &str, horizon: NonZeroUsize, window: NonZeroUsize) -> Result<(), String> {
    if horizon >= window {
        return Err(format!(
            "the {what} {horizon} must be below the window {window}"
        ));
    }
    Ok(())
}

/// A checked integer parameter of at least 1 as a count.
fn count(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("checked integer of at least 1")
}

/// The derivative cutoff of [`OneEuroFilter`], in cycles per update: the
/// reference implementation's default.
const ONE_EURO_D_CUTOFF: f64 = 1.0;

/// `Err` if `value` is negative.
fn check_non_negative(what: &str, value: f64) -> Result<(), String> {
    if value < 0.0 {
        return Err(format!("the {what} {value} must be at least 0"));
    }
    Ok(())
}

/// The stability region [`AlphaBetaFilter::new`] asserts beyond the
/// per-kind checks: `beta <= 2` and `4 - 2 alpha - beta > 0`, computed as
/// the constructor does, since near `alpha = 1, beta = 2` rounding decides.
fn check_alpha_beta(alpha: f64, beta: f64) -> Result<(), String> {
    if beta > 2.0 {
        return Err(format!("the beta {beta} must be at most 2"));
    }
    if 4.0 - 2.0 * alpha - beta <= 0.0 {
        return Err(format!(
            "the gains alpha {alpha} and beta {beta} must satisfy 4 - 2 alpha - beta > 0"
        ));
    }
    Ok(())
}

/// `Err` unless [`Kalman1d`]'s variance sums stay finite: the estimate
/// variance stays at most about `r`, so an update sums at most about
/// `q + 2 r`.
fn check_kalman(q: f64, r: f64) -> Result<(), String> {
    if !(2.0 * (q + r)).is_finite() {
        return Err(format!(
            "the variances q {q} and r {r} must sum to below half the largest float"
        ));
    }
    Ok(())
}

/// The regularizer of [`NlmsFilter`], in squared input units: small
/// against the tap energy of most inputs, yet it bounds a weight step to
/// `mu |error| / (2 sqrt(epsilon))` while the taps are near zero.
const NLMS_EPSILON: f64 = 1e-12;

/// `Err` unless the [`NlmsFilter`] step is below 2, which its constructor
/// asserts for stability.
fn check_nlms_mu(mu: f64) -> Result<(), String> {
    if mu >= 2.0 {
        return Err(format!("the mu {mu} must be below 2"));
    }
    Ok(())
}

/// The initialization of [`RlsFilter`]'s inverse correlation matrix
/// `I / delta`: Haykin's small delta for a weak prior on the weights,
/// which the forgetting factor soon forgets.
const RLS_DELTA: f64 = 0.01;

/// The output range of [`Cusum`] with mean 0 and standard deviation 1 for
/// an argument in `arg`: before an update both statistics are at most `h`
/// (an alarm resets them), and an update adds at most `hi - k` to `S+`
/// and `-lo - k` to `S-`.
fn cusum_range(arg: Domain, k: f64, h: f64) -> Domain {
    let up = (h + arg.hi() - k).max(0.0);
    let down = (h - arg.lo() - k).max(0.0);
    Domain::closed(-down, up)
}

/// The forgetting factor of [`PageHinkley`]'s statistics: scikit-multiflow's
/// `1 - 0.0001`.
const PAGE_HINKLEY_ALPHA: f64 = 0.9999;

/// The [`PageHinkleyDirection`] of a checked `Integer { min: 0, max: 2 }`.
fn page_hinkley_direction(direction: usize) -> PageHinkleyDirection {
    match direction {
        0 => PageHinkleyDirection::Both,
        1 => PageHinkleyDirection::Increase,
        2 => PageHinkleyDirection::Decrease,
        _ => unreachable!("checked integer in [0, 2]"),
    }
}

/// The output range of [`PageHinkley`] for an argument in `arg`: an update
/// adds at most `w = max(0, hi - lo - delta)` to a statistic (the running
/// mean lies in `[lo, hi]`). After `min_instances` values a statistic
/// above `lambda` resets, so it reaches at most `lambda + w`; before (also
/// after each reset) it accumulates to at most `min_instances w`.
fn page_hinkley_range(arg: Domain, p: Params<'_>) -> Domain {
    let (min_instances, delta, lambda) = (window_f64(p.window(0)), p.real(1), p.real(2));
    let step = (arg.hi() - arg.lo() - delta).max(0.0);
    let bound = (min_instances * step).max(lambda + step);
    match page_hinkley_direction(p.integer(3)) {
        PageHinkleyDirection::Both => Domain::closed(-bound, bound),
        PageHinkleyDirection::Increase => Domain::closed(0.0, bound),
        PageHinkleyDirection::Decrease => Domain::closed(-bound, 0.0),
    }
}

/// The longest run length [`Bocpd`] tracks: it costs O(max_run_length) per
/// update with a logarithm and an exponential per run length, so 1000
/// keeps an update near 10^4 operations, like [`MAX_LOG_LINEAR_WINDOW`].
const MAX_RUN_LENGTH: usize = 1000;

/// `Err` if [`Bocpd`]'s `max_run_length` exceeds [`MAX_RUN_LENGTH`].
fn check_max_run_length(max_run_length: NonZeroUsize) -> Result<(), String> {
    if max_run_length.get() > MAX_RUN_LENGTH {
        return Err(format!(
            "the max_run_length {max_run_length} must be at most {MAX_RUN_LENGTH}, the cost per update grows with it"
        ));
    }
    Ok(())
}

/// The largest magnitude of a rate per period of [`SharpeRatio`] and
/// [`SortinoRatio`]: far beyond any rate (even in percent), yet small
/// enough that the excess return over the smallest positive deviation
/// (about 1e-162, whose square is the smallest float) stays finite.
const MAX_RATE: f64 = 1e6;

/// `Err` if a rate per period exceeds [`MAX_RATE`] in magnitude.
fn check_rate(what: &str, rate: f64) -> Result<(), String> {
    if rate.abs() > MAX_RATE {
        return Err(format!(
            "the {what} {rate} must be at most {MAX_RATE} in magnitude, it is a rate per period"
        ));
    }
    Ok(())
}

/// The buckets per level of [`Adwin`]'s exponential histogram: MOA's default.
const ADWIN_MAX_BUCKETS: NonZeroUsize = NonZeroUsize::new(5).expect("5 is not zero");

/// The updates between [`Adwin`]'s cut checks: MOA's default.
const ADWIN_CLOCK: NonZeroUsize = NonZeroUsize::new(32).expect("32 is not zero");

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

/// The Box-Cox transform of `v`, as [`crate::pure_functions::BoxCox`]
/// computes it. Defined at `v = 0` and `v = inf` (as limits) so that it can
/// map domain bounds; the caller admits only positive `v`.
pub(super) fn box_cox(v: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        v.ln()
    } else {
        (v.powf(lambda) - 1.0) / lambda
    }
}

/// The Yeo-Johnson transform of `v`, as [`crate::pure_functions::YeoJohnson`]
/// computes it, also at `v = +-inf` to map domain bounds.
pub(super) fn yeo_johnson(v: f64, lambda: f64) -> f64 {
    if v >= 0.0 {
        box_cox(v + 1.0, lambda)
    } else {
        -box_cox(-v + 1.0, 2.0 - lambda)
    }
}

/// The output range of [`QuantileNormalizer`]: the normal quantiles of the
/// lowest and highest plotting positions `1 / (n + 1)` and `n / (n + 1)`,
/// computed as the view does.
fn gaussian_rank_range(window: NonZeroUsize) -> Domain {
    let n = window_f64(window);
    let (lo, hi) = (1.0 / (n + 1.0), n / (n + 1.0));
    // Beyond about 2^52 values the highest position rounds to 1, where
    // the quantile is infinite; no window that long fits in memory.
    if hi >= 1.0 {
        return Domain::REAL;
    }
    Domain::closed(inverse_normal_cdf(lo), inverse_normal_cdf(hi))
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
        assert!(Function::Jma.check_params(&[9.0, 100.0], 100).is_ok());
        assert!(Function::Jma.check_params(&[9.0, -100.0], 100).is_ok());
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
            (Function::InverseFisher, &[0.0]),
            (Function::InverseFisher, &[-1.0]),
            (Function::InverseFisher, &[f64::INFINITY]),
            (Function::EhlersStochastic, &[2.0, 2.0, 1.0]),
            (Function::UniversalOscillator, &[1.0]),
            (Function::ReversionIndex, &[1.0, 4.0]),
            (Function::Hma, &[1.0]),
            (Function::T3, &[5.0, 1.0]),
            (Function::T3, &[5.0, 0.0]),
            (Function::Jma, &[9.0, 100.5]),
            (Function::Jma, &[9.0, -101.0]),
            (Function::Cmo, &[1.0]),
            (Function::Macd, &[12.0, 12.0]),
            (Function::Macd, &[26.0, 12.0]),
            (Function::Ppo, &[13.0, 12.0]),
            (Function::SchaffTrendCycle, &[50.0, 23.0, 10.0]),
            (Function::Kst, &[10.0]),
            (Function::LinRegSlope, &[1.0]),
            (Function::StandardError, &[2.0]),
            (Function::Vhf, &[1.0]),
            (Function::Quantile, &[1.0, 0.5]),
            (Function::Quantile, &[10.0, 0.0]),
            (Function::Quantile, &[10.0, 1.0]),
            (Function::Median, &[1.0]),
            (Function::Iqr, &[1.0]),
            (Function::Skewness, &[2.0]),
            (Function::Kurtosis, &[3.0]),
            (Function::EwmaVariance, &[1.0]),
            (Function::BoxCox, &[f64::NAN]),
            (Function::YeoJohnson, &[f64::INFINITY]),
            (Function::ShannonEntropy, &[10.0, 1.0, 0.0, 1.0]),
            (Function::ShannonEntropy, &[10.0, 101.0, 0.0, 1.0]),
            (Function::ShannonEntropy, &[10.0, 10.0, 0.0, 0.0]),
            (Function::ShannonEntropy, &[10.0, 10.0, 1e20, 1.0]),
            (Function::ShannonEntropy, &[10.0, 10.0, 1e308, 1e308]),
            (Function::PermutationEntropy, &[10.0, 8.0]),
            (Function::PermutationEntropy, &[10.0, 1.0]),
            (Function::SampleEntropy, &[101.0, 2.0, 0.2]),
            (Function::SampleEntropy, &[19.0, 2.0, 0.2]),
            (Function::SampleEntropy, &[20.0, 6.0, 0.2]),
            (Function::SampleEntropy, &[10.0, 2.0, 0.0]),
            (Function::Hurst, &[31.0]),
            (Function::Hurst, &[1001.0]),
            (Function::VarianceRatio, &[1.0, 10.0]),
            (Function::VarianceRatio, &[10.0, 10.0]),
            (Function::Autocorrelation, &[10.0, 10.0]),
            (Function::OuHalfLife, &[2.0]),
            (Function::OneEuro, &[0.0, 0.5]),
            (Function::OneEuro, &[0.5, -0.1]),
            (Function::AlphaBeta, &[0.5, 2.5]),
            // 4 - 2 alpha rounds to 2 just below alpha = 1.
            (Function::AlphaBeta, &[1.0 - f64::EPSILON / 2.0, 2.0]),
            (Function::Kalman, &[1e308, 1e308]),
            (Function::Kalman, &[0.0, 1.0]),
            (Function::Nlms, &[4.0, 2.0]),
            (Function::Nlms, &[0.0, 0.5]),
            (Function::Nlms, &[65.0, 0.5]),
            (Function::Rls, &[33.0, 0.5]),
            (Function::Rls, &[4.0, 1.0]),
            (Function::HuberEma, &[1.0, 0.5, 1.0]),
            (Function::Hampel, &[2.0, 3.0]),
            (Function::Hampel, &[1001.0, 3.0]),
            (Function::Hampel, &[10.0, 0.0]),
            (Function::Cusum, &[-0.1, 5.0]),
            (Function::Cusum, &[0.5, 0.0]),
            (Function::PageHinkley, &[0.0, 0.0, 1.0, 0.0]),
            (Function::PageHinkley, &[10.0, -0.1, 1.0, 0.0]),
            (Function::PageHinkley, &[10.0, 0.0, 0.0, 0.0]),
            (Function::PageHinkley, &[10.0, 0.0, 1.0, 3.0]),
            (Function::Bocpd, &[1.0, 10.0]),
            (Function::Bocpd, &[10.0, 1001.0]),
            (Function::Adwin, &[1.0]),
            (Function::Var, &[1.0, 0.95]),
            (Function::Var, &[20.0, 1.0]),
            (Function::Cvar, &[20.0, 0.0]),
            (Function::UlcerIndex, &[1.0]),
            (Function::Sharpe, &[1.0, 0.0]),
            (Function::Sharpe, &[20.0, 1e7]),
            (Function::Sortino, &[20.0, -1e7]),
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
        let err = Function::Macd.check_params(&[26.0, 12.0], 100).unwrap_err();
        assert_eq!(
            err,
            "macd(a, window, window): the fast window 26 must be below the slow window 12"
        );
        let err = Function::Jma.check_params(&[9.0, 100.5], 100).unwrap_err();
        assert_eq!(
            err,
            "jma(a, window, real): the phase 100.5 must be in [-100, 100]"
        );
        let err = Function::Mama.check_params(&[0.5, 0.51], 100).unwrap_err();
        assert_eq!(
            err,
            "mama(a, fraction, fraction): the slow limit 0.51 must be at most the fast limit 0.5"
        );
        for (function, params) in [
            (Function::SampleEntropy, &[100.0, 5.0, 1e300][..]),
            (Function::SampleEntropy, &[20.0, 5.0, 0.2]),
            (Function::Hurst, &[1000.0]),
            (Function::ShannonEntropy, &[10.0, 100.0, -1e308, 1e308]),
            (Function::ShannonEntropy, &[10.0, 2.0, 1e20, 1e5]),
        ] {
            function
                .check_params(params, 100_000)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let err = Function::Autocorrelation
            .check_params(&[20.0, 10.0], 100)
            .unwrap_err();
        assert_eq!(
            err,
            "autocorrelation(a, window, window): the lag 20 must be below the window 10"
        );
        let err = Function::SampleEntropy
            .check_params(&[101.0, 2.0, 0.2], 1000)
            .unwrap_err();
        assert_eq!(
            err,
            "sample_entropy(a, window, integer, positive): the window 101 must be at most 100, \
             the cost per update grows with its square"
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
            (Function::UniversalOscillator, &[huge]),
            (Function::UniversalOscillator, &[2.0]),
            (Function::Dema, &[huge]),
            (Function::Tema, &[huge]),
            (Function::T3, &[huge, 0.999_999]),
            (Function::T3, &[1.0, 1e-9]),
            (Function::Ehma, &[huge]),
            (Function::Ehma, &[2.0]),
            (Function::McGinleyDynamic, &[huge]),
            (Function::Jma, &[huge, 100.0]),
            (Function::Jma, &[2.0, -100.0]),
            (Function::Macd, &[1.0, huge]),
            (Function::Macd, &[huge - 1.0, huge]),
            (Function::Ppo, &[1.0, 2.0]),
            (Function::Trix, &[huge]),
            (Function::Tsi, &[huge, huge]),
            (Function::Rsx, &[huge]),
            (Function::Rsx, &[2.0]),
            (Function::EfficiencyRatio, &[1.0]),
            (Function::LinRegSlope, &[2.0]),
            (Function::LinRegIntercept, &[2.0]),
            (Function::LinRegAngle, &[2.0]),
            (Function::RSquared, &[2.0]),
            (Function::Tsf, &[2.0]),
            (Function::StandardError, &[3.0]),
            (Function::Vhf, &[2.0]),
            (Function::EwmaVariance, &[huge]),
            (Function::EwmaVariance, &[2.0]),
            (Function::ShannonEntropy, &[2.0, 100.0, -1e308, 1e308]),
            (Function::ShannonEntropy, &[2.0, 2.0, 1e20, 1e5]),
            (Function::PermutationEntropy, &[2.0, 7.0]),
            (Function::SampleEntropy, &[20.0, 1.0, 1e-300]),
            (Function::SampleEntropy, &[100.0, 5.0, 1e300]),
            (Function::Hurst, &[32.0]),
            (Function::VarianceRatio, &[2.0, 3.0]),
            (Function::Autocorrelation, &[1.0, 2.0]),
            (Function::OuHalfLife, &[3.0]),
            (Function::OneEuro, &[5e-324, 1e308]),
            (Function::OneEuro, &[1e308, 0.0]),
            (Function::AlphaBeta, &[0.999_999, 2.0]),
            (Function::AlphaBeta, &[1e-9, 1e-300]),
            (Function::Kalman, &[4e307, 4e307]),
            (Function::Kalman, &[1e-300, 1e300]),
            (Function::Kalman, &[5e-324, 5e-324]),
            (Function::HuberEma, &[1e-9, 0.999_999, 1e-300]),
            (Function::HuberEma, &[0.999_999, 1e-9, 1e300]),
            (Function::Nlms, &[1.0, 1e-300]),
            (Function::Nlms, &[64.0, 1.999_999]),
            (Function::Rls, &[1.0, 1e-9]),
            (Function::Rls, &[32.0, 0.999_999]),
            (Function::Hampel, &[3.0, 1e-300]),
            (Function::Hampel, &[1000.0, 1e300]),
            (Function::Cusum, &[0.0, 1e-300]),
            (Function::Cusum, &[1e308, 1e308]),
            (Function::PageHinkley, &[huge, 0.0, 5e-324, 0.0]),
            (Function::PageHinkley, &[1.0, 1e308, 1e308, 2.0]),
            (Function::Bocpd, &[2.0, 1000.0]),
            (Function::Bocpd, &[huge, 2.0]),
            (Function::Adwin, &[1e-300]),
            (Function::Adwin, &[0.999_999]),
            (Function::Var, &[2.0, 1e-9]),
            (Function::Cvar, &[2.0, 0.999_999]),
            (Function::UlcerIndex, &[2.0]),
            (Function::Sharpe, &[2.0, 1e6]),
            (Function::Sortino, &[2.0, -1e6]),
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
        assert_eq!(
            Function::InverseFisher.output_domain(real, &[0.5]),
            Some(Domain::closed(-1.0, 1.0))
        );
        assert_eq!(
            Function::InverseFisher.output_domain(Domain::closed(0.0, 2.0), &[0.5]),
            Some(Domain::closed(0.0, 1.0_f64.tanh()))
        );
        let unit = Domain::closed(0.0, 1.0);
        assert_eq!(
            Function::Iqr.output_domain(unit, &[5.0]),
            Some(Domain::closed(0.0, 1.0))
        );
        assert_eq!(
            Function::EwmaVariance.output_domain(unit, &[5.0]),
            Some(Domain::closed(0.0, 0.25))
        );
        assert_eq!(
            Function::EwmaVariance.output_domain(real, &[5.0]),
            Some(Domain::non_negative())
        );
        // The extreme ranks of 3 values sit at 1/4 and 3/4: z = -+0.674.
        let Some(rank) = Function::GaussianRank.output_domain(real, &[3.0]) else {
            panic!("always admitted");
        };
        assert!((rank.lo() + 0.674_489_75).abs() < 1e-8, "{rank}");
        assert!((rank.hi() - 0.674_489_75).abs() < 1e-8, "{rank}");
        // Increasing, from the limits at 0 (and infinity for lambda < 0).
        assert_eq!(
            Function::BoxCox.output_domain(real, &[2.0]),
            Some(Domain::closed(-0.5, f64::INFINITY))
        );
        assert_eq!(
            Function::BoxCox.output_domain(real, &[-1.0]),
            Some(Domain::closed(f64::NEG_INFINITY, 1.0))
        );
        assert_eq!(
            Function::YeoJohnson.output_domain(Domain::closed(-1.0, 3.0), &[0.0]),
            Some(Domain::closed(-1.5, 4.0_f64.ln()))
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

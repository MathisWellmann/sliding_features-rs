//! Turns a validated [`Expr`] into a runtime chain of this crate's views.

use std::num::NonZeroUsize;

use super::{
    BinaryOp,
    Expr,
    Function,
    SpecError,
};
use crate::{
    View,
    pure_functions::{
        Abs,
        Add,
        Constant,
        Echo,
        GTE,
        LTE,
        Ln,
        Multiply,
        Subtract,
        Tanh,
    },
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

/// A view assembled at runtime.
#[deprecated(note = "use `pipeline::Feature`, a single-output pipeline that is a `View`")]
pub type DynView = Box<dyn View<f64> + Send>;

/// Resource bounds of a spec, so a spec from an untrusted author (e.g. an
/// LLM) cannot make a feature arbitrarily slow or large.
#[deprecated(
    note = "use the `pipeline` module: its text form compiles every spec to the same values"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The largest window parameter.
    pub max_window: usize,
    /// The most nodes in the tree.
    pub max_nodes: usize,
    /// The longest root-to-leaf path.
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_window: 100_000,
            max_nodes: 64,
            max_depth: 16,
        }
    }
}

impl Limits {
    /// No bounds besides the signatures: what parsing checks.
    pub fn unbounded() -> Self {
        Self {
            max_window: usize::MAX,
            max_nodes: usize::MAX,
            max_depth: usize::MAX,
        }
    }
}

/// Passes only finite values on: one non-finite value (e.g. an overflow)
/// would otherwise stay in every window downstream for its whole length.
struct Finite<V>(V);

impl<V: View<f64>> View<f64> for Finite<V> {
    fn update(&mut self, val: f64) {
        self.0.update(val);
    }

    fn last(&self) -> Option<f64> {
        self.0.last().filter(|v| v.is_finite())
    }
}

/// Passes only positive values on, for functions defined on them.
struct Positive<V>(V);

impl<V: View<f64>> View<f64> for Positive<V> {
    fn update(&mut self, val: f64) {
        self.0.update(val);
    }

    fn last(&self) -> Option<f64> {
        self.0.last().filter(|v| *v > 0.0)
    }
}

/// `a / b`, not ready while `b` is zero (the library's `Divide` asserts a
/// non-zero divisor, which a spec cannot guarantee).
struct SafeDiv<A, B> {
    a: A,
    b: B,
}

impl<A: View<f64>, B: View<f64>> View<f64> for SafeDiv<A, B> {
    fn update(&mut self, val: f64) {
        self.a.update(val);
        self.b.update(val);
    }

    fn last(&self) -> Option<f64> {
        let (a, b) = (self.a.last()?, self.b.last()?);
        (b != 0.0).then(|| a / b).filter(|v| v.is_finite())
    }
}

impl Expr {
    /// Check the spec against `limits` and assemble its view chain.
    ///
    /// Feed the view finite values only. Its output is always finite: a
    /// division by zero, a logarithm of a non-positive value or an overflow
    /// makes that node not ready (`last()` is `None`) instead of passing a
    /// non-finite value on.
    pub fn build(&self, limits: &Limits) -> Result<DynView, SpecError> {
        self.validate(limits)?;
        Ok(self.assemble())
    }

    fn assemble(&self) -> DynView {
        let view: DynView = match self {
            Expr::Input => Box::new(Echo::new()),
            Expr::Const(c) => Box::new(Constant::new(*c)),
            Expr::Neg(e) => Box::new(Multiply::new(Constant::new(-1.0), e.assemble())),
            Expr::Binary { op, lhs, rhs } => {
                let (a, b) = (lhs.assemble(), rhs.assemble());
                match op {
                    BinaryOp::Add => Box::new(Add::new(a, b)),
                    BinaryOp::Sub => Box::new(Subtract::new(a, b)),
                    BinaryOp::Mul => Box::new(Multiply::new(a, b)),
                    BinaryOp::Div => Box::new(SafeDiv { a, b }),
                }
            }
            Expr::Call {
                function,
                arg,
                params,
            } => call(*function, arg.assemble(), params),
        };
        Box::new(Finite(view))
    }
}

/// The view of one validated call.
fn call(function: Function, a: DynView, params: &[f64]) -> DynView {
    // Validated: positive integers far below 2^53, exact as f64.
    let w = |i: usize| NonZeroUsize::new(params[i] as usize).expect("validated window");
    let ema = |i: usize| Ema::new(Echo::new(), w(i));
    use Function as F;
    match function {
        F::Abs => Box::new(Abs::new(a)),
        F::Ln => Box::new(Ln::new(a)),
        F::Tanh => Box::new(Tanh::new(a)),
        F::LnReturn => Box::new(LnReturn::new(Positive(a))),
        F::Drawdown => Box::new(Drawdown::new(Positive(a))),
        F::Gte => Box::new(GTE::new(a, params[0])),
        F::Lte => Box::new(LTE::new(a, params[0])),
        F::Sma => Box::new(Sma::new(a, w(0))),
        F::Ema => Box::new(Ema::new(a, w(0))),
        F::Alma => Box::new(Alma::new(a, w(0))),
        F::Roc => Box::new(Roc::new(a, w(0))),
        F::Lag => Box::new(Lag::new(a, w(0))),
        F::Min => Box::new(Min::new(a, w(0))),
        F::Max => Box::new(Max::new(a, w(0))),
        F::MinMax => Box::new(MinMaxNormalizer::new(a, w(0))),
        F::Mad => Box::new(MadScaler::new(a, w(0))),
        F::ZScore => Box::new(ZScoreStandardization::new(a, w(0))),
        F::Std => Box::new(WelfordOnline::new(a, w(0))),
        F::Cumulative => Box::new(Cumulative::new(a, w(0))),
        F::Rsi => Box::new(Rsi::new(a, w(0))),
        F::MyRsi => Box::new(MyRSI::new(a, w(0))),
        F::LaguerreRsi => Box::new(LaguerreRSI::new(a, w(0))),
        F::LaguerreFilter => Box::new(LaguerreFilter::new(a, params[0])),
        F::SuperSmoother => Box::new(SuperSmoother::new(a, w(0))),
        F::RoofingFilter => Box::new(RoofingFilter::new(a, w(0), w(1))),
        F::ReFlex => Box::new(ReFlex::new(a, w(0))),
        F::TrendFlex => Box::new(TrendFlex::new(a, w(0))),
        F::CyberCycle => Box::new(CyberCycle::new(a, w(0))),
        F::CenterOfGravity => Box::new(CenterOfGravity::new(a, w(0))),
        F::Cti => Box::new(CorrelationTrendIndicator::new(a, w(0))),
        F::Pfe => Box::new(PolarizedFractalEfficiency::new(a, ema(1), w(0))),
        F::Fisher => Box::new(EhlersFisherTransform::new(a, ema(1), w(0))),
        F::BinaryEntropy => Box::new(BinaryEntropy::new(a, w(0))),
        F::Vst => Box::new(Vst::new(a, w(0))),
        F::Net => Box::new(NoiseEliminationTechnology::new(a, w(0))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Param;

    fn build(spec: &str) -> DynView {
        Expr::parse(spec)
            .expect("parses")
            .build(&Limits::default())
            .expect("builds")
    }

    /// A built spec computes exactly what the hand-written chain computes.
    #[test]
    fn matches_the_hand_written_chain() {
        let window = |n| NonZeroUsize::new(n).expect("non zero");
        let mut built = build("zscore(ema(x, 5) - sma(x, 10), 20)");
        let mut manual = ZScoreStandardization::new(
            Subtract::new(
                Ema::new(Echo::new(), window(5)),
                Sma::new(Echo::new(), window(10)),
            ),
            window(20),
        );
        let rng = romu::Rng::from_seed_with_64bit(1);
        let mut price = 100.0;
        for _ in 0..500 {
            price += rng.f64() - 0.5;
            built.update(price);
            manual.update(price);
            assert_eq!(built.last(), manual.last());
        }
        assert!(built.last().is_some());
    }

    #[test]
    fn division_by_zero_is_not_ready_instead_of_infinite() {
        let mut v = build("x / (x - x)");
        v.update(3.0);
        assert_eq!(v.last(), None);
        let mut v = build("1 / x");
        v.update(4.0);
        assert_eq!(v.last(), Some(0.25));
    }

    #[test]
    fn ln_return_ignores_non_positive_input() {
        let mut v = build("ln_return(x - 1)");
        for x in [2.0, 0.5, 3.0] {
            v.update(x);
        }
        // 0.5 - 1 < 0 never reached the log return.
        assert!((v.last().expect("ready") - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn limits_are_enforced() {
        let e = Expr::parse("sma(x, 500)").expect("parses");
        let tight = Limits {
            max_window: 100,
            ..Limits::default()
        };
        assert!(e.build(&tight).is_err());
        assert!(e.build(&Limits::default()).is_ok());
    }

    /// Inputs that break naive indicators: a trend, a constant, zeros,
    /// alternating signs, huge values.
    fn adversarial_inputs() -> Vec<Vec<f64>> {
        let rng = romu::Rng::from_seed_with_64bit(7);
        let mut walk = 100.0;
        vec![
            (0..300)
                .map(|_| {
                    walk += rng.f64() - 0.5;
                    walk
                })
                .collect(),
            vec![5.0; 300],
            vec![0.0; 300],
            (0..300)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            (0..300).map(|i| 1e12 * f64::from(i % 7)).collect(),
        ]
    }

    /// Every function, at its minimum window, on every adversarial input
    /// and composed with a sign-changing argument, never panics (debug
    /// assertions on) and only ever outputs finite values.
    #[test]
    fn every_function_survives_adversarial_input() {
        let mut failures = Vec::new();
        for function in Function::ALL {
            let params = Vec::from_iter(function.params().iter().map(|p| match p {
                Param::Window => function.min_window().to_string(),
                Param::Real => "0.5".to_string(),
            }));
            let tail = params.iter().map(|p| format!(", {p}")).collect::<String>();
            for arg in ["x", "roc(x, 1)", "x - x"] {
                let spec = format!("{}({arg}{tail})", function.name());
                for input in adversarial_inputs() {
                    let run = std::panic::catch_unwind(|| {
                        let mut v = build(&spec);
                        input.iter().all(|x| {
                            v.update(*x);
                            v.last().is_none_or(f64::is_finite)
                        })
                    });
                    match run {
                        Ok(true) => {}
                        Ok(false) => failures.push(format!("{spec}: non-finite output")),
                        Err(_) => failures.push(format!("{spec}: panicked on {:?}", &input[..3])),
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}

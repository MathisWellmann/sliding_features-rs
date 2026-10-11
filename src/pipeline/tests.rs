//! Behaviour of compiled pipelines: they compute what hand-written view
//! chains compute, survive adversarial input and stay in their domains.

use std::num::NonZeroUsize;

use super::*;
use crate::{
    View,
    pure_functions::{
        Echo,
        Subtract,
    },
    sliding_windows::{
        Ema,
        Sma,
        ZScoreStandardization,
    },
};

fn window(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn single_input() -> (Builder, NodeId) {
    let b = Builder::new(vec![Input::real("x")], Limits::default()).unwrap();
    let x = b.lookup("x").unwrap();
    (b, x)
}

fn random_walk(seed: u64, len: usize) -> Vec<f64> {
    let rng = romu::Rng::from_seed_with_64bit(seed);
    let mut price = 100.0;
    Vec::from_iter((0..len).map(|_| {
        price += rng.f64() - 0.5;
        price
    }))
}

/// Parameters at the smallest valid values.
fn min_params(function: Function) -> Vec<f64> {
    Vec::from_iter(function.params().iter().map(|p| match p {
        Param::Window => f64::from(u32::try_from(function.min_window()).unwrap()),
        Param::Real | Param::Fraction => 0.5,
    }))
}

#[test]
fn matches_the_hand_written_chain() {
    let (mut b, x) = single_input();
    let fast = b.call(Function::Ema, x, &[5.0]).unwrap();
    let slow = b.call(Function::Sma, x, &[10.0]).unwrap();
    let diff = b.binary(BinaryOp::Sub, fast, slow).unwrap();
    let z = b.call(Function::ZScore, diff, &[20.0]).unwrap();
    b.output("z", z).unwrap();
    let mut pipeline = b.build().unwrap();
    let mut manual = ZScoreStandardization::new(
        Subtract::new(
            Ema::new(Echo::new(), window(5)),
            Sma::new(Echo::new(), window(10)),
        ),
        window(20),
    );
    for price in random_walk(1, 500) {
        pipeline.update(&[price]);
        manual.update(price);
        assert_eq!(pipeline.output(0), manual.last());
    }
    assert!(pipeline.output(0).is_some());
}

#[test]
fn shared_nodes_are_computed_once() {
    let (mut b, x) = single_input();
    let r = b.call(Function::LnReturn, x, &[]).unwrap();
    let fast = b.call(Function::Std, r, &[10.0]).unwrap();
    let slow = b.call(Function::Std, r, &[60.0]).unwrap();
    let ratio = b.binary(BinaryOp::Div, fast, slow).unwrap();
    b.output("ratio", ratio).unwrap();
    let again = b.call(Function::Std, r, &[10.0]).unwrap();
    b.output("fast", again).unwrap();
    let pipeline = b.build().unwrap();
    // x, ln_return, two stds and the division.
    assert_eq!(pipeline.op_count(), 5);
    assert_eq!(Vec::from_iter(pipeline.output_names()), ["ratio", "fast"]);
    assert_eq!(pipeline.output_index("fast"), Some(1));
}

#[test]
fn unused_helpers_are_not_run() {
    let (mut b, x) = single_input();
    let helper = b.call(Function::Sma, x, &[10.0]).unwrap();
    b.bind("helper", helper).unwrap();
    let out = b.call(Function::Ema, x, &[10.0]).unwrap();
    b.output("out", out).unwrap();
    assert_eq!(b.build().unwrap().op_count(), 2);
}

#[test]
fn a_view_holds_its_value_while_its_argument_is_not_ready() {
    let (mut b, x) = single_input();
    let one = b.constant(1.0).unwrap();
    let inv = b.binary(BinaryOp::Div, one, x).unwrap();
    let sma = b.call(Function::Sma, inv, &[2.0]).unwrap();
    b.output("inv", inv).unwrap();
    b.output("sma", sma).unwrap();
    let mut p = b.build().unwrap();
    p.update(&[1.0]);
    p.update(&[2.0]);
    assert_eq!(Vec::from_iter(p.outputs()), [Some(0.5), Some(0.75)]);
    p.update(&[0.0]);
    assert_eq!(Vec::from_iter(p.outputs()), [None, Some(0.75)]);
    p.update(&[4.0]);
    assert_eq!(Vec::from_iter(p.outputs()), [Some(0.25), Some(0.375)]);
}

#[test]
fn requirements_guard_the_views() {
    let (mut b, x) = single_input();
    let one = b.constant(1.0).unwrap();
    let shifted = b.binary(BinaryOp::Sub, x, one).unwrap();
    let r = b.call(Function::LnReturn, shifted, &[]).unwrap();
    b.output("r", r).unwrap();
    let mut p = b.build().unwrap();
    for x in [2.0, 0.5, 3.0] {
        p.update(&[x]);
    }
    // 0.5 - 1 < 0 never reached the log return.
    assert!((p.output(0).unwrap() - 2.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn non_finite_input_is_not_ready() {
    let (mut b, x) = single_input();
    let sma = b.call(Function::Sma, x, &[1.0]).unwrap();
    b.output("raw", x).unwrap();
    b.output("sma", sma).unwrap();
    let mut p = b.build().unwrap();
    p.update(&[f64::INFINITY]);
    assert_eq!(Vec::from_iter(p.outputs()), [None, None]);
    p.update(&[2.0]);
    assert_eq!(Vec::from_iter(p.outputs()), [Some(2.0), Some(2.0)]);
}

#[test]
fn several_inputs() {
    let inputs = vec![
        Input::new("price", Domain::positive()),
        Input::new("volume", Domain::non_negative()),
    ];
    let mut b = Builder::new(inputs, Limits::default()).unwrap();
    let (price, volume) = (b.lookup("price").unwrap(), b.lookup("volume").unwrap());
    let turnover = b.binary(BinaryOp::Mul, price, volume).unwrap();
    let num = b.call(Function::Cumulative, turnover, &[2.0]).unwrap();
    let den = b.call(Function::Cumulative, volume, &[2.0]).unwrap();
    let vwap = b.binary(BinaryOp::Div, num, den).unwrap();
    b.output("vwap", vwap).unwrap();
    let mut p = b.build().unwrap();
    p.update(&[10.0, 1.0]);
    p.update(&[20.0, 3.0]);
    assert_eq!(p.output(0), Some(17.5));
    assert_eq!(p.inputs()[1].name(), "volume");
}

#[test]
#[should_panic(expected = "one value per input")]
fn update_checks_the_input_count() {
    let (mut b, x) = single_input();
    b.output("raw", x).unwrap();
    b.build().unwrap().update(&[1.0, 2.0]);
}

#[test]
fn clone_branches_and_reset_starts_over() {
    let (mut b, x) = single_input();
    let rsi = b.call(Function::Rsi, x, &[14.0]).unwrap();
    b.output("rsi", rsi).unwrap();
    let mut p = b.build().unwrap();
    let fresh = p.clone();
    let walk = random_walk(2, 200);
    for v in &walk[..100] {
        p.update(&[*v]);
    }
    let mut branch = p.clone();
    for v in &walk[100..] {
        p.update(&[*v]);
        branch.update(&[*v]);
        assert_eq!(p.output(0), branch.output(0));
    }
    p.reset();
    let mut fresh = fresh;
    for v in &walk {
        p.update(&[*v]);
        fresh.update(&[*v]);
        assert_eq!(p.output(0), fresh.output(0));
    }
}

#[test]
fn a_single_output_pipeline_is_a_view() {
    let (mut b, x) = single_input();
    let ema = b.call(Function::Ema, x, &[3.0]).unwrap();
    b.output("ema", ema).unwrap();
    let feature = Feature::try_from(b.build().unwrap()).unwrap();
    let mut chained = Sma::new(feature, window(2));
    let mut manual = Sma::new(Ema::new(Echo::new(), window(3)), window(2));
    for v in random_walk(3, 50) {
        chained.update(v);
        manual.update(v);
        assert_eq!(chained.last(), manual.last());
    }

    let (mut b, x) = single_input();
    b.output("a", x).unwrap();
    b.output("b", x).unwrap();
    let err = Feature::try_from(b.build().unwrap()).unwrap_err();
    assert!(err.to_string().contains("has 1 and 2"), "{err}");
}

/// Inputs that break naive indicators: a trend, a constant, zeros,
/// alternating signs, huge values.
fn adversarial_inputs() -> Vec<Vec<f64>> {
    vec![
        random_walk(7, 300),
        vec![5.0; 300],
        vec![0.0; 300],
        Vec::from_iter((0..300).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })),
        Vec::from_iter((0..300).map(|i| 1e12 * f64::from(i % 7))),
    ]
}

/// The arguments every function is tried on: the input, and series of
/// both signs that are exactly zero on constant input. (`x - x` would be
/// rejected as constant.)
fn arguments(b: &mut Builder, x: NodeId) -> Vec<NodeId> {
    let roc = b.call(Function::Roc, x, &[1.0]).unwrap();
    let lag = b.call(Function::Lag, x, &[1.0]).unwrap();
    let diff = b.binary(BinaryOp::Sub, x, lag).unwrap();
    vec![x, roc, diff]
}

/// Every function, at its minimum parameters, on every adversarial input
/// and composed with arguments of both signs, never panics (debug
/// assertions on) and only ever outputs finite values.
#[test]
fn every_function_survives_adversarial_input() {
    let mut failures = Vec::new();
    for function in Function::ALL {
        for input in adversarial_inputs() {
            let run = std::panic::catch_unwind(|| {
                let (mut b, x) = single_input();
                for (i, arg) in arguments(&mut b, x).into_iter().enumerate() {
                    let call = b.call(*function, arg, &min_params(*function)).unwrap();
                    b.output(&format!("out{i}"), call).unwrap();
                }
                let mut p = b.build().unwrap();
                input.iter().all(|v| {
                    p.update(&[*v]);
                    p.outputs().all(|o| o.is_none_or(f64::is_finite))
                })
            });
            match run {
                Ok(true) => {}
                Ok(false) => failures.push(format!("{}: non-finite output", function.name())),
                Err(_) => failures.push(format!(
                    "{}: panicked on {:?}",
                    function.name(),
                    &input[..3]
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every ready output lies in the domain the function declares for an
/// argument in the hull of the input, up to rounding.
#[test]
fn outputs_stay_in_their_declared_domain() {
    let mut failures = Vec::new();
    for function in Function::ALL {
        let params = min_params(*function);
        for input in adversarial_inputs() {
            let lo = input.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut b, x) = single_input();
            let call = b.call(*function, x, &params).unwrap();
            b.output("out", call).unwrap();
            let mut p = b.build().unwrap();
            let domain = function.output_domain(Domain::closed(lo, hi), &params);
            for v in &input {
                p.update(&[*v]);
                let Some(out) = p.output(0) else { continue };
                let Some(domain) = domain else {
                    // A drawdown that never updated reports 0.
                    if *function != Function::Drawdown {
                        failures.push(format!("{}: ready but never admitted", function.name()));
                    }
                    break;
                };
                // Indicators round, e.g. a correlation of 1 + 1e-8.
                let tolerance = 1e-6 * out.abs().max(1.0);
                let inside = domain.contains(out)
                    || (out >= domain.lo() - tolerance && out <= domain.hi() + tolerance);
                if !inside {
                    failures.push(format!(
                        "{}: {out} outside {domain} on [{lo}, {hi}]",
                        function.name()
                    ));
                    break;
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// After its estimated warm-up on a positive random walk, every function
/// is ready, at its minimum parameters and at larger ones.
#[test]
fn the_warm_up_estimate_suffices() {
    let mut failures = Vec::new();
    for function in Function::ALL {
        let larger = Vec::from_iter(function.params().iter().map(|p| match p {
            Param::Window => 10.0,
            Param::Real | Param::Fraction => 0.5,
        }));
        for params in [min_params(*function), larger] {
            let (mut b, x) = single_input();
            let inner = b.call(Function::Ema, x, &[3.0]).unwrap();
            let call = b.call(*function, inner, &params).unwrap();
            b.output("out", call).unwrap();
            let mut p = b.build().unwrap();
            let walk = random_walk(4, p.warm_up());
            for v in &walk {
                p.update(&[*v]);
            }
            if p.output(0).is_none() {
                failures.push(format!(
                    "{}{params:?}: not ready after {}",
                    function.name(),
                    p.warm_up()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

fn build_errors(b: Builder) -> Vec<String> {
    let err = b.build().unwrap_err();
    Vec::from_iter(err.diagnostics().iter().map(ToString::to_string))
}

#[test]
fn never_ready_calls_and_constant_outputs_are_errors() {
    let (mut b, x) = single_input();
    let abs = b.call(Function::Abs, x, &[]).unwrap();
    let neg = b.neg(abs).unwrap();
    let ln = b.call(Function::Ln, neg, &[]).unwrap();
    b.output("ln", ln).unwrap();
    let zero = b.binary(BinaryOp::Sub, x, x).unwrap();
    let div = b.binary(BinaryOp::Div, x, zero).unwrap();
    b.output("div", div).unwrap();
    b.output("zero", zero).unwrap();
    let sma = b.call(Function::Sma, zero, &[5.0]).unwrap();
    b.output("flat", sma).unwrap();
    assert_eq!(
        build_errors(b),
        [
            "error: `ln(-abs(x))` is never ready: `ln` needs positive values, but `-abs(x)` lies in (-inf, 0]",
            "error: `x / zero` divides by zero, so it is never ready",
            "error: output `zero` is constant: it is always 0",
            "error: output `flat` is constant: it is always 0",
        ]
    );
}

#[test]
fn suspicious_pipelines_compile_with_warnings() {
    let (mut b, x) = single_input();
    let ln = b.call(Function::Ln, x, &[]).unwrap();
    b.output("ln", ln).unwrap();
    let abs = b.call(Function::Abs, x, &[]).unwrap();
    let entropy = b.call(Function::BinaryEntropy, abs, &[10.0]).unwrap();
    b.output("entropy", entropy).unwrap();
    b.output("again", ln).unwrap();
    let unused = b.call(Function::Ema, x, &[3.0]).unwrap();
    b.bind("unused", unused).unwrap();
    let p = b.build().unwrap();
    let warnings = Vec::from_iter(p.warnings().iter().map(ToString::to_string));
    assert_eq!(
        warnings,
        [
            "warning: `ln(x)` is not ready while `x` is not positive (it lies in (-inf, inf)); \
             declare the domain of input `x` if it is always positive",
            "warning: `binary_entropy(abs(x), 10)` is constant: `abs(x)` never changes sign (it lies in [0, inf))",
            "warning: output `again` computes the same as output `ln`",
            "warning: `unused` is not used by any output",
        ]
    );
}

#[test]
fn declared_input_domains_silence_warnings() {
    let inputs = vec![Input::new("price", Domain::positive())];
    let mut b = Builder::new(inputs, Limits::default()).unwrap();
    let price = b.lookup("price").unwrap();
    let ln = b.call(Function::Ln, price, &[]).unwrap();
    let r = b.call(Function::LnReturn, price, &[]).unwrap();
    b.output("ln", ln).unwrap();
    b.output("r", r).unwrap();
    let p = b.build().unwrap();
    assert!(p.warnings().is_empty(), "{:?}", p.warnings());
}

#[test]
fn output_domains_and_warm_ups() {
    let (mut b, x) = single_input();
    let rsi = b.call(Function::Rsi, x, &[14.0]).unwrap();
    let smooth = b.call(Function::Ema, rsi, &[5.0]).unwrap();
    let tanh = b.call(Function::Tanh, x, &[]).unwrap();
    let lag = b.call(Function::Lag, x, &[100.0]).unwrap();
    let sum = b.binary(BinaryOp::Add, smooth, lag).unwrap();
    b.output("smooth", smooth).unwrap();
    b.output("tanh", tanh).unwrap();
    b.output("sum", sum).unwrap();
    let p = b.build().unwrap();
    assert_eq!(p.output_domain(0), Domain::closed(0.0, 100.0));
    assert_eq!(p.output_domain(1), Domain::closed(-1.0, 1.0));
    assert_eq!(p.output_domain(2), Domain::REAL);
    assert_eq!(p.output_warm_up(0), 19);
    assert_eq!(p.output_warm_up(1), 0);
    assert_eq!(p.output_warm_up(2), 100);
    assert_eq!(p.warm_up(), 100);
}

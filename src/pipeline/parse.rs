// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Recursive-descent parser of the text form, driving a [`Builder`].
//!
//! ```text
//! program   := (statement (sep statement)*)?
//! statement := 'let' name '=' expr | name '=' expr | expr
//! expr      := term (('+' | '-') term)*
//! term      := unary (('*' | '/') unary)*
//! unary     := '-' unary | atom
//! atom      := number | name | name '(' expr (',' param)* ')' | '(' expr ')'
//! param     := '-'? number
//! ```

use super::{
    BinaryOp,
    Builder,
    CompileError,
    Function,
    Location,
    graph::NodeId,
    lex::{
        Spanned,
        Token,
        lex,
    },
};

/// Parse `source` into `builder`, statement by statement.
pub(super) fn parse(source: &str, builder: Builder) -> Result<Builder, CompileError> {
    let max_depth = builder.limits().max_depth;
    let mut parser = Parser {
        tokens: lex(source)?,
        pos: 0,
        builder,
        depth: 0,
        max_depth,
    };
    parser.program()?;
    assert_eq!(parser.depth, 0, "nesting is balanced");
    Ok(parser.builder)
}

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    builder: Builder,
    /// The current nesting of parentheses, calls and negations: bounded
    /// so a malicious text cannot overflow the stack.
    depth: usize,
    max_depth: usize,
}

impl Parser {
    fn peek(&self) -> &Spanned {
        &self.tokens[self.pos]
    }

    fn peek_second(&self) -> &Token {
        &self.tokens[(self.pos + 1).min(self.tokens.len() - 1)].0
    }

    fn next(&mut self) -> Spanned {
        let spanned = self.tokens[self.pos].clone();
        if spanned.0 != Token::End {
            self.pos += 1;
        }
        spanned
    }

    fn expect(&mut self, expected: &Token, what: &str) -> Result<(), CompileError> {
        let (token, at) = self.next();
        if &token == expected {
            return Ok(());
        }
        Err(CompileError::at(
            at,
            format!("expected {what}, found {}", token.describe()),
        ))
    }

    fn program(&mut self) -> Result<(), CompileError> {
        loop {
            while self.peek().0 == Token::Sep {
                self.next();
            }
            if self.peek().0 == Token::End {
                return Ok(());
            }
            self.statement()?;
            let (token, at) = self.next();
            match token {
                Token::Sep => {}
                Token::End => return Ok(()),
                other => {
                    return Err(CompileError::at(
                        at,
                        format!(
                            "expected a new line or `;` after the statement, found {}",
                            other.describe()
                        ),
                    ));
                }
            }
        }
    }

    fn statement(&mut self) -> Result<(), CompileError> {
        let (first, at) = self.peek().clone();
        match (first, self.peek_second()) {
            (Token::Name(keyword), Token::Name(_)) if keyword == "let" => {
                self.next();
                let (Token::Name(name), name_at) = self.next() else {
                    unreachable!("peeked a name");
                };
                self.expect(&Token::Assign, &format!("`=` after `let {name}`"))?;
                let node = self.expr()?;
                self.builder.set_location(name_at);
                self.builder.bind(&name, node)
            }
            (Token::Name(name), Token::Assign) => {
                self.next();
                self.next();
                let node = self.expr()?;
                self.builder.set_location(at);
                self.builder.output(&name, node)
            }
            _ => {
                let node = self.expr()?;
                self.builder.set_location(at);
                self.builder.anonymous_output(node)
            }
        }
    }

    fn expr(&mut self) -> Result<NodeId, CompileError> {
        let mut lhs = self.term()?;
        while let (Token::Op(c @ ('+' | '-')), at) = self.peek().clone() {
            self.next();
            let rhs = self.term()?;
            let op = if c == '+' {
                BinaryOp::Add
            } else {
                BinaryOp::Sub
            };
            self.builder.set_location(at);
            lhs = self.builder.binary(op, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<NodeId, CompileError> {
        let mut lhs = self.unary()?;
        while let (Token::Op(c @ ('*' | '/')), at) = self.peek().clone() {
            self.next();
            let rhs = self.unary()?;
            let op = if c == '*' {
                BinaryOp::Mul
            } else {
                BinaryOp::Div
            };
            self.builder.set_location(at);
            lhs = self.builder.binary(op, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<NodeId, CompileError> {
        let (token, at) = self.peek().clone();
        if token != Token::Op('-') {
            return self.atom();
        }
        self.next();
        let inner = self.nested(at, Self::unary)?;
        self.builder.set_location(at);
        // A negated literal folds into a negative constant.
        self.builder.neg(inner)
    }

    fn atom(&mut self) -> Result<NodeId, CompileError> {
        let (token, at) = self.next();
        self.builder.set_location(at);
        match token {
            Token::Number(value) => self.builder.constant(value),
            Token::Name(name) if self.peek().0 == Token::Open => {
                self.nested(at, |p| p.call(&name, at))
            }
            Token::Name(name) => self
                .builder
                .lookup(&name)
                .ok_or_else(|| self.unknown_name(&name, at)),
            Token::Open => {
                let inner = self.nested(at, Self::expr)?;
                self.expect(&Token::Close, "`)`")?;
                Ok(inner)
            }
            other => Err(CompileError::at(
                at,
                format!("expected an expression, found {}", other.describe()),
            )),
        }
    }

    fn call(&mut self, name: &str, at: Location) -> Result<NodeId, CompileError> {
        let function = Function::from_name(name).ok_or_else(|| self.unknown_function(name, at))?;
        self.expect(&Token::Open, &format!("`(` after `{name}`"))?;
        let arg = self.expr()?;
        let mut params = Vec::new();
        while self.peek().0 == Token::Comma {
            self.next();
            params.push(self.param(name)?);
        }
        self.expect(&Token::Close, "`)` or `,`")?;
        self.builder.set_location(at);
        self.builder.call(function, arg, &params)
    }

    fn param(&mut self, function: &str) -> Result<f64, CompileError> {
        let (token, at) = self.next();
        let negative = token == Token::Op('-');
        let token = if negative { self.next().0 } else { token };
        match token {
            Token::Number(value) if negative => Ok(-value),
            Token::Number(value) => Ok(value),
            other => Err(CompileError::at(
                at,
                format!(
                    "the parameters of `{function}` after its argument must be numbers, found {}",
                    other.describe()
                ),
            )),
        }
    }

    /// Run `f` one level deeper, failing past the depth limit.
    fn nested<T>(
        &mut self,
        at: Location,
        f: impl FnOnce(&mut Self) -> Result<T, CompileError>,
    ) -> Result<T, CompileError> {
        if self.depth >= self.max_depth {
            return Err(CompileError::at(
                at,
                format!(
                    "the expression is nested deeper than the limit of {}",
                    self.max_depth
                ),
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn unknown_name(&self, name: &str, at: Location) -> CompileError {
        let message = if let Some(function) = Function::from_name(name) {
            format!(
                "`{name}` is a function, call it as `{}`",
                function.signature()
            )
        } else {
            let known = Vec::from_iter(self.builder.names());
            let hint = match closest(name, known.iter().copied()) {
                Some(close) => format!("did you mean `{close}`?"),
                None => format!("defined so far: {}", known.join(", ")),
            };
            format!("unknown name `{name}`; {hint}")
        };
        CompileError::at(at, message)
    }

    fn unknown_function(&self, name: &str, at: Location) -> CompileError {
        let hint = if self.builder.lookup(name).is_some() {
            format!("`{name}` is a value, not a function")
        } else if let Some(close) = closest(name, Function::ALL.iter().map(|f| f.name())) {
            format!("did you mean `{close}`?")
        } else {
            "see the function reference".to_string()
        };
        CompileError::at(at, format!("unknown function `{name}`; {hint}"))
    }
}

/// The candidate closest to `name` by edit distance, if close enough to be
/// a likely typo: a third of the name may change, but not all of it (`y`
/// is no typo of `x`).
fn closest<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let len = name.chars().count();
    let limit = (len / 3).max(1);
    candidates
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, _)| *d <= limit && *d < len)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// The Levenshtein distance between two strings, in characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = diagonal + usize::from(ca != *cb);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(row[j + 1] + 1);
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::{
        super::{
            Domain,
            Input,
            Limits,
            Param,
            Pipeline,
        },
        *,
    };

    /// Compile over `x`, check that the canonical form compiles to the
    /// same canonical form, and return it.
    fn canonical(source: &str) -> String {
        let pipeline: Pipeline = source.parse().unwrap_or_else(|e| panic!("{source}: {e}"));
        let printed = pipeline.to_string();
        let again: Pipeline = printed.parse().unwrap_or_else(|e| panic!("{printed}: {e}"));
        assert_eq!(again.to_string(), printed, "{source}");
        printed
    }

    fn error(source: &str) -> String {
        source.parse::<Pipeline>().expect_err(source).to_string()
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(canonical("x + x * 2"), "x + (x * 2)");
        assert_eq!(canonical("(x + x) * 2"), "(x + x) * 2");
        assert_eq!(canonical("x - 1 - x"), "(x - 1) - x");
        assert_eq!(canonical("x / 2 / 4"), "(x / 2) / 4");
        assert_eq!(canonical("-x * 2 + -(x - 1)"), "(-x * 2) + -(x - 1)");
    }

    #[test]
    fn calls_unary_minus_and_numbers() {
        assert_eq!(
            canonical("zscore(ema(x, 10) - ema(x,50), 200)"),
            "zscore(ema(x, 10) - ema(x, 50), 200)"
        );
        assert_eq!(canonical("x * -0.5"), "x * -0.5");
        assert_eq!(canonical("x * -(2 * 3)"), "x * -6", "constants fold");
        assert_eq!(canonical("gte(x, -1.5e-3)"), "gte(x, -0.0015)");
        assert_eq!(canonical("--x"), "--x");
        assert_eq!(canonical("-(x + 1)"), "-(x + 1)");
        assert_eq!(canonical("pfe(roc(x, 1), 20, 5)"), "pfe(roc(x, 1), 20, 5)");
        assert_eq!(
            canonical("  laguerre_filter( x ,0.8 ) "),
            "laguerre_filter(x, 0.8)"
        );
    }

    #[test]
    fn every_function_round_trips() {
        for function in Function::ALL {
            let mut source = format!("{}(x", function.name());
            for value in function.example_params() {
                write!(source, ", {value}").unwrap();
            }
            source.push(')');
            assert_eq!(canonical(&source), source);
        }
    }

    #[test]
    fn statements_bindings_and_sharing() {
        let source = "
            # helpers are named in the canonical form
            let r = ln_return(x)
            a = std(r, 10) / std(r, 20); b = std(ln_return(x), 10)
            ema(x, 3) -
                x
            rsi = rsi(x, 14)
            c = ema(rsi, 2)
        ";
        assert_eq!(
            canonical(source),
            "let r = ln_return(x)\n\
             a = std(r, 10) / std(r, 20)\n\
             b = std(r, 10)\n\
             ema(x, 3) - x\n\
             rsi = rsi(x, 14)\n\
             c = ema(rsi, 2)"
        );
        let p: Pipeline = source.parse().unwrap();
        assert_eq!(
            Vec::from_iter(p.output_names()),
            ["a", "b", "ema(x, 3) - x", "rsi", "c"]
        );
        // x, r, two stds, a, ema, the difference, rsi and its ema.
        assert_eq!(p.op_count(), 9);
    }

    #[test]
    fn several_inputs() {
        let inputs = vec![
            Input::new("price", Domain::positive()),
            Input::new("volume", Domain::non_negative()),
        ];
        let source = "vwap = cumulative(price * volume, 2) / cumulative(volume, 2)";
        let mut p = Pipeline::compile(source, inputs, Limits::default()).unwrap();
        p.update(&[10.0, 1.0]);
        p.update(&[20.0, 3.0]);
        assert_eq!(p.output(0), Some(17.5));
        assert_eq!(
            p.to_string(),
            "vwap = cumulative(price * volume, 2) / cumulative(volume, 2)"
        );
    }

    #[test]
    fn errors_point_at_the_problem() {
        for (source, expected) in [
            (
                "sma(x)",
                "error at 1:1: sma(a, window) takes 1 parameter(s) after its argument, got 0",
            ),
            (
                "smaa(x, 3)",
                "error at 1:1: unknown function `smaa`; did you mean `sma`?",
            ),
            (
                "sma(x, y)",
                "error at 1:8: the parameters of `sma` after its argument must be numbers, found `y`",
            ),
            (
                "sma(x, sma(x, 2))",
                "error at 1:8: the parameters of `sma` after its argument must be numbers, found `sma`",
            ),
            ("x +", "error at 1:4: expected an expression, found the end"),
            ("(x", "error at 1:3: expected `)`, found the end"),
            (
                "x x",
                "error at 1:3: expected a new line or `;` after the statement, found `x`",
            ),
            ("x $ 2", "error at 1:3: unexpected `$`"),
            (
                "sma(x, 2.5)",
                "error at 1:1: sma(a, window): the window parameter 2.5 must be an integer in [1, 100000]",
            ),
            ("1..2", "error at 1:1: invalid number `1..2`"),
            (
                "a = x\nb = y",
                "error at 2:5: unknown name `y`; defined so far: x, a",
            ),
            (
                "a = x\nb = aa",
                "error at 2:5: unknown name `aa`; did you mean `a`?",
            ),
            (
                "a = sma",
                "error at 1:5: `sma` is a function, call it as `sma(a, window)`",
            ),
            (
                "let a = x\na(x)",
                "error at 2:1: unknown function `a`; `a` is a value, not a function",
            ),
            (
                "let a = x\na = x",
                "error at 2:1: the name `a` is already defined",
            ),
            ("let = 3", "error at 1:1: the name `let` is a keyword"),
            (
                "ema(x, 3); ema(x, 3)",
                "error at 1:12: the output `ema(x, 3)` is already defined",
            ),
            ("a = 1 / (2 - 2)", "error at 1:7: `1 / 0` is not finite"),
            ("", "error: a pipeline needs at least one output"),
        ] {
            assert_eq!(error(source), expected, "{source}");
        }
    }

    #[test]
    fn analysis_findings_point_at_their_expression() {
        assert_eq!(
            error("a = x\nb = ln(-abs(x) - 1)\nx - x"),
            "error at 2:5: `ln(-abs(x) - 1)` is never ready: `ln` needs positive values, \
             but `-abs(x) - 1` lies in (-inf, -1]\n\
             error at 3:1: output `x - x` is constant: it is always 0"
        );
        let p: Pipeline = "a = ema(x, 2)\nb = 1 + ln(x)".parse().unwrap();
        let warning = &p.warnings()[0];
        assert_eq!(
            warning.location().map(|l| (l.line(), l.column())),
            Some((2, 9))
        );
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_stack_overflow() {
        for source in [
            format!("{}x{}", "(".repeat(100_000), ")".repeat(100_000)),
            format!("{}x", "-".repeat(100_000)),
            format!("{}x{}", "abs(".repeat(100_000), ")".repeat(100_000)),
        ] {
            let err = error(&source);
            assert!(err.contains("nested deeper than the limit of 32"), "{err}");
        }
    }

    /// A random expression over the builder's inputs, `None` if the
    /// builder rejects it (e.g. a constant folds to infinity).
    fn random_expr(b: &mut super::Builder, rng: &romu::Rng, depth: usize) -> Option<NodeId> {
        let leaf = depth == 0 || rng.range_usize(0..4) == 0;
        if leaf {
            return match rng.range_usize(0..3) {
                0 => b.lookup("y"),
                1 => b.constant(f64::from(rng.range_u8(1..=40)) / 8.0 - 2.5).ok(),
                _ => b.lookup("x"),
            };
        }
        let a = random_expr(b, rng, depth - 1)?;
        match rng.range_usize(0..5) {
            0 => b.neg(a).ok(),
            1 | 2 => {
                let ops = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];
                let rhs = random_expr(b, rng, depth - 1)?;
                b.binary(ops[rng.range_usize(0..ops.len())], a, rhs).ok()
            }
            _ => {
                let function = Function::ALL[rng.range_usize(0..Function::ALL.len())];
                // Random, but often valid: windows above the example ones
                // (which pass the checks between parameters).
                let example = function.example_params();
                let params = Vec::from_iter(function.params().iter().zip(example).map(
                    |(p, example)| match *p {
                        Param::Window => example + f64::from(rng.range_u8(0..=9)),
                        Param::Real => f64::from(rng.range_u8(0..=8)) / 4.0 - 1.0,
                        Param::Fraction => 0.5,
                        Param::Positive => f64::from(rng.range_u8(1..=8)) / 4.0,
                        Param::Integer { min, max } => f64::from(rng.range_u8(min..=max)),
                    },
                ));
                b.call(function, a, &params).ok()
            }
        }
    }

    /// Random pipelines print to text that compiles to the same text and
    /// computes the same values.
    #[test]
    fn random_pipelines_round_trip_through_text() {
        let rng = romu::Rng::from_seed_with_64bit(11);
        let inputs = || vec![Input::new("x", Domain::positive()), Input::real("y")];
        let mut compared = 0;
        for _ in 0..500 {
            let mut b = super::Builder::new(inputs(), Limits::default()).unwrap();
            let Some(helper) = random_expr(&mut b, &rng, 3) else {
                continue;
            };
            if b.bind("h", helper).is_err() {
                continue;
            }
            let Some(out) = random_expr(&mut b, &rng, 4) else {
                continue;
            };
            let both = b
                .lookup("h")
                .and_then(|h| b.binary(BinaryOp::Add, h, out).ok());
            let Some(both) = both else { continue };
            if b.output("out", out).is_err() || b.output("both", both).is_err() {
                continue;
            }
            let Ok(mut original) = b.build() else {
                continue;
            };
            let text = original.to_string();
            let mut parsed = Pipeline::compile(&text, inputs(), Limits::default())
                .unwrap_or_else(|e| panic!("{text}\n{e}"));
            assert_eq!(parsed.to_string(), text);
            assert_eq!(parsed.op_count(), original.op_count(), "{text}");
            for i in 0..50 {
                let v = [100.0 + f64::from(i % 7), f64::from(i % 5) - 2.0];
                original.update(&v);
                parsed.update(&v);
                let same = original.outputs().zip(parsed.outputs()).all(|(a, b)| {
                    a == b || a.zip(b).is_some_and(|(a, b)| a.is_nan() && b.is_nan())
                });
                assert!(same, "{text}");
            }
            compared += 1;
        }
        assert!(compared > 100, "only {compared} pipelines compiled");
    }

    #[test]
    fn edit_distances() {
        assert_eq!(edit_distance("sma", "sma"), 0);
        assert_eq!(edit_distance("smaa", "sma"), 1);
        assert_eq!(edit_distance("zscroe", "zscore"), 2);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(
            closest("emaa", ["ema", "sma", "rsi"].into_iter()),
            Some("ema")
        );
        assert_eq!(closest("foo", ["ema", "sma"].into_iter()), None);
    }
}

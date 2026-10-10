//! Recursive-descent parser of the spec language.
//!
//! ```text
//! expr  := term (('+' | '-') term)*
//! term  := unary (('*' | '/') unary)*
//! unary := '-' unary | atom
//! atom  := number | 'x' | name '(' expr (',' number)* ')' | '(' expr ')'
//! ```

use super::{
    BinaryOp,
    Expr,
    Function,
    SpecError,
};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Open,
    Close,
    Comma,
    Op(char),
    End,
}

/// A token and the byte offset it starts at.
type Spanned = (Token, usize);

fn lex(s: &str) -> Result<Vec<Spanned>, SpecError> {
    let bytes = s.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'(' | b')' | b',' | b'+' | b'-' | b'*' | b'/' => {
                tokens.push((
                    match c {
                        b'(' => Token::Open,
                        b')' => Token::Close,
                        b',' => Token::Comma,
                        _ => Token::Op(char::from(c)),
                    },
                    start,
                ));
                i += 1;
            }
            b'0'..=b'9' | b'.' => {
                i = number_end(bytes, i);
                let text = &s[start..i];
                let value = text
                    .parse::<f64>()
                    .map_err(|_| SpecError::parse(start, format!("invalid number `{text}`")))?;
                tokens.push((Token::Number(value), start));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                tokens.push((Token::Name(s[start..i].to_string()), start));
            }
            _ => {
                let ch = s[start..].chars().next().expect("in bounds");
                return Err(SpecError::parse(start, format!("unexpected `{ch}`")));
            }
        }
    }
    tokens.push((Token::End, s.len()));
    Ok(tokens)
}

/// The end of the number starting at `i`: digits, a point, an exponent.
fn number_end(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        if j < bytes.len() && bytes[j].is_ascii_digit() {
            i = j;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    i
}

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Spanned {
        &self.tokens[self.pos]
    }

    fn next(&mut self) -> Spanned {
        let token = self.tokens[self.pos].clone();
        if token.0 != Token::End {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, expected: &Token, what: &str) -> Result<(), SpecError> {
        let (token, at) = self.next();
        if &token == expected {
            Ok(())
        } else {
            Err(SpecError::parse(
                at,
                format!("expected {what}, found {}", describe(&token)),
            ))
        }
    }

    fn expr(&mut self) -> Result<Expr, SpecError> {
        let mut lhs = self.term()?;
        while let Token::Op(c @ ('+' | '-')) = self.peek().0 {
            self.next();
            let op = if c == '+' {
                BinaryOp::Add
            } else {
                BinaryOp::Sub
            };
            let rhs = self.term()?;
            lhs = binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr, SpecError> {
        let mut lhs = self.unary()?;
        while let Token::Op(c @ ('*' | '/')) = self.peek().0 {
            self.next();
            let op = if c == '*' {
                BinaryOp::Mul
            } else {
                BinaryOp::Div
            };
            let rhs = self.unary()?;
            lhs = binary(op, lhs, rhs);
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, SpecError> {
        if self.peek().0 == Token::Op('-') {
            self.next();
            // A negated literal is a negative constant, so `-0.5` prints and
            // parses back as the same tree.
            return Ok(match self.unary()? {
                Expr::Const(c) => Expr::Const(-c),
                e => Expr::Neg(Box::new(e)),
            });
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, SpecError> {
        let (token, at) = self.next();
        match token {
            Token::Number(value) => Ok(Expr::Const(value)),
            Token::Name(name) if name == "x" => Ok(Expr::Input),
            Token::Name(name) => self.call(&name, at),
            Token::Open => {
                let inner = self.expr()?;
                self.expect(&Token::Close, "`)`")?;
                Ok(inner)
            }
            other => Err(SpecError::parse(
                at,
                format!("expected an expression, found {}", describe(&other)),
            )),
        }
    }

    fn call(&mut self, name: &str, at: usize) -> Result<Expr, SpecError> {
        let function = Function::from_name(name)
            .ok_or_else(|| SpecError::parse(at, format!("unknown function `{name}`")))?;
        self.expect(&Token::Open, &format!("`(` after `{name}`"))?;
        let arg = self.expr()?;
        let mut params = Vec::new();
        while self.peek().0 == Token::Comma {
            self.next();
            let param_at = self.peek().1;
            match self.unary()? {
                Expr::Const(value) => params.push(value),
                _ => {
                    return Err(SpecError::parse(
                        param_at,
                        format!("parameters of `{name}` must be numbers"),
                    ));
                }
            }
        }
        self.expect(&Token::Close, "`)` or `,`")?;
        super::expr::validate_call(function, &params, &super::Limits::unbounded())
            .map_err(|e| SpecError::parse(at, e.message().to_string()))?;
        Ok(Expr::Call {
            function,
            arg: Box::new(arg),
            params,
        })
    }
}

fn binary(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn describe(token: &Token) -> String {
    match token {
        Token::Number(v) => format!("number `{v}`"),
        Token::Name(n) => format!("`{n}`"),
        Token::Open => "`(`".to_string(),
        Token::Close => "`)`".to_string(),
        Token::Comma => "`,`".to_string(),
        Token::Op(c) => format!("`{c}`"),
        Token::End => "the end".to_string(),
    }
}

/// Parse a whole spec.
pub(super) fn parse(s: &str) -> Result<Expr, SpecError> {
    let mut parser = Parser {
        tokens: lex(s)?,
        pos: 0,
    };
    let expr = parser.expr()?;
    parser.expect(&Token::End, "the end of the spec")?;
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &str) -> Expr {
        let e = parse(s).unwrap_or_else(|err| panic!("{s}: {err}"));
        let printed = e.to_string();
        assert_eq!(
            parse(&printed).expect("canonical form parses"),
            e,
            "{printed}"
        );
        e
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(roundtrip("x + x * 2").to_string(), "(x + (x * 2))");
        assert_eq!(roundtrip("(x + x) * 2").to_string(), "((x + x) * 2)");
        assert_eq!(roundtrip("x - x - 1").to_string(), "((x - x) - 1)");
        assert_eq!(roundtrip("x / 2 / 4").to_string(), "((x / 2) / 4)");
    }

    #[test]
    fn calls_unary_minus_and_numbers() {
        let e = roundtrip("zscore(ema(x, 10) - ema(x,50), 200)");
        assert_eq!(e.to_string(), "zscore((ema(x, 10) - ema(x, 50)), 200)");
        assert_eq!(roundtrip("-0.5"), Expr::Const(-0.5));
        assert_eq!(roundtrip("gte(x, -1.5e-3)").to_string(), "gte(x, -0.0015)");
        assert_eq!(roundtrip("--x").to_string(), "--x");
        assert_eq!(roundtrip("-(x + 1)").to_string(), "-(x + 1)");
        roundtrip("pfe(roc(x, 1), 20, 5)");
        roundtrip("  laguerre_filter( x ,0.8 ) ");
    }

    #[test]
    fn every_function_round_trips() {
        for function in Function::ALL {
            let params = Vec::from_iter(function.params().iter().map(|p| match p {
                super::super::Param::Window => function.min_window().to_string(),
                super::super::Param::Real => "0.5".to_string(),
            }));
            let mut s = format!("{}(x", function.name());
            for p in params {
                s.push_str(", ");
                s.push_str(&p);
            }
            s.push(')');
            roundtrip(&s);
        }
    }

    #[test]
    fn errors_point_at_the_problem() {
        for (spec, position, fragment) in [
            ("sma(x)", 0, "takes 1 parameter"),
            ("smaa(x, 3)", 0, "unknown function"),
            ("sma(x, y)", 7, "`y`"),
            ("sma(x, sma(x, 2))", 7, "must be numbers"),
            ("x +", 3, "expected an expression"),
            ("(x", 2, "expected `)`"),
            ("x x", 2, "end of the spec"),
            ("x # 2", 2, "unexpected `#`"),
            ("sma(x, 2.5)", 0, "must be an integer"),
            ("1..2", 0, "invalid number"),
        ] {
            let err = parse(spec).expect_err(spec);
            assert_eq!(err.position(), Some(position), "{spec}: {err}");
            assert!(err.to_string().contains(fragment), "{spec}: {err}");
        }
    }
}

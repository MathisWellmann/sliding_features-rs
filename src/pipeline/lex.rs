// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Splits the text form of a pipeline into tokens with their locations.

use super::{
    CompileError,
    Location,
};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Token {
    Number(f64),
    Name(String),
    Open,
    Close,
    Comma,
    /// `+ - * /`
    Op(char),
    Assign,
    /// The end of a statement: `;` or a line break.
    Sep,
    End,
}

impl Token {
    /// The token as an error message names it.
    pub(super) fn describe(&self) -> String {
        match self {
            Token::Number(v) => format!("number `{v}`"),
            Token::Name(n) => format!("`{n}`"),
            Token::Open => "`(`".to_string(),
            Token::Close => "`)`".to_string(),
            Token::Comma => "`,`".to_string(),
            Token::Op(c) => format!("`{c}`"),
            Token::Assign => "`=`".to_string(),
            Token::Sep => "the end of the line".to_string(),
            Token::End => "the end".to_string(),
        }
    }

    /// Whether a line break after the token continues the statement, as
    /// after an operator or inside a call.
    fn continues(&self) -> bool {
        matches!(
            self,
            Token::Op(_) | Token::Comma | Token::Assign | Token::Open | Token::Sep
        )
    }
}

/// A token and where it starts.
pub(super) type Spanned = (Token, Location);

/// Tracks the location while walking the source.
struct Cursor<'a> {
    source: &'a str,
    offset: usize,
    line: usize,
    column: usize,
}

impl Cursor<'_> {
    fn location(&self) -> Location {
        Location::new(self.offset, self.line, self.column)
    }

    fn peek(&self) -> Option<char> {
        self.source[self.offset..].chars().next()
    }

    fn bump(&mut self) {
        let c = self.peek().expect("not at the end");
        self.offset += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }

    fn bump_while(&mut self, pred: impl Fn(char) -> bool) -> &str {
        let start = self.offset;
        while self.peek().is_some_and(&pred) {
            self.bump();
        }
        &self.source[start..self.offset]
    }
}

/// The tokens of `source`. Line breaks end a statement only outside
/// parentheses and after a complete expression, so long expressions can
/// span lines.
pub(super) fn lex(source: &str) -> Result<Vec<Spanned>, CompileError> {
    let mut cursor = Cursor {
        source,
        offset: 0,
        line: 1,
        column: 1,
    };
    let mut tokens: Vec<Spanned> = Vec::new();
    let mut depth = 0_usize;
    while let Some(c) = cursor.peek() {
        let at = cursor.location();
        let token = match c {
            '#' => {
                cursor.bump_while(|c| c != '\n');
                continue;
            }
            '\n' | ';' => {
                cursor.bump();
                let continued = tokens.last().is_none_or(|(t, _)| t.continues());
                if depth > 0 || (c == '\n' && continued) {
                    continue;
                }
                Token::Sep
            }
            c if c.is_whitespace() => {
                cursor.bump();
                continue;
            }
            '(' | ')' | ',' | '+' | '-' | '*' | '/' | '=' => {
                cursor.bump();
                match c {
                    '(' => {
                        depth += 1;
                        Token::Open
                    }
                    ')' => {
                        depth = depth.saturating_sub(1);
                        Token::Close
                    }
                    ',' => Token::Comma,
                    '=' => Token::Assign,
                    op => Token::Op(op),
                }
            }
            '0'..='9' | '.' => number(&mut cursor, at)?,
            c if c.is_ascii_alphabetic() || c == '_' => {
                let name = cursor.bump_while(|c| c.is_ascii_alphanumeric() || c == '_');
                Token::Name(name.to_string())
            }
            other => {
                return Err(CompileError::at(at, format!("unexpected `{other}`")));
            }
        };
        tokens.push((token, at));
    }
    tokens.push((Token::End, cursor.location()));
    Ok(tokens)
}

/// A number: digits with an optional point, then an optional exponent.
fn number(cursor: &mut Cursor<'_>, at: Location) -> Result<Token, CompileError> {
    let start = cursor.offset;
    cursor.bump_while(|c| c.is_ascii_digit() || c == '.');
    let rest = &cursor.source[cursor.offset..];
    let mut exponent = rest.chars();
    if matches!(exponent.next(), Some('e' | 'E')) {
        let sign = usize::from(matches!(rest[1..].chars().next(), Some('+' | '-')));
        if rest[1 + sign..].starts_with(|c: char| c.is_ascii_digit()) {
            for _ in 0..=sign {
                cursor.bump();
            }
            cursor.bump_while(|c| c.is_ascii_digit());
        }
    }
    let text = &cursor.source[start..cursor.offset];
    text.parse::<f64>()
        .map(Token::Number)
        .map_err(|_| CompileError::at(at, format!("invalid number `{text}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> Vec<Token> {
        Vec::from_iter(lex(source).unwrap().into_iter().map(|(t, _)| t))
    }

    #[test]
    fn numbers() {
        assert_eq!(
            tokens("1 2.5 .5 1e3 1.5E-2 3e"),
            [
                Token::Number(1.0),
                Token::Number(2.5),
                Token::Number(0.5),
                Token::Number(1000.0),
                Token::Number(0.015),
                Token::Number(3.0),
                Token::Name("e".to_string()),
                Token::End
            ]
        );
        let err = lex("1..2").unwrap_err();
        assert_eq!(err.to_string(), "error at 1:1: invalid number `1..2`");
    }

    #[test]
    fn line_breaks_end_statements_only_after_complete_expressions() {
        use Token::*;
        let name = |n: &str| Name(n.to_string());
        assert_eq!(
            tokens("a = x +\n  1\nb = sma(x,\n 2) # comment\n\n"),
            [
                name("a"),
                Assign,
                name("x"),
                Op('+'),
                Number(1.0),
                Sep,
                name("b"),
                Assign,
                name("sma"),
                Open,
                name("x"),
                Comma,
                Number(2.0),
                Close,
                Sep,
                End
            ]
        );
        assert_eq!(tokens("x; y"), [name("x"), Sep, name("y"), End]);
    }

    #[test]
    fn locations_count_lines_and_columns() {
        let spanned = lex("a = 1\n  b = 2").unwrap();
        let b = &spanned[4];
        assert_eq!(b.0, Token::Name("b".to_string()));
        assert_eq!((b.1.line(), b.1.column(), b.1.offset()), (2, 3, 8));
        let err = lex("x\n  $").unwrap_err();
        assert_eq!(err.to_string(), "error at 2:3: unexpected `$`");
    }
}

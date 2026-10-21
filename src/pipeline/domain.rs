// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The value range of a node, tracked by the compiler to find problems
//! before a pipeline sees any data.

use std::fmt;

/// One end of a [`Domain`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bound {
    value: f64,
    /// Whether `value` itself belongs to the domain. Infinite bounds are
    /// always open, because a node only ever outputs finite values.
    closed: bool,
}

impl Bound {
    fn new(value: f64, closed: bool) -> Self {
        assert!(!value.is_nan(), "a bound is a number");
        Self {
            // Adding zero turns -0 into 0, which prints as `0`.
            value: value + 0.0,
            closed: closed && value.is_finite(),
        }
    }
}

/// A set of real numbers: an interval whose ends are open, closed or
/// infinite, e.g. `(0, inf)` or `[0, 100]`.
///
/// The compiler derives the domain of every node from the domains of the
/// inputs, to report a function that can never be ready (e.g. `ln` of a
/// negative argument) or an output that is constant before any data is
/// seen, and to tell the range of each output. Domains over-approximate:
/// a node's values always lie inside its domain, but may not cover it.
///
/// Arithmetic on domains follows real arithmetic, so rounding can step
/// past an open end (a product of two tiny positives can underflow to 0).
/// Functions that saturate in floating point, like `tanh` or `exp`, reach
/// their limits, so their domains include them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Domain {
    lo: Bound,
    hi: Bound,
}

impl Domain {
    /// All finite real numbers, `(-inf, inf)`.
    pub const REAL: Domain = Domain {
        lo: Bound {
            value: f64::NEG_INFINITY,
            closed: false,
        },
        hi: Bound {
            value: f64::INFINITY,
            closed: false,
        },
    };

    /// The domain between two bounds, `None` if it is empty.
    fn new(lo: Bound, hi: Bound) -> Option<Self> {
        let non_empty = lo.value < hi.value || (lo.value == hi.value && lo.closed && hi.closed);
        non_empty.then_some(Self { lo, hi })
    }

    /// The domain between two bounds that rounding may have collapsed
    /// onto one value: then that value, which over-approximates, or all
    /// numbers if the value overflowed.
    fn new_or_point(lo: Bound, hi: Bound) -> Self {
        Self::new(lo, hi).unwrap_or_else(|| {
            assert_eq!(lo.value, hi.value, "bounds are ordered");
            if lo.value.is_finite() {
                Self::point(lo.value)
            } else {
                Self::REAL
            }
        })
    }

    /// The positive numbers, `(0, inf)`, e.g. of prices.
    pub fn positive() -> Self {
        Self {
            lo: Bound::new(0.0, false),
            hi: Self::REAL.hi,
        }
    }

    /// The non-negative numbers, `[0, inf)`, e.g. of volumes.
    pub fn non_negative() -> Self {
        Self {
            lo: Bound::new(0.0, true),
            hi: Self::REAL.hi,
        }
    }

    /// The numbers from `lo` to `hi`, both included (infinite ends are
    /// open).
    ///
    /// # Panics
    ///
    /// If a bound is NaN or `lo > hi`.
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self::new(Bound::new(lo, true), Bound::new(hi, true))
            .expect("the lower bound must not exceed the upper bound")
    }

    /// The numbers strictly between `lo` and `hi`.
    ///
    /// # Panics
    ///
    /// If a bound is NaN or `lo >= hi`.
    pub fn open(lo: f64, hi: f64) -> Self {
        Self::new(Bound::new(lo, false), Bound::new(hi, false))
            .expect("the lower bound must be below the upper bound")
    }

    /// The single number `value`.
    ///
    /// # Panics
    ///
    /// If `value` is not finite.
    pub fn point(value: f64) -> Self {
        assert!(value.is_finite(), "a point domain holds a finite value");
        Self::closed(value, value)
    }

    /// The lower end, `-inf` when unbounded below.
    pub fn lo(&self) -> f64 {
        self.lo.value
    }

    /// The upper end, `inf` when unbounded above.
    pub fn hi(&self) -> f64 {
        self.hi.value
    }

    /// Whether the domain holds `value`.
    pub fn contains(&self, value: f64) -> bool {
        let above_lo = value > self.lo.value || (value == self.lo.value && self.lo.closed);
        let below_hi = value < self.hi.value || (value == self.hi.value && self.hi.closed);
        above_lo && below_hi
    }

    /// The only value of a single-value domain, e.g. of a constant.
    pub fn as_point(&self) -> Option<f64> {
        (self.lo.value == self.hi.value).then_some(self.lo.value)
    }

    /// The values in both domains, `None` if there are none.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let lo = match self.lo.value.total_cmp(&other.lo.value) {
            std::cmp::Ordering::Less => other.lo,
            std::cmp::Ordering::Greater => self.lo,
            std::cmp::Ordering::Equal => {
                Bound::new(self.lo.value, self.lo.closed && other.lo.closed)
            }
        };
        let hi = match self.hi.value.total_cmp(&other.hi.value) {
            std::cmp::Ordering::Less => self.hi,
            std::cmp::Ordering::Greater => other.hi,
            std::cmp::Ordering::Equal => {
                Bound::new(self.hi.value, self.hi.closed && other.hi.closed)
            }
        };
        Self::new(lo, hi)
    }

    /// `-a` for `a` in the domain.
    pub(super) fn neg(self) -> Self {
        Self {
            lo: Bound::new(-self.hi.value, self.hi.closed),
            hi: Bound::new(-self.lo.value, self.lo.closed),
        }
    }

    /// `a + b` for `a` in `self` and `b` in `other`.
    pub(super) fn add(self, other: Self) -> Self {
        // Lower ends are below `inf` and upper ends above `-inf`, so the
        // sums are never `inf - inf`; an overflow becomes an open end.
        let lo = Bound::new(
            self.lo.value + other.lo.value,
            self.lo.closed && other.lo.closed,
        );
        let hi = Bound::new(
            self.hi.value + other.hi.value,
            self.hi.closed && other.hi.closed,
        );
        Self::new_or_point(lo, hi)
    }

    /// `a - b` for `a` in `self` and `b` in `other`.
    pub(super) fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    /// `a * b` for `a` in `self` and `b` in `other`.
    ///
    /// The extremes of a product lie at the corners, with `0 * inf = 0`.
    pub(super) fn mul(self, other: Self) -> Self {
        let corners = [
            product(self.lo, other.lo),
            product(self.lo, other.hi),
            product(self.hi, other.lo),
            product(self.hi, other.hi),
        ];
        let lo = extreme(corners, std::cmp::Ordering::Less);
        let hi = extreme(corners, std::cmp::Ordering::Greater);
        Self::new_or_point(lo, hi)
    }

    /// `a / b` for `a` in `self` and `b` in `other`, or `None` when `b` can
    /// be zero (the quotient is then unbounded).
    pub(super) fn div(self, other: Self) -> Option<Self> {
        if other.contains(0.0) {
            return None;
        }
        // Zero is outside, so `other` lies on one side of it; a zero end is
        // open and its reciprocal is the infinity on that side.
        let positive = other.lo.value >= 0.0;
        let recip = |b: Bound| {
            let value = if b.value == 0.0 {
                if positive {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                1.0 / b.value
            };
            Bound::new(value, b.closed)
        };
        let reciprocal = Self::new_or_point(recip(other.hi), recip(other.lo));
        Some(self.mul(reciprocal))
    }

    /// `|a|` for `a` in the domain.
    pub(super) fn abs(self) -> Self {
        if self.lo.value >= 0.0 {
            return self;
        }
        if self.hi.value <= 0.0 {
            return self.neg();
        }
        let (neg, pos) = (-self.lo.value, self.hi.value);
        let hi = match neg.total_cmp(&pos) {
            std::cmp::Ordering::Less => self.hi,
            std::cmp::Ordering::Greater => Bound::new(neg, self.lo.closed),
            std::cmp::Ordering::Equal => Bound::new(pos, self.lo.closed || self.hi.closed),
        };
        Self::new_or_point(Bound::new(0.0, true), hi)
    }

    /// `f(a)` for `a` in the domain and a non-decreasing `f` that is
    /// defined on all of it.
    ///
    /// Finite ends of the result are closed: in floating point `f` reaches
    /// its limits (`tanh(20) == 1`, `exp(-800) == 0`).
    pub(super) fn map_increasing(self, f: impl Fn(f64) -> f64) -> Self {
        let (lo, hi) = (f(self.lo.value), f(self.hi.value));
        assert!(!lo.is_nan(), "f is defined on the domain");
        assert!(!hi.is_nan(), "f is defined on the domain");
        Self::new_or_point(Bound::new(lo, true), Bound::new(hi, true))
    }

    /// The sign of `a` for `a` in the domain: values in `{-1, 0, 1}`.
    pub(super) fn sign(self) -> Self {
        let sign = |b: Bound, open_zero: f64| {
            if b.value > 0.0 {
                1.0
            } else if b.value < 0.0 {
                -1.0
            } else if b.closed {
                0.0
            } else {
                open_zero
            }
        };
        Self::closed(sign(self.lo, 1.0), sign(self.hi, -1.0))
    }

    /// `max(a, c)` for `a` in the domain.
    pub(super) fn max_with(self, c: f64) -> Self {
        assert!(c.is_finite(), "a clipping bound is finite");
        let lo = if c > self.lo.value {
            Bound::new(c, true)
        } else {
            self.lo
        };
        let hi = if c >= self.hi.value {
            Bound::new(c, true)
        } else {
            self.hi
        };
        Self::new_or_point(lo, hi)
    }

    /// `min(a, c)` for `a` in the domain.
    pub(super) fn min_with(self, c: f64) -> Self {
        self.neg().max_with(-c).neg()
    }
}

/// A corner of a product of two domains, with `0 * inf = 0`: a zero end
/// that belongs to its domain makes the product zero whatever the other
/// factor is.
fn product(a: Bound, b: Bound) -> Bound {
    let a_zero = a.value == 0.0;
    let b_zero = b.value == 0.0;
    let value = if a_zero || b_zero {
        0.0
    } else {
        a.value * b.value
    };
    let closed = (a.closed && b.closed) || (a_zero && a.closed) || (b_zero && b.closed);
    Bound::new(value, closed)
}

/// The smallest (`Less`) or largest (`Greater`) corner; it belongs to the
/// domain if any corner of that value does.
fn extreme(corners: [Bound; 4], order: std::cmp::Ordering) -> Bound {
    corners
        .into_iter()
        .reduce(|best, c| match c.value.total_cmp(&best.value) {
            std::cmp::Ordering::Equal => Bound::new(c.value, c.closed || best.closed),
            o if o == order => c,
            _ => best,
        })
        .expect("four corners")
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let open = if self.lo.closed { '[' } else { '(' };
        let close = if self.hi.closed { ']' } else { ')' };
        write!(f, "{open}{}, {}{close}", self.lo.value, self.hi.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_constructors() {
        assert_eq!(Domain::REAL.to_string(), "(-inf, inf)");
        assert_eq!(Domain::positive().to_string(), "(0, inf)");
        assert_eq!(Domain::non_negative().to_string(), "[0, inf)");
        assert_eq!(Domain::closed(0.0, 100.0).to_string(), "[0, 100]");
        assert_eq!(Domain::open(-1.0, 1.0).to_string(), "(-1, 1)");
        assert_eq!(Domain::point(2.5).as_point(), Some(2.5));
        assert_eq!(Domain::closed(1.0, f64::INFINITY).to_string(), "[1, inf)");
        assert_eq!(Domain::REAL.as_point(), None);
    }

    #[test]
    #[should_panic(expected = "must not exceed")]
    fn reversed_bounds_panic() {
        let _ = Domain::closed(1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "must be below")]
    fn empty_open_domain_panics() {
        let _ = Domain::open(1.0, 1.0);
    }

    #[test]
    fn contains_respects_open_and_closed_ends() {
        let d = Domain::positive();
        assert!(!d.contains(0.0));
        assert!(d.contains(f64::MIN_POSITIVE));
        assert!(!d.contains(-1.0));
        assert!(Domain::non_negative().contains(0.0));
        assert!(!Domain::REAL.contains(f64::INFINITY));
        assert!(Domain::closed(1.0, 2.0).contains(2.0));
    }

    #[test]
    fn intersection() {
        let r = Domain::REAL.intersect(Domain::positive());
        assert_eq!(r, Some(Domain::positive()));
        let neg = Domain::closed(f64::NEG_INFINITY, 0.0);
        assert_eq!(neg.intersect(Domain::positive()), None);
        assert_eq!(
            neg.intersect(Domain::non_negative()),
            Some(Domain::point(0.0))
        );
        assert_eq!(
            Domain::closed(-1.0, 5.0).intersect(Domain::open(0.0, 10.0)),
            Some(Domain::new(Bound::new(0.0, false), Bound::new(5.0, true)).unwrap())
        );
    }

    #[test]
    fn arithmetic_examples() {
        let pos = Domain::positive();
        assert_eq!(pos.mul(pos), pos);
        assert_eq!(pos.add(pos), pos);
        assert_eq!(pos.sub(pos), Domain::REAL);
        assert_eq!(pos.neg().to_string(), "(-inf, 0)");
        assert_eq!(
            Domain::closed(0.0, 5.0).mul(pos),
            Domain::non_negative(),
            "0 * inf is 0, and 0 is reached"
        );
        assert_eq!(
            Domain::closed(1.0, 2.0).mul(Domain::closed(-3.0, 4.0)),
            Domain::closed(-6.0, 8.0)
        );
        assert_eq!(
            Domain::closed(1.0, 2.0).div(Domain::closed(-1.0, 1.0)),
            None
        );
        assert_eq!(
            Domain::closed(1.0, 2.0).div(Domain::closed(2.0, 4.0)),
            Some(Domain::closed(0.25, 1.0))
        );
        assert_eq!(Domain::point(1.0).div(pos), Some(pos));
        assert_eq!(Domain::point(1.0).div(pos.neg()), Some(pos.neg()));
        assert_eq!(Domain::closed(-3.0, 2.0).abs(), Domain::closed(0.0, 3.0));
        assert_eq!(Domain::REAL.abs(), Domain::non_negative());
        assert_eq!(Domain::REAL.sign(), Domain::closed(-1.0, 1.0));
        assert_eq!(pos.sign(), Domain::point(1.0));
        assert_eq!(
            Domain::REAL.map_increasing(f64::tanh),
            Domain::closed(-1.0, 1.0)
        );
        assert_eq!(
            Domain::REAL.map_increasing(f64::exp),
            Domain::non_negative(),
            "exp underflows to 0"
        );
        assert_eq!(
            Domain::REAL.max_with(-1.0),
            Domain::closed(-1.0, f64::INFINITY)
        );
        assert_eq!(Domain::closed(0.0, 1.0).min_with(-2.0), Domain::point(-2.0));
    }

    #[test]
    fn rounding_collapse_yields_a_point_not_an_empty_domain() {
        let far = Domain::open(1000.0, 2000.0);
        assert_eq!(far.map_increasing(f64::tanh), Domain::point(1.0));
        let huge = Domain::point(f64::MAX);
        assert_eq!(huge.add(huge), Domain::REAL, "an overflow is unbounded");
    }

    /// Bounds that exercise zeros, infinities and open ends.
    fn random_domain(rng: &romu::Rng) -> Domain {
        const VALUES: [f64; 9] = [
            f64::NEG_INFINITY,
            -10.0,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            10.0,
            f64::INFINITY,
        ];
        loop {
            let pick = || VALUES[rng.range_usize(0..VALUES.len())];
            let (a, b) = (pick(), pick());
            let lo = Bound::new(a.min(b), rng.bool());
            let hi = Bound::new(a.max(b), rng.bool());
            if let Some(d) = Domain::new(lo, hi) {
                return d;
            }
        }
    }

    /// A value of the domain, often one of its closed ends.
    fn random_member(d: Domain, rng: &romu::Rng) -> f64 {
        loop {
            let v = match rng.range_usize(0..4) {
                0 => d.lo(),
                1 => d.hi(),
                _ => {
                    let lo = d.lo().max(-20.0);
                    let hi = d.hi().min(20.0);
                    lo + (hi - lo) * rng.f64()
                }
            };
            if d.contains(v) {
                return v;
            }
        }
    }

    /// Soundness: every result of an operation on members of domains is a
    /// member of the operation's domain.
    #[test]
    fn operations_contain_every_result() {
        let rng = romu::Rng::from_seed_with_64bit(3);
        for _ in 0..20_000 {
            let (da, db) = (random_domain(&rng), random_domain(&rng));
            let (a, b) = (random_member(da, &rng), random_member(db, &rng));
            let c = random_member(Domain::closed(-2.0, 2.0), &rng);
            let check = |name: &str, d: Domain, v: f64| {
                assert!(
                    d.contains(v),
                    "{name}: {v} not in {d} for {a} in {da}, {b} in {db}"
                );
            };
            check("neg", da.neg(), -a);
            check("add", da.add(db), a + b);
            check("sub", da.sub(db), a - b);
            check("mul", da.mul(db), a * b);
            if let Some(q) = da.div(db) {
                check("div", q, a / b);
            } else {
                assert!(db.contains(0.0));
            }
            check("abs", da.abs(), a.abs());
            let sign = if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else {
                0.0
            };
            check("sign", da.sign(), sign);
            check("tanh", da.map_increasing(f64::tanh), a.tanh());
            check("exp", da.map_increasing(f64::exp), a.exp());
            check("max_with", da.max_with(c), a.max(c));
            check("min_with", da.min_with(c), a.min(c));
            if let Some(i) = da.intersect(db) {
                if da.contains(b) {
                    check("intersect", i, b);
                }
            } else {
                assert!(!da.contains(b), "{b} in {da} and {db}");
            }
        }
    }
}

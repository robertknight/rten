//! Symbolic expressions representing integer values.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// A named variable.
///
/// The variable may carry assumptions about its value, such as being >= 0.
///
/// Two symbols are equal if they have the same name.
#[derive(Clone, PartialEq)]
pub struct Symbol {
    pub name: String,

    // True if this value is assumed to be >= 0.
    pub positive: bool,
}

/// Symbolic expression representing an integer value.
///
/// Expressions can be known integer values, named symbols or composite
/// expressions.
#[derive(Clone)]
pub enum SymExpr {
    /// Element with a known integer value.
    Value(i32),
    /// Symbolic value
    Var(Arc<Symbol>),
    /// Addition of two symbolic values
    Add(Arc<SymExpr>, Arc<SymExpr>),
    /// Subtraction of two symbolic values
    Sub(Arc<SymExpr>, Arc<SymExpr>),
    /// Multiplication of two symbolic values
    Mul(Arc<SymExpr>, Arc<SymExpr>),
    /// Flooring division of first expression by second.
    Div(Arc<SymExpr>, Arc<SymExpr>),
    /// Ceiling division of first expression by second.
    DivCeil(Arc<SymExpr>, Arc<SymExpr>),
    /// Maximum of two symbolic values
    Max(Arc<SymExpr>, Arc<SymExpr>),
    /// Minimum of two symbolic values
    Min(Arc<SymExpr>, Arc<SymExpr>),
    /// Broadcast two symbolic values.
    ///
    /// This behaves like `Max`, except it implies that both expressions are
    /// positive and either equal or 1.
    Broadcast(Arc<SymExpr>, Arc<SymExpr>),
    /// Negation of a value
    Neg(Arc<SymExpr>),
}

impl SymExpr {
    /// Return the range of possible values this element may have.
    pub fn range(&self) -> (i32, i32) {
        match self {
            Self::Value(x) => (*x, *x),
            Self::Var(sym) => {
                if sym.positive {
                    (0, i32::MAX)
                } else {
                    (i32::MIN, i32::MAX)
                }
            }
            Self::Neg(x) => {
                if x.is_positive() {
                    (i32::MIN, -1)
                } else {
                    (i32::MIN, i32::MAX)
                }
            }
            Self::Add(lhs, rhs)
            | Self::Mul(lhs, rhs)
            | Self::Max(lhs, rhs)
            | Self::Min(lhs, rhs)
            | Self::Div(lhs, rhs)
            | Self::DivCeil(lhs, rhs) => {
                let (lhs_min, lhs_max) = lhs.range();
                let (rhs_min, rhs_max) = rhs.range();
                (lhs_min.min(rhs_min), lhs_max.max(rhs_max))
            }
            Self::Sub(_lhs, _rhs) => {
                // Note: Unlike for addition, subtraction involving two
                // positive symbols may produce a negative result.
                (i32::MIN, i32::MAX)
            }
            Self::Broadcast(lhs, rhs) => {
                let (lhs_min, lhs_max) = lhs.range();
                let (rhs_min, rhs_max) = rhs.range();
                (lhs_min.min(rhs_min).max(0), lhs_max.max(rhs_max).max(0))
            }
        }
    }

    /// Return true if the value of this expression is known to be >= 0.
    pub fn is_positive(&self) -> bool {
        match self {
            Self::Value(x) => *x >= 0,
            Self::Var(sym) => sym.positive,
            Self::Neg(_expr) => false,
            Self::Add(lhs, rhs) => lhs.is_positive() && rhs.is_positive(),
            Self::Sub(_lhs, _rhs) => false,
            Self::Mul(lhs, rhs) => lhs.is_positive() && rhs.is_positive(),
            Self::Div(lhs, rhs) | Self::DivCeil(lhs, rhs) => lhs.is_positive() && rhs.is_positive(),
            Self::Max(lhs, rhs) => lhs.is_positive() || rhs.is_positive(),
            Self::Min(lhs, rhs) => lhs.is_positive() && rhs.is_positive(),
            Self::Broadcast(..) => true,
        }
    }

    /// Return the maximum of `self` and `other`.
    pub fn max(&self, other: &SymExpr) -> SymExpr {
        Self::Max(self.clone().into(), other.clone().into())
    }

    /// Return the minimum of `self` and `other`.
    pub fn min(&self, other: &SymExpr) -> SymExpr {
        Self::Min(self.clone().into(), other.clone().into())
    }

    /// Return the result of broadcasting `self` and `other`.
    pub fn broadcast(&self, other: &SymExpr) -> SymExpr {
        Self::Broadcast(self.clone().into(), other.clone().into())
    }

    /// Return the result of dividing `self` by `other`, rounded up.
    pub fn div_ceil(&self, other: &SymExpr) -> SymExpr {
        Self::DivCeil(self.clone().into(), other.clone().into())
    }

    fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    // Re-order and re-associate operands of commutative and associative
    // operations so that constants are on the left or "canonical order".
    //
    // For example `Mul(Mul(a, 2), Mul(b, 3))` becomes
    // `Mul(Mul(2, 3), Mul(a, b))`.
    fn canonicalize(&self) -> SymExpr {
        fn collect_terms(
            terms: &mut Vec<SymExpr>,
            term: &SymExpr,
            extract_lhs_rhs: &impl Fn(&SymExpr) -> Option<(&Arc<SymExpr>, &Arc<SymExpr>)>,
        ) {
            if let Some((lhs, rhs)) = extract_lhs_rhs(term) {
                collect_terms(terms, lhs, extract_lhs_rhs);
                collect_terms(terms, rhs, extract_lhs_rhs);
            } else {
                terms.push(term.canonicalize());
            }
        }

        // Re-associate and simplify terms in a nested associative expression.
        //
        // This operates in 4 steps:
        //
        // 1. Collect all the terms in nested expressions of the same type
        // 2. Sort the terms in canonical order
        // 3. Simplify the result by removing any redundant terms
        // 4. Reduce the terms into a new expression, or return `default` if
        //    step (3) removed all the terms
        fn reassociate_terms(
            term: &SymExpr,
            extract_terms: &impl Fn(&SymExpr) -> Option<(&Arc<SymExpr>, &Arc<SymExpr>)>,
            simplify: impl Fn(Vec<SymExpr>) -> Vec<SymExpr>,
            default: SymExpr,
            reduce: impl Fn(SymExpr, SymExpr) -> SymExpr,
        ) -> SymExpr {
            let mut terms = Vec::new();
            collect_terms(&mut terms, term, extract_terms);
            terms.sort_by(cmp_values_first);
            let terms = simplify(terms);
            terms.into_iter().reduce(reduce).unwrap_or(default)
        }

        // Remove adjacent equal terms.
        //
        // This is a simplification for idempotent operations
        // (eg. max(x, max(x, y)) => max(x, y)).
        let remove_adjacent_equal_terms = |mut terms: Vec<SymExpr>| {
            let mut idx = 0;
            while idx < terms.len().saturating_sub(1) {
                if terms[idx] == terms[idx + 1].clone() {
                    terms.remove(idx);
                } else {
                    idx += 1;
                }
            }
            terms
        };

        match self {
            Self::Value(_) | Self::Var(_) => self.clone(),
            Self::Neg(expr) => Self::Neg(expr.canonicalize().into()),
            Self::Mul(..) => reassociate_terms(
                self,
                &|term| {
                    if let Self::Mul(lhs, rhs) = term {
                        Some((lhs, rhs))
                    } else {
                        None
                    }
                },
                |terms| terms,
                SymExpr::Value(1),
                |prod, x| prod * x,
            ),
            Self::Add(..) => {
                // Remove adjacent terms which cancel.
                let remove_adjacent_opposite_terms = |mut terms: Vec<SymExpr>| {
                    let mut idx = 0;
                    while idx < terms.len().saturating_sub(1) {
                        if terms[idx].is_negation_of(&terms[idx + 1]) {
                            terms.remove(idx);
                            terms.remove(idx);
                        } else {
                            idx += 1;
                        }
                    }
                    terms
                };

                reassociate_terms(
                    self,
                    &|term| match term {
                        Self::Add(lhs, rhs) => Some((lhs, rhs)),
                        _ => None,
                    },
                    remove_adjacent_opposite_terms,
                    SymExpr::Value(0),
                    |sum, x| sum + x,
                )
            }
            Self::Max(..) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Max(lhs, rhs) => Some((lhs, rhs)),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymExpr::Value(i32::MIN),
                |max, x| max.max(&x),
            ),
            Self::Min(..) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Min(lhs, rhs) => Some((lhs, rhs)),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymExpr::Value(i32::MAX),
                |min, x| min.min(&x),
            ),
            Self::Sub(lhs, rhs) => {
                // Rewrite `x - y` as `x + (-y)`. This makes it easier to
                // simplify expressions by canceling opposite terms.
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::Add(lhs.into(), (-rhs).into()).canonicalize()
            }
            Self::Div(lhs, rhs) => {
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::Div(lhs.into(), rhs.into())
            }
            Self::DivCeil(lhs, rhs) => {
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::DivCeil(lhs.into(), rhs.into())
            }
            Self::Broadcast(..) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Broadcast(lhs, rhs) => Some((lhs, rhs)),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymExpr::Value(1),
                |result, x| result.broadcast(&x),
            ),
        }
    }

    /// Simplify an expression.
    ///
    /// This simplifies expressions such as identities (eg. `x + 0` becomes `x`).
    pub fn simplify(&self) -> SymExpr {
        self.canonicalize().simplify_canonical()
    }

    /// Simplify an expression which is assumed to have been put in canonical
    /// form by [`canonicalize`](Self::canonicalize).
    fn simplify_canonical(self) -> SymExpr {
        match self {
            Self::Value(_) | Self::Var(_) => self.clone(),
            Self::Neg(expr) => match Arc::unwrap_or_clone(expr).simplify_canonical() {
                SymExpr::Value(x) => SymExpr::Value(-x),
                expr => Self::Neg(expr.into()),
            },
            Self::Add(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                match (lhs, rhs) {
                    (SymExpr::Value(0), rhs) => rhs,
                    (lhs, SymExpr::Value(0)) => lhs,
                    (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x + y),
                    (lhs, SymExpr::Neg(rhs)) if lhs == *rhs => SymExpr::Value(0),
                    (lhs, rhs) => lhs + rhs,
                }
            }
            Self::Sub(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                match (lhs, rhs) {
                    (lhs, SymExpr::Value(0)) => lhs,
                    (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x - y),
                    (lhs, rhs) if lhs == rhs => SymExpr::Value(0),
                    (lhs, rhs) => lhs - rhs,
                }
            }
            Self::Mul(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                match (lhs, rhs) {
                    (SymExpr::Value(1), rhs) => rhs,
                    (lhs, SymExpr::Value(1)) => lhs,
                    (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x * y),
                    (lhs, rhs) => lhs * rhs,
                }
            }
            Self::Div(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();
                let (lhs, rhs) = remove_common_factors(lhs, rhs);

                match (lhs, rhs) {
                    (lhs, SymExpr::Value(1)) => lhs,
                    (SymExpr::Value(x), SymExpr::Value(y)) if y != 0 => SymExpr::Value(x / y),
                    // x / b / c => x / (b * c)
                    (SymExpr::Div(lhs, c1), c2) => match (&*c1, c2) {
                        (SymExpr::Value(c1), SymExpr::Value(c2)) if *c1 != 0 && c2 != 0 => {
                            (*lhs).clone() / SymExpr::Value(c1 * c2)
                        }
                        (c1, c2) => (*lhs).clone() / (c1.clone() * c2),
                    },
                    (lhs, rhs) => lhs / rhs,
                }
            }
            Self::DivCeil(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                match (lhs, rhs) {
                    (lhs, SymExpr::Value(1)) => lhs,
                    (SymExpr::Value(x), SymExpr::Value(y)) if y != 0 => {
                        SymExpr::Value(div_ceil(x, y))
                    }
                    // x/x => 1
                    //
                    // Where we assume the RHS is non-zero.
                    //
                    // This is a special case of canceling common terms. The
                    // more general case (eg. XY / XZ => Y/Z) still needs to
                    // be implemented.
                    (lhs, rhs) if lhs == rhs => SymExpr::Value(1),

                    // x.div_ceil(b).div_ceil(c) => x.div_ceil(b * c) if b > 0
                    // and c > 0.
                    (SymExpr::DivCeil(lhs, c1), c2) => match (&*c1, c2) {
                        (SymExpr::Value(c1), SymExpr::Value(c2)) if *c1 > 0 && c2 > 0 => {
                            lhs.div_ceil(&SymExpr::Value(c1 * c2))
                        }
                        (c1, c2) => lhs.div_ceil(&(c1.clone() * c2)),
                    },
                    (lhs, rhs) => lhs.div_ceil(&rhs),
                }
            }
            Self::Max(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                if lhs == rhs {
                    lhs
                } else {
                    match (lhs, rhs) {
                        (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x.max(y)),
                        (lhs, rhs) => Self::Max(lhs.into(), rhs.into()),
                    }
                }
            }
            Self::Min(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                if lhs == rhs {
                    lhs
                } else {
                    match (lhs, rhs) {
                        (SymExpr::Value(x), SymExpr::Value(y)) => SymExpr::Value(x.min(y)),
                        (lhs, rhs) => Self::Min(lhs.into(), rhs.into()),
                    }
                }
            }
            Self::Broadcast(lhs, rhs) => {
                let lhs = Arc::unwrap_or_clone(lhs).simplify_canonical();
                let rhs = Arc::unwrap_or_clone(rhs).simplify_canonical();

                match (lhs, rhs) {
                    (SymExpr::Value(x), SymExpr::Value(y)) if x == y => SymExpr::Value(x),
                    (SymExpr::Value(1), y) => y,
                    (x, SymExpr::Value(1)) => x,
                    (SymExpr::Value(x), y) if x != 1 => SymExpr::Value(x),
                    (x, SymExpr::Value(y)) if y != 1 => SymExpr::Value(y),
                    (lhs, rhs) if lhs == rhs => lhs,
                    (lhs, rhs) => SymExpr::Broadcast(lhs.into(), rhs.into()),
                }
            }
        }
    }

    /// Return the precedence of the operator.
    ///
    /// This is used to add parentheses when formatting an expression tree.
    fn precedence(&self) -> u8 {
        match self {
            // Functions and atomic values have the maximum precedence, so they
            // never need to be wrapped in parens when formatting an expression.
            Self::Value(_) | Self::Var(_) | Self::Max(..) | Self::Min(..) | Self::Broadcast(..) => {
                4
            }
            Self::Div(..) | Self::DivCeil(..) => 3,
            Self::Mul(..) => 2,
            Self::Add(..) => 1,
            Self::Sub(..) | Self::Neg(_) => 0,
        }
    }

    /// Create a named symbol, with no assumptions about the value.
    pub fn var(name: &str) -> Self {
        SymExpr::Var(
            Symbol {
                name: name.to_string(),
                positive: false,
            }
            .into(),
        )
    }

    /// Create a named symbol representing a positive value (ie. `>= 0`).
    pub fn pos_var(name: &str) -> Self {
        SymExpr::Var(
            Symbol {
                name: name.to_string(),
                positive: true,
            }
            .into(),
        )
    }

    /// Return the name of the symbol in a unary expression.
    ///
    /// Returns `None` if the expression is not unary or has a fixed value.
    fn name(&self) -> Option<&str> {
        match self {
            SymExpr::Value(_) => None,
            SymExpr::Var(sym) => Some(&sym.name),
            SymExpr::Neg(x) => x.name(),
            SymExpr::Add(..)
            | SymExpr::Sub(..)
            | SymExpr::Mul(..)
            | SymExpr::Div(..)
            | SymExpr::DivCeil(..)
            | SymExpr::Max(..)
            | SymExpr::Min(..)
            | SymExpr::Broadcast(..) => None,
        }
    }

    /// Return true if `self` and `other` are negations of each other, meaning
    /// that adding the two terms together will produce zero.
    fn is_negation_of(&self, other: &SymExpr) -> bool {
        match (self, other) {
            (x, SymExpr::Neg(y)) if *x == **y => true,
            (SymExpr::Neg(x), y) if **x == *y => true,
            _ => false,
        }
    }
}

/// Sort terms in an order that makes simplification easier, by making terms
/// which can be combined or eliminated adjacent.
fn cmp_values_first(a: &SymExpr, b: &SymExpr) -> Ordering {
    match (a.is_value(), b.is_value()) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => match (a.name(), b.name()) {
            (Some(a_name), Some(b_name)) => a_name.cmp(b_name),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            _ => Ordering::Equal,
        },
    }
}

/// Remove common factors from `lhs` and `rhs`.
fn remove_common_factors(lhs: SymExpr, rhs: SymExpr) -> (SymExpr, SymExpr) {
    fn collect_terms(terms: &mut Vec<SymExpr>, term: &SymExpr) {
        if let SymExpr::Mul(lhs, rhs) = term {
            collect_terms(terms, lhs);
            collect_terms(terms, rhs);
        } else {
            terms.push(term.clone());
        }
    }

    // Collect multiplication terms from LHS and RHS
    let mut lhs_terms = Vec::new();
    collect_terms(&mut lhs_terms, &lhs);

    let mut rhs_terms = Vec::new();
    collect_terms(&mut rhs_terms, &rhs);

    // Remove common factors from `lhs_terms` and `rhs_terms`
    let mut i = 0;
    while i < lhs_terms.len() {
        let lhs_term = &lhs_terms[i];
        let k = rhs_terms.iter().position(|t| lhs_term == t);
        if let Some(k) = k {
            lhs_terms.remove(i);
            rhs_terms.remove(k);
        } else {
            i += 1;
        }
    }

    // Construct simplified LHS and RHS
    let lhs = lhs_terms
        .into_iter()
        .reduce(|prod, x| prod * x)
        .unwrap_or(SymExpr::Value(1));
    let rhs = rhs_terms
        .into_iter()
        .reduce(|prod, x| prod * x)
        .unwrap_or(SymExpr::Value(1));
    (lhs, rhs)
}

impl PartialEq<SymExpr> for SymExpr {
    fn eq(&self, other: &SymExpr) -> bool {
        let commutative_eq = |self_lhs, self_rhs, other_lhs, other_rhs| {
            (self_lhs == other_lhs && self_rhs == other_rhs)
                || (self_lhs == other_rhs && self_rhs == other_lhs)
        };

        // Symbols are equal if they have the same value or the same name.
        match self {
            Self::Value(x) => match other {
                Self::Value(y) => x == y,
                _ => false,
            },
            Self::Var(x) => match other {
                Self::Var(y) => x.name == y.name,
                _ => false,
            },
            Self::Neg(x) => match other {
                Self::Neg(y) => x == y,
                _ => false,
            },
            Self::Add(a, b) => match other {
                Self::Add(c, d) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Mul(a, b) => match other {
                Self::Mul(c, d) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Max(a, b) => match other {
                Self::Max(c, d) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Min(a, b) => match other {
                Self::Min(c, d) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Sub(a, b) => match other {
                Self::Sub(c, d) => a == c && b == d,
                _ => false,
            },
            Self::Div(a, b) => match other {
                Self::Div(c, d) => a == c && b == d,
                _ => false,
            },
            Self::DivCeil(a, b) => match other {
                Self::DivCeil(c, d) => a == c && b == d,
                _ => false,
            },
            Self::Broadcast(a, b) => match other {
                Self::Broadcast(c, d) => commutative_eq(a, b, c, d),
                _ => false,
            },
        }
    }
}

impl Add<SymExpr> for SymExpr {
    type Output = SymExpr;

    fn add(self, rhs: SymExpr) -> Self {
        Self::Add(self.into(), rhs.into())
    }
}

impl Sub<SymExpr> for SymExpr {
    type Output = SymExpr;

    fn sub(self, rhs: SymExpr) -> Self {
        Self::Sub(self.into(), rhs.into())
    }
}

impl AddAssign<SymExpr> for SymExpr {
    fn add_assign(&mut self, rhs: SymExpr) {
        *self = Self::Add(self.clone().into(), rhs.into());
    }
}

impl Mul<SymExpr> for SymExpr {
    type Output = SymExpr;

    fn mul(self, rhs: SymExpr) -> Self {
        Self::Mul(self.into(), rhs.into())
    }
}

impl Div<SymExpr> for SymExpr {
    type Output = SymExpr;

    fn div(self, rhs: SymExpr) -> Self {
        Self::Div(self.into(), rhs.into())
    }
}

impl Neg for SymExpr {
    type Output = SymExpr;

    fn neg(self) -> Self {
        Self::Neg(self.into())
    }
}

impl From<Symbol> for SymExpr {
    fn from(val: Symbol) -> Self {
        Self::Var(val.into())
    }
}

/// Create a symbol with a given name and an assumption that the value is
/// positive (`>= 0`).
///
/// The rationale for the positivity assumption is that during shape inference,
/// the most common use of symbols is to represent dimension sizes.
impl<'a> From<&'a str> for SymExpr {
    fn from(name: &'a str) -> Self {
        SymExpr::Var(
            Symbol {
                name: name.to_string(),
                positive: true,
            }
            .into(),
        )
    }
}

impl From<i32> for SymExpr {
    fn from(val: i32) -> Self {
        SymExpr::Value(val)
    }
}

impl fmt::Debug for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let add_parens = |f: &mut fmt::Formatter<'_>, expr: &SymExpr| {
            if expr.precedence() < self.precedence() {
                write!(f, "({:?})", expr)
            } else {
                write!(f, "{:?}", expr)
            }
        };
        let write_binop = |f: &mut fmt::Formatter<'_>, op, lhs, rhs| {
            add_parens(f, lhs)?;
            write!(f, " {op} ")?;
            add_parens(f, rhs)
        };
        match self {
            Self::Value(val) => write!(f, "{}", val),
            Self::Var(sym) => write!(
                f,
                "\"{}\"{}",
                sym.name,
                if sym.positive { 'u' } else { 'i' }
            ),
            // nb. No space between "-" and expression to make formatting
            // distinct from subtraction.
            Self::Neg(expr) => write!(f, "-{:?}", expr),
            Self::Add(lhs, rhs) => write_binop(f, '+', lhs, rhs),
            Self::Sub(lhs, rhs) => write_binop(f, '-', lhs, rhs),
            Self::Mul(lhs, rhs) => write_binop(f, '*', lhs, rhs),
            Self::Div(lhs, rhs) => write_binop(f, '/', lhs, rhs),
            Self::DivCeil(lhs, rhs) => write!(f, "ceil_div({:?}, {:?})", lhs, rhs),
            Self::Max(lhs, rhs) => write!(f, "max({:?}, {:?})", lhs, rhs),
            Self::Min(lhs, rhs) => write!(f, "min({:?}, {:?})", lhs, rhs),
            Self::Broadcast(lhs, rhs) => write!(f, "broadcast({:?}, {:?})", lhs, rhs),
        }
    }
}

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let add_parens = |f: &mut fmt::Formatter<'_>, expr: &SymExpr| {
            if expr.precedence() < self.precedence() {
                write!(f, "({})", expr)
            } else {
                write!(f, "{}", expr)
            }
        };
        let write_binop = |f: &mut fmt::Formatter<'_>, op, lhs, rhs| {
            add_parens(f, lhs)?;
            write!(f, " {op} ")?;
            add_parens(f, rhs)
        };
        match self {
            Self::Value(val) => write!(f, "{}", val),
            Self::Var(sym) => write!(f, "{}", sym.name),
            // nb. No space between "-" and expression to make formatting
            // distinct from subtraction.
            Self::Neg(expr) => write!(f, "-{}", expr),
            Self::Add(lhs, rhs) => write_binop(f, '+', lhs, rhs),
            Self::Sub(lhs, rhs) => write_binop(f, '-', lhs, rhs),
            Self::Mul(lhs, rhs) => write_binop(f, '*', lhs, rhs),
            Self::Div(lhs, rhs) => write_binop(f, '/', lhs, rhs),
            Self::DivCeil(lhs, rhs) => write!(f, "ceil_div({}, {})", lhs, rhs),
            Self::Max(lhs, rhs) => write!(f, "max({}, {})", lhs, rhs),
            Self::Min(lhs, rhs) => write!(f, "min({}, {})", lhs, rhs),
            Self::Broadcast(lhs, rhs) => write!(f, "broadcast({}, {})", lhs, rhs),
        }
    }
}

/// Copied from unstable [`i32::div_ceil`] in the standard library.
pub const fn div_ceil(lhs: i32, rhs: i32) -> i32 {
    let d = lhs / rhs;
    let r = lhs % rhs;

    // When remainder is non-zero we have a.div_ceil(b) == 1 + a.div_floor(b),
    // so we can re-use the algorithm from div_floor, just adding 1.
    let correction = 1 + ((lhs ^ rhs) >> (i32::BITS - 1));
    if r != 0 { d + correction } else { d }
}

#[cfg(test)]
mod tests {
    use super::SymExpr;

    #[test]
    fn test_range() {
        let x = SymExpr::pos_var("x");
        assert_eq!(x.range(), (0, i32::MAX));

        let y = SymExpr::var("y");
        assert_eq!(y.range(), (i32::MIN, i32::MAX));
    }

    #[test]
    fn test_simplify_add() {
        let x = SymExpr::pos_var("x");
        let zero = SymExpr::from(0);
        let one = SymExpr::from(1);

        let expr = x.clone() + zero.clone();
        assert_eq!(expr, SymExpr::Add(x.clone().into(), zero.clone().into()));
        assert_eq!(expr.simplify(), x);

        let expr_2 = x.clone() + one.clone();
        assert_eq!(
            expr_2.simplify(),
            SymExpr::Add(x.clone().into(), one.clone().into())
        );
    }

    // Check `C + X + D` is simplified to `S + X` where C and D are
    // constants and `S = C+D`.
    #[test]
    fn test_simplify_add_reassociate() {
        let x = SymExpr::from("x");
        let c1 = SymExpr::from(3);
        let c2 = SymExpr::from(4);

        // C + X + D => S + X
        let expr = (x.clone() + c1.clone()) + c2.clone();
        let simplified = expr.simplify();
        assert_eq!(simplified, SymExpr::from(7) + x.clone());

        // C + X + D + X => S + X + X
        let expr = (x.clone() + c1) + (x.clone() + c2);
        let simplified = expr.simplify();
        assert_eq!(simplified, SymExpr::from(7) + x.clone() + x);
    }

    #[test]
    fn test_simplify_sub() {
        let x = SymExpr::pos_var("x");
        let zero = SymExpr::from(0);
        let one = SymExpr::from(1);

        // x - 0 => x
        let expr = x.clone() - zero.clone();
        assert_eq!(expr, SymExpr::Sub(x.clone().into(), zero.clone().into()));
        assert_eq!(expr.simplify(), x);

        // x - x => 0
        let expr = x.clone() - x.clone();
        assert_eq!(expr.simplify(), SymExpr::Value(0));

        // x - 1 => x + (-1)
        let expr_2 = x.clone() - one.clone();
        assert_eq!(
            expr_2.simplify(),
            SymExpr::Add(x.clone().into(), SymExpr::from(-1).into())
        );

        // x + y - x => y
        let y = SymExpr::pos_var("y");
        let expr = x.clone() + y.clone() - x.clone();
        assert_eq!(expr.simplify(), y.clone());

        // x + x + y - x => x + y.
        let expr = x.clone() + x.clone() + y.clone() - x.clone();
        assert_eq!(expr.simplify(), x.clone() + y.clone());

        // x + y - x - y => 0
        let expr = x.clone() + y.clone() - x.clone() - y.clone();
        assert_eq!(expr.simplify(), 0.into());

        // -x + x => 0
        let expr = -x.clone() + x.clone();
        assert_eq!(expr.simplify(), 0.into());

        // x + (-x) => 0
        let expr = x.clone() + (-x.clone());
        assert_eq!(expr.simplify(), 0.into());

        // (x + y) - (x + y) => 0
        let expr = (x.clone() + y.clone()) - (x.clone() + y.clone());
        assert_eq!(expr.simplify(), 0.into());
    }

    #[test]
    fn test_simplify_mul() {
        let x = SymExpr::pos_var("x");
        let one = SymExpr::from(1);
        let two = SymExpr::from(2);

        let expr = x.clone() * one.clone();
        assert_eq!(expr, SymExpr::Mul(x.clone().into(), one.clone().into()));
        assert_eq!(expr.simplify(), x);

        let expr_2 = x.clone() * two.clone();
        assert_eq!(
            expr_2.simplify(),
            SymExpr::Mul(x.clone().into(), two.clone().into())
        );
    }

    #[test]
    fn test_simplify_div() {
        let x = SymExpr::pos_var("x");
        let one = SymExpr::from(1);
        let two = SymExpr::from(2);

        // Constant eval
        let expr = SymExpr::from(5) / SymExpr::from(2);
        assert_eq!(expr.simplify(), SymExpr::from(2));

        // Constant with zero divisor
        let expr = SymExpr::from(5) / SymExpr::from(0);
        assert_eq!(expr.simplify(), SymExpr::from(5) / SymExpr::from(0));

        // x / 1 => x
        let expr = x.clone() / one.clone();
        assert_eq!(expr, SymExpr::Div(x.clone().into(), one.clone().into()));
        assert_eq!(expr.simplify(), x);

        // x / x => 1
        let expr = x.clone() / x.clone();
        assert_eq!(expr.simplify(), one);

        // x / 2 => x / 2
        let expr_2 = x.clone() / two.clone();
        assert_eq!(
            expr_2.simplify(),
            SymExpr::Div(x.clone().into(), two.clone().into())
        );

        // x / 2 / 2 => x / 4
        let expr = x.clone() / two.clone() / two.clone();
        assert_eq!(expr.simplify(), x.clone() / SymExpr::from(4));

        // x / 0 / 2 => not simplified (divisor is zero)
        let zero = SymExpr::from(0);
        let expr = x.clone() / zero.clone() / two.clone();
        assert_eq!(expr.simplify(), x.clone() / (zero.clone() * two.clone()));

        // x / 2 / 0 => not simplified (divisor is zero)
        let expr = x.clone() / two.clone() / zero.clone();
        assert_eq!(expr.simplify(), x.clone() / (two.clone() * zero));

        // (x * y) / (x * z) => y / z
        let y = SymExpr::from("y");
        let z = SymExpr::from("z");
        let expr = (x.clone() * y.clone()) / (x.clone() * z.clone());
        assert_eq!(expr.simplify(), y.clone() / z.clone());

        // (x * y * z) / (x * y * 2) => z / 2
        let expr = (x.clone() * y.clone() * z.clone()) / (x.clone() * y.clone() * two.clone());
        assert_eq!(expr.simplify(), z.clone() / two.clone());

        // (x * y) / x => y
        let expr = (x.clone() * y.clone()) / x.clone();
        assert_eq!(expr.simplify(), y.clone());

        // (x * (y + z)) / x => y + z
        let expr = (x.clone() * (y.clone() + z.clone())) / x.clone();
        assert_eq!(expr.simplify(), y.clone() + z.clone());
    }

    #[test]
    fn test_simplify_div_ceil() {
        let x = SymExpr::pos_var("x");
        let one = SymExpr::from(1);
        let two = SymExpr::from(2);

        // Constant eval
        let expr = SymExpr::from(5).div_ceil(&SymExpr::from(2));
        assert_eq!(expr.simplify(), SymExpr::from(3));

        // Constant with zero divisor
        let expr = SymExpr::from(5).div_ceil(&SymExpr::from(0));
        assert_eq!(
            expr.simplify(),
            SymExpr::from(5).div_ceil(&SymExpr::from(0))
        );

        // x / 1 => x
        let expr = x.clone().div_ceil(&one);
        assert_eq!(expr, SymExpr::DivCeil(x.clone().into(), one.clone().into()));
        assert_eq!(expr.simplify(), x);

        // x / x => 1
        let expr = x.clone().div_ceil(&x);
        assert_eq!(expr.simplify(), one);

        // x / 2 => x / 2
        let expr_2 = x.clone().div_ceil(&two);
        assert_eq!(
            expr_2.simplify(),
            SymExpr::DivCeil(x.clone().into(), two.clone().into())
        );

        // x / 2 / 2 => x / 4
        let expr = x.clone().div_ceil(&two).div_ceil(&two);
        assert_eq!(expr.simplify(), x.clone().div_ceil(&SymExpr::from(4)));

        // x.div_ceil(0).div_ceil(2) => not simplified (divisor is zero)
        let zero = SymExpr::from(0);
        let expr = x.clone().div_ceil(&zero).div_ceil(&two);
        assert_eq!(
            expr.simplify(),
            x.clone().div_ceil(&(zero.clone() * two.clone()))
        );

        // x.div_ceil(-1).div_ceil(2) => not simplified (divisor is negative)
        let neg_one = SymExpr::from(-1);
        let expr = x.clone().div_ceil(&neg_one).div_ceil(&two);
        assert_eq!(expr.simplify(), x.div_ceil(&(neg_one.clone() * two)));
    }

    // Check `C * X * D` is simplified to `CD * X` where C and D are
    // constants.
    #[test]
    fn test_simplify_mul_reassociate() {
        let x = SymExpr::from("x");
        let c1 = SymExpr::from(3);
        let c2 = SymExpr::from(4);

        // C * X * D => CD * X
        let expr = (x.clone() * c1.clone()) * c2.clone();
        let simplified = expr.simplify();
        assert_eq!(simplified, SymExpr::from(12) * x.clone());

        // Same as above, but contained inside an addition expression.
        let expr = SymExpr::from(5) + expr;
        let simplified = expr.simplify();
        assert_eq!(simplified, SymExpr::from(5) + SymExpr::from(12) * x.clone());

        // C * X * D * X => CD * X * X
        let expr = (x.clone() * c1) * (x.clone() * c2);
        let simplified = expr.simplify();
        assert_eq!(simplified, SymExpr::from(12) * x.clone() * x);
    }

    #[test]
    fn test_simplify_max() {
        let one = SymExpr::from(1);
        let two = SymExpr::from(2);
        let expr = one.max(&two);

        assert_eq!(expr, SymExpr::Max(one.clone().into(), two.clone().into()));
        assert_eq!(expr.simplify(), two.clone());
    }

    #[test]
    fn test_simplify_nested_max() {
        let expr = SymExpr::from(10)
            .max(&SymExpr::from(5).max(&SymExpr::from(11)))
            .simplify();
        assert_eq!(expr, SymExpr::from(11));
    }

    #[test]
    fn test_simplify_min() {
        let one = SymExpr::from(1);
        let two = SymExpr::from(2);
        let expr = one.min(&two);

        assert_eq!(expr, SymExpr::Min(one.clone().into(), two.clone().into()));
        assert_eq!(expr.simplify(), one.clone());
    }

    #[test]
    fn test_simplify_nested_min() {
        let expr = SymExpr::from(10)
            .min(&SymExpr::from(5).min(&SymExpr::from(3)))
            .simplify();
        assert_eq!(expr, SymExpr::from(3));
    }

    #[test]
    fn test_simplify_broadcast() {
        let one = SymExpr::from(1);
        let ten = SymExpr::from(10);
        let foo = SymExpr::from("foo");

        // (x, N) where N != 1 => N
        assert_eq!(ten.broadcast(&ten).simplify(), ten.clone());
        assert_eq!(ten.broadcast(&foo).simplify(), ten.clone());
        assert_eq!(one.broadcast(&ten).simplify(), ten.clone());
        assert_eq!(ten.broadcast(&one).simplify(), ten.clone());

        // (x, N) where N == 1 => x
        assert_eq!(foo.broadcast(&one).simplify(), foo.clone());
        assert_eq!(one.broadcast(&foo).simplify(), foo.clone());

        // (x, x) => x
        assert_eq!(foo.broadcast(&foo).simplify(), foo.clone());
    }

    #[test]
    fn test_simplify_nested_broadcast() {
        let foo = SymExpr::from("foo");
        let ten = SymExpr::from(10);
        let expr = foo.broadcast(&foo.broadcast(&ten)).simplify();
        assert_eq!(expr, SymExpr::from(10));
    }

    #[test]
    fn test_simplify_neg() {
        let minus_one = -SymExpr::from(1);
        assert_eq!(minus_one.simplify(), SymExpr::from(-1));
    }

    #[test]
    fn test_display() {
        let expr = (SymExpr::from(1) + SymExpr::pos_var("foo")) * SymExpr::from(3)
            + SymExpr::from(4)
            - SymExpr::from(5);
        assert_eq!(expr.to_string(), "(1 + foo) * 3 + 4 - 5");
    }

    #[test]
    fn test_debug() {
        let expr = (SymExpr::from(1) + SymExpr::pos_var("foo")) * SymExpr::from(3)
            + SymExpr::var("bar")
            - SymExpr::from(5);
        assert_eq!(format!("{:?}", expr), "(1 + \"foo\"u) * 3 + \"bar\"i - 5");
    }
}

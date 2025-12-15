//! Tensors with symbolic shapes and values.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Vector or scalar with integer values.
#[derive(Clone, Eq, Hash, PartialEq)]
pub enum Constant {
    Scalar(i32),
    Vector(Vec<i32>),
}

impl Constant {
    pub fn ndim(&self) -> usize {
        match self {
            Self::Scalar(_) => 0,
            Self::Vector(_) => 1,
        }
    }

    pub fn values(&self) -> &[i32] {
        match self {
            Self::Scalar(elem) => std::slice::from_ref(elem),
            Self::Vector(vec) => vec.as_slice(),
        }
    }

    pub fn into_vec(self) -> Vec<i32> {
        match self {
            Self::Scalar(x) => vec![x],
            Self::Vector(vec) => vec,
        }
    }
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(val) => write!(f, "{}", val),
            Self::Vector(vec) => write!(f, "{:?}", vec),
        }
    }
}

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

/// Element in a symbolic tensor.
///
/// Elements can be integer values, named symbols or composite expressions.
#[derive(Clone)]
pub enum SymElem {
    /// Element with a known integer value.
    Value(i32),
    /// Symbolic value
    Var(Rc<Symbol>),
    /// Addition of two symbolic values
    Add((Rc<SymElem>, Rc<SymElem>)),
    /// Subtraction of two symbolic values
    Sub((Rc<SymElem>, Rc<SymElem>)),
    /// Multiplication of two symbolic values
    Mul((Rc<SymElem>, Rc<SymElem>)),
    /// Flooring division of first expression by second.
    Div((Rc<SymElem>, Rc<SymElem>)),
    /// Ceiling division of first expression by second.
    DivCeil((Rc<SymElem>, Rc<SymElem>)),
    /// Maximum of two symbolic values
    Max((Rc<SymElem>, Rc<SymElem>)),
    /// Minimum of two symbolic values
    Min((Rc<SymElem>, Rc<SymElem>)),
    /// Broadcast two symbolic values.
    ///
    /// This behaves like `Max`, except it implies that both expressions are
    /// positive and either equal or 1.
    Broadcast((Rc<SymElem>, Rc<SymElem>)),
    /// Negation of a value
    Neg(Rc<SymElem>),
}

impl SymElem {
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
            Self::Add((lhs, rhs))
            | Self::Mul((lhs, rhs))
            | Self::Max((lhs, rhs))
            | Self::Min((lhs, rhs))
            | Self::Div((lhs, rhs))
            | Self::DivCeil((lhs, rhs)) => {
                let (lhs_min, lhs_max) = lhs.range();
                let (rhs_min, rhs_max) = rhs.range();
                (lhs_min.min(rhs_min), lhs_max.max(rhs_max))
            }
            Self::Sub((_lhs, _rhs)) => {
                // Note: Unlike for addition, subtraction involving two
                // positive symbols may produce a negative result.
                (i32::MIN, i32::MAX)
            }
            Self::Broadcast((lhs, rhs)) => {
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
            Self::Add((lhs, rhs)) => lhs.is_positive() && rhs.is_positive(),
            Self::Sub((_lhs, _rhs)) => false,
            Self::Mul((lhs, rhs)) => lhs.is_positive() && rhs.is_positive(),
            Self::Div((lhs, rhs)) | Self::DivCeil((lhs, rhs)) => {
                lhs.is_positive() && rhs.is_positive()
            }
            Self::Max((lhs, rhs)) => lhs.is_positive() || rhs.is_positive(),
            Self::Min((lhs, rhs)) => lhs.is_positive() && rhs.is_positive(),
            Self::Broadcast(_) => true,
        }
    }

    /// Return the maximum of `self` and `other`.
    pub fn max(&self, other: &SymElem) -> SymElem {
        Self::Max((self.clone().into(), other.clone().into()))
    }

    /// Return the minimum of `self` and `other`.
    pub fn min(&self, other: &SymElem) -> SymElem {
        Self::Min((self.clone().into(), other.clone().into()))
    }

    /// Return the result of broadcasting `self` and `other`.
    pub fn broadcast(&self, other: &SymElem) -> SymElem {
        Self::Broadcast((self.clone().into(), other.clone().into()))
    }

    /// Return the result of dividing `self` by `other`, rounded up.
    pub fn div_ceil(&self, other: &SymElem) -> SymElem {
        Self::DivCeil((self.clone().into(), other.clone().into()))
    }

    fn is_value(&self) -> bool {
        matches!(self, Self::Value(_))
    }

    // Re-order and re-associate operands of commutative and associative
    // operations so that constants are on the left or "canonical order".
    //
    // For example `Mul(Mul(a, 2), Mul(b, 3))` becomes
    // `Mul(Mul(2, 3), Mul(a, b))`.
    fn canonicalize(&self) -> SymElem {
        fn collect_terms(
            terms: &mut Vec<SymElem>,
            term: &SymElem,
            extract_lhs_rhs: &impl Fn(&SymElem) -> Option<&(Rc<SymElem>, Rc<SymElem>)>,
        ) {
            if let Some((lhs, rhs)) = extract_lhs_rhs(term) {
                collect_terms(terms, lhs, extract_lhs_rhs);
                collect_terms(terms, rhs, extract_lhs_rhs);
            } else {
                terms.push(term.canonicalize())
            }
        }

        // Re-associate and simplify terms in a nested associative expression.
        //
        // This operates in 4 steps:
        //
        // 1. Collect all the terms in nested expressions of the same type
        // 2. Sort the terms in canonical order
        // 3. Simplify the result by removing any redundant terms
        // 4. Fold the terms back into a new expression
        fn reassociate_terms(
            term: &SymElem,
            extract_terms: &impl Fn(&SymElem) -> Option<&(Rc<SymElem>, Rc<SymElem>)>,
            simplify: impl Fn(Vec<SymElem>) -> Vec<SymElem>,
            init: SymElem,
            fold: impl Fn(SymElem, SymElem) -> SymElem,
        ) -> SymElem {
            let mut terms = Vec::new();
            collect_terms(&mut terms, term, extract_terms);
            terms.sort_by(cmp_values_first);
            let terms = simplify(terms);
            terms.into_iter().fold(init, fold)
        }

        // Remove adjacent equal terms.
        //
        // This is a simplification for idempotent operations
        // (eg. max(x, max(x, y)) => max(x, y)).
        let remove_adjacent_equal_terms = |mut terms: Vec<SymElem>| {
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
            Self::Mul(_) => reassociate_terms(
                self,
                &|term| {
                    if let Self::Mul(inner) = term {
                        Some(inner)
                    } else {
                        None
                    }
                },
                |terms| terms,
                SymElem::Value(1),
                |prod, x| prod * x,
            ),
            Self::Add(_) => {
                // Remove adjacent terms which cancel.
                let remove_adjacent_opposite_terms = |mut terms: Vec<SymElem>| {
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
                        Self::Add(inner) => Some(inner),
                        _ => None,
                    },
                    remove_adjacent_opposite_terms,
                    SymElem::Value(0),
                    |sum, x| sum + x,
                )
            }
            Self::Max(_) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Max(inner) => Some(inner),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymElem::Value(i32::MIN),
                |max, x| max.max(&x),
            ),
            Self::Min(_) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Min(inner) => Some(inner),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymElem::Value(i32::MAX),
                |min, x| min.min(&x),
            ),
            Self::Sub((lhs, rhs)) => {
                // Rewrite `x - y` as `x + (-y)`. This makes it easier to
                // simplify expressions by canceling opposite terms.
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::Add((lhs.into(), (-rhs).into())).canonicalize()
            }
            Self::Div((lhs, rhs)) => {
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::Div((lhs.into(), rhs.into()))
            }
            Self::DivCeil((lhs, rhs)) => {
                let lhs = lhs.canonicalize();
                let rhs = rhs.canonicalize();
                Self::DivCeil((lhs.into(), rhs.into()))
            }
            Self::Broadcast(_) => reassociate_terms(
                self,
                &|term| match term {
                    Self::Broadcast(inner) => Some(inner),
                    _ => None,
                },
                remove_adjacent_equal_terms,
                SymElem::Value(1),
                |result, x| result.broadcast(&x),
            ),
        }
    }

    /// Simplify an expression.
    ///
    /// This simplifies expressions such as identities (eg. `x + 0` becomes `x`).
    pub fn simplify(&self) -> SymElem {
        self.canonicalize().simplify_canonical()
    }

    /// Simplify an expression which is assumed to have been put in canonical
    /// form by [`canonicalize`](Self::canonicalize).
    fn simplify_canonical(&self) -> SymElem {
        match self {
            Self::Value(_) | Self::Var(_) => self.clone(),
            Self::Neg(expr) => match expr.simplify_canonical() {
                SymElem::Value(x) => SymElem::Value(-x),
                expr => Self::Neg(expr.into()),
            },
            Self::Add((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (SymElem::Value(0), rhs) => rhs,
                    (lhs, SymElem::Value(0)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x + y),
                    (lhs, SymElem::Neg(rhs)) if lhs == *rhs => SymElem::Value(0),
                    (lhs, rhs) => lhs + rhs,
                }
            }
            Self::Sub((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (lhs, SymElem::Value(0)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x - y),
                    (lhs, rhs) if lhs == rhs => SymElem::Value(0),
                    (lhs, rhs) => lhs - rhs,
                }
            }
            Self::Mul((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (SymElem::Value(1), rhs) => rhs,
                    (lhs, SymElem::Value(1)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x * y),
                    (lhs, rhs) => lhs * rhs,
                }
            }
            Self::Div((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (lhs, SymElem::Value(1)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) if y != 0 => SymElem::Value(x / y),
                    // x/x => 1
                    //
                    // Where we assume the RHS is non-zero.
                    //
                    // This is a special case of canceling common terms. The
                    // more general case (eg. XY / XZ => Y/Z) still needs to
                    // be implemented.
                    (lhs, rhs) if lhs == rhs => SymElem::Value(1),
                    (lhs, rhs) => lhs / rhs,
                }
            }
            Self::DivCeil((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (lhs, SymElem::Value(1)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) if y != 0 => {
                        SymElem::Value(div_ceil(x, y))
                    }
                    // x/x => 1
                    //
                    // Where we assume the RHS is non-zero.
                    //
                    // This is a special case of canceling common terms. The
                    // more general case (eg. XY / XZ => Y/Z) still needs to
                    // be implemented.
                    (lhs, rhs) if lhs == rhs => SymElem::Value(1),
                    (lhs, rhs) => lhs.div_ceil(&rhs),
                }
            }
            Self::Max((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                if lhs == rhs {
                    lhs
                } else {
                    match (lhs, rhs) {
                        (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x.max(y)),
                        (lhs, rhs) => Self::Max((lhs.into(), rhs.into())),
                    }
                }
            }
            Self::Min((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                if lhs == rhs {
                    lhs
                } else {
                    match (lhs, rhs) {
                        (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x.min(y)),
                        (lhs, rhs) => Self::Min((lhs.into(), rhs.into())),
                    }
                }
            }
            Self::Broadcast((lhs, rhs)) => {
                let lhs = lhs.simplify_canonical();
                let rhs = rhs.simplify_canonical();

                match (lhs, rhs) {
                    (SymElem::Value(x), SymElem::Value(y)) if x == y => SymElem::Value(x),
                    (SymElem::Value(1), y) => y,
                    (x, SymElem::Value(1)) => x,
                    (SymElem::Value(x), y) if x != 1 => SymElem::Value(x),
                    (x, SymElem::Value(y)) if y != 1 => SymElem::Value(y),
                    (lhs, rhs) if lhs == rhs => lhs,
                    (lhs, rhs) => SymElem::Broadcast((lhs.into(), rhs.into())),
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
            Self::Value(_) | Self::Var(_) | Self::Max(_) | Self::Min(_) | Self::Broadcast(_) => 4,
            Self::Div(_) | Self::DivCeil(_) => 3,
            Self::Mul(_) => 2,
            Self::Add(_) => 1,
            Self::Sub(_) | Self::Neg(_) => 0,
        }
    }

    /// Create a named symbol, with no assumptions about the value.
    pub fn var(name: &str) -> Self {
        SymElem::Var(
            Symbol {
                name: name.to_string(),
                positive: false,
            }
            .into(),
        )
    }

    /// Create a named symbol representing a positive value (ie. `>= 0`).
    pub fn pos_var(name: &str) -> Self {
        SymElem::Var(
            Symbol {
                name: name.to_string(),
                positive: true,
            }
            .into(),
        )
    }

    /// Compute `self / rhs` as an expression, or return `None` if an exact
    /// division is not possible.
    pub fn exact_div(&self, rhs: &SymElem) -> Option<SymElem> {
        let lhs = self;
        match (lhs, rhs) {
            // Fixed values
            (SymElem::Value(lhs), SymElem::Value(rhs)) => {
                if *rhs != 0 && lhs % rhs == 0 {
                    Some(SymElem::Value(lhs / rhs))
                } else {
                    None
                }
            }
            // Identities
            (lhs, rhs) if lhs == rhs => Some(SymElem::Value(1)),
            (lhs, SymElem::Value(1)) => Some(lhs.clone()),
            // If LHS is a product, recurse
            (SymElem::Mul((lhs_a, lhs_b)), rhs) => {
                if let Some(new_lhs_a) = lhs_a.exact_div(rhs) {
                    Some(SymElem::Mul((new_lhs_a.into(), lhs_b.clone())))
                } else {
                    lhs_b
                        .exact_div(rhs)
                        .map(|new_lhs_b| SymElem::Mul((lhs_a.clone(), new_lhs_b.into())))
                }
            }
            _ => None,
        }
    }

    /// Return the name of the symbol in a unary expression.
    ///
    /// Returns `None` if the expression is not unary or has a fixed value.
    fn name(&self) -> Option<&str> {
        match self {
            SymElem::Value(_) => None,
            SymElem::Var(sym) => Some(&sym.name),
            SymElem::Neg(x) => x.name(),
            SymElem::Add(_)
            | SymElem::Sub(_)
            | SymElem::Mul(_)
            | SymElem::Div(_)
            | SymElem::DivCeil(_)
            | SymElem::Max(_)
            | SymElem::Min(_)
            | SymElem::Broadcast(_) => None,
        }
    }

    /// Return true if `self` and `other` are negations of each other, meaning
    /// that adding the two terms together will produce zero.
    fn is_negation_of(&self, other: &SymElem) -> bool {
        match (self, other) {
            (x, SymElem::Neg(y)) if *x == **y => true,
            (SymElem::Neg(x), y) if **x == *y => true,
            _ => false,
        }
    }
}

/// Sort terms in an order that makes simplification easier, by making terms
/// which can be combined or eliminated adjacent.
fn cmp_values_first(a: &SymElem, b: &SymElem) -> Ordering {
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

impl PartialEq<SymElem> for SymElem {
    fn eq(&self, other: &SymElem) -> bool {
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
            Self::Add((a, b)) => match other {
                Self::Add((c, d)) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Mul((a, b)) => match other {
                Self::Mul((c, d)) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Max((a, b)) => match other {
                Self::Max((c, d)) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Min((a, b)) => match other {
                Self::Min((c, d)) => commutative_eq(a, b, c, d),
                _ => false,
            },
            Self::Sub((a, b)) => match other {
                Self::Sub((c, d)) => a == c && b == d,
                _ => false,
            },
            Self::Div((a, b)) => match other {
                Self::Div((c, d)) => a == c && b == d,
                _ => false,
            },
            Self::DivCeil((a, b)) => match other {
                Self::DivCeil((c, d)) => a == c && b == d,
                _ => false,
            },
            Self::Broadcast((a, b)) => match other {
                Self::Broadcast((c, d)) => commutative_eq(a, b, c, d),
                _ => false,
            },
        }
    }
}

impl Add<SymElem> for SymElem {
    type Output = SymElem;

    fn add(self, rhs: SymElem) -> Self {
        Self::Add((self.into(), rhs.into()))
    }
}

impl Sub<SymElem> for SymElem {
    type Output = SymElem;

    fn sub(self, rhs: SymElem) -> Self {
        Self::Sub((self.into(), rhs.into()))
    }
}

impl AddAssign<SymElem> for SymElem {
    fn add_assign(&mut self, rhs: SymElem) {
        *self = Self::Add((self.clone().into(), rhs.into()));
    }
}

impl Mul<SymElem> for SymElem {
    type Output = SymElem;

    fn mul(self, rhs: SymElem) -> Self {
        Self::Mul((self.into(), rhs.into()))
    }
}

impl Div<SymElem> for SymElem {
    type Output = SymElem;

    fn div(self, rhs: SymElem) -> Self {
        Self::Div((self.into(), rhs.into()))
    }
}

impl Neg for SymElem {
    type Output = SymElem;

    fn neg(self) -> Self {
        Self::Neg(self.into())
    }
}

impl From<Symbol> for SymElem {
    fn from(val: Symbol) -> Self {
        Self::Var(val.into())
    }
}

/// Create a symbol with a given name and an assumption that the value is
/// positive (`>= 0`).
///
/// The rationale for the positivity assumption is that during shape inference,
/// the most common use of symbols is to represent dimension sizes.
impl<'a> From<&'a str> for SymElem {
    fn from(name: &'a str) -> Self {
        SymElem::Var(
            Symbol {
                name: name.to_string(),
                positive: true,
            }
            .into(),
        )
    }
}

impl From<i32> for SymElem {
    fn from(val: i32) -> Self {
        SymElem::Value(val)
    }
}

impl fmt::Debug for SymElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let add_parens = |f: &mut fmt::Formatter<'_>, expr: &SymElem| {
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
            Self::Add((lhs, rhs)) => write_binop(f, '+', lhs, rhs),
            Self::Sub((lhs, rhs)) => write_binop(f, '-', lhs, rhs),
            Self::Mul((lhs, rhs)) => write_binop(f, '*', lhs, rhs),
            Self::Div((lhs, rhs)) => write_binop(f, '/', lhs, rhs),
            Self::DivCeil((lhs, rhs)) => write!(f, "ceil_div({:?}, {:?})", lhs, rhs),
            Self::Max((lhs, rhs)) => write!(f, "max({:?}, {:?})", lhs, rhs),
            Self::Min((lhs, rhs)) => write!(f, "min({:?}, {:?})", lhs, rhs),
            Self::Broadcast((lhs, rhs)) => write!(f, "broadcast({:?}, {:?})", lhs, rhs),
        }
    }
}

impl fmt::Display for SymElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let add_parens = |f: &mut fmt::Formatter<'_>, expr: &SymElem| {
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
            Self::Add((lhs, rhs)) => write_binop(f, '+', lhs, rhs),
            Self::Sub((lhs, rhs)) => write_binop(f, '-', lhs, rhs),
            Self::Mul((lhs, rhs)) => write_binop(f, '*', lhs, rhs),
            Self::Div((lhs, rhs)) => write_binop(f, '/', lhs, rhs),
            Self::DivCeil((lhs, rhs)) => write!(f, "ceil_div({}, {})", lhs, rhs),
            Self::Max((lhs, rhs)) => write!(f, "max({}, {})", lhs, rhs),
            Self::Min((lhs, rhs)) => write!(f, "min({}, {})", lhs, rhs),
            Self::Broadcast((lhs, rhs)) => write!(f, "broadcast({}, {})", lhs, rhs),
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

#[derive(Clone, Debug, PartialEq)]
enum SymTensorKind {
    Scalar(SymElem),
    Vector(Vec<SymElem>),
    Shape(Vec<SymElem>),
    Unknown {
        /// Note about why this Unknown value was created, for debugging purposes.
        note: &'static str,
    },
}

/// Tensor with symbolic shape and elements.
///
/// This is a tensor where the elements and dimension sizes can be either
/// concrete values or symbolic expressions. This type is used during shape
/// inference to represent the shapes of operator inputs and outputs, as well as
/// the values of operations which manipulate shapes.
///
/// The symbolic expressions can be integers, symbols with names and assumptions
/// about their values or composite expressions (addition, multiplication etc.)
///
/// ```
/// use rten_shape_inference::sym_tensor::{SymTensor, SymElem};
///
/// // Create a matrix with `nr` rows, `nc` columns and unknown values.
/// let nr = SymElem::from("nr");
/// let nc = SymElem::from("nc");
/// let matrix = SymTensor::from_shape(vec![nr.clone(), nc.clone()]);
/// assert_eq!(matrix.ndim(), Some(2));
/// assert_eq!(matrix.size(0), Some(nr.clone()));
/// assert_eq!(matrix.size(1), Some(nc.clone()));
///
/// // Turn the matrix's shape into a vector with values `["nr", "nc"]`.
/// let shape = SymTensor::from_vec(matrix.shape().unwrap().collect());
/// assert_eq!(shape.ndim(), Some(1));
/// assert_eq!(shape.values(), Some([nr.clone(), nc.clone()].as_slice()));
///
/// // Get the number of elements in the matrix as an expression `Some(nr * nc)`.
/// let len = shape.values().map(|v| v.iter().fold(
///     SymElem::Value(1),
///     |prod, dim| prod * dim.clone()
/// ).simplify());
/// assert_eq!(len, Some(SymElem::Mul((nr.into(), nc.into()))));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SymTensor(SymTensorKind);

impl SymTensor {
    /// Create a new symbolic tensor with unknown shape and values.
    ///
    /// `note` is a short string indicating the reason why the tensor shape
    /// and values are unknown. This is used for debugging purposes.
    pub fn unknown(note: &'static str) -> Self {
        Self(SymTensorKind::Unknown { note })
    }

    /// Create a new symbolic tensor with the given shape and unknown values.
    pub fn from_shape(shape: Vec<SymElem>) -> Self {
        Self(SymTensorKind::Shape(shape))
    }

    /// Create a new symbolic tensor with the given shape and unknown values.
    pub fn from_fixed_shape(shape: &[usize]) -> Self {
        Self(SymTensorKind::Shape(
            shape
                .iter()
                .copied()
                .map(|size| SymElem::Value(size as i32))
                .collect(),
        ))
    }

    /// Create a new symbolic vector.
    pub fn from_vec(vec: Vec<SymElem>) -> Self {
        Self(SymTensorKind::Vector(vec))
    }

    /// Create a new symbolic scalar.
    pub fn from_scalar(item: SymElem) -> Self {
        Self(SymTensorKind::Scalar(item))
    }

    /// Return this tensor's single element, if it is a scalar.
    pub fn as_scalar(&self) -> Option<&SymElem> {
        match &self.0 {
            SymTensorKind::Scalar(item) => Some(item),
            _ => None,
        }
    }

    /// Return this tensor's values as a slice, if it is a vector.
    pub fn as_vector(&self) -> Option<&[SymElem]> {
        match &self.0 {
            SymTensorKind::Vector(vec) => Some(vec),
            _ => None,
        }
    }

    /// Return this tensor's fixed values, if it is a scalar or a vector and
    /// all values are fixed.
    pub fn to_constant(&self) -> Option<Constant> {
        match &self.0 {
            SymTensorKind::Scalar(val) => match val {
                SymElem::Value(v) => Some(Constant::Scalar(*v)),
                _ => None,
            },
            SymTensorKind::Vector(vec) => {
                let values = vec
                    .iter()
                    .map(|v| match v {
                        SymElem::Value(v) => Some(*v),
                        _ => None,
                    })
                    .collect::<Option<Vec<i32>>>()?;
                Some(Constant::Vector(values))
            }
            SymTensorKind::Shape(_) | SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Return the number of dimensions, if known.
    pub fn ndim(&self) -> Option<usize> {
        match &self.0 {
            SymTensorKind::Scalar(_) => Some(0),
            SymTensorKind::Vector(_) => Some(1),
            SymTensorKind::Shape(val) => Some(val.len()),
            SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Return the size of the index'th dimension.
    ///
    /// Returns `None` if the index is out of bounds or the tensor's shape
    /// is unknown.
    pub fn size(&self, index: usize) -> Option<SymElem> {
        match &self.0 {
            SymTensorKind::Scalar(_) => None,
            SymTensorKind::Vector(val) => {
                if index == 0 {
                    Some(SymElem::Value(val.len() as i32))
                } else {
                    None
                }
            }
            SymTensorKind::Shape(val) => val.get(index).cloned(),
            SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Return an iterator over the dimensions or `None` if unknown.
    pub fn shape(&self) -> Option<impl ExactSizeIterator<Item = SymElem> + Clone> {
        let ndim = self.ndim()?;
        let dims = (0..ndim).map(|d| self.size(d).unwrap());
        Some(dims)
    }

    /// Return the symbolic values in this tensor, or `None` if unknown.
    pub fn values(&self) -> Option<&[SymElem]> {
        match &self.0 {
            SymTensorKind::Scalar(item) => Some(std::slice::from_ref(item)),
            SymTensorKind::Vector(val) => Some(val),
            SymTensorKind::Shape(_) | SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Simplify symbolic expressions in this tensor.
    ///
    /// See [`SymElem::simplify`].
    pub fn simplify(self) -> Self {
        match self.0 {
            SymTensorKind::Scalar(item) => Self::from_scalar(item.simplify()),
            SymTensorKind::Vector(vec) => {
                Self::from_vec(vec.into_iter().map(|x| x.simplify()).collect())
            }
            SymTensorKind::Shape(shape) => {
                Self::from_shape(shape.into_iter().map(|d| d.simplify()).collect())
            }
            _ => self,
        }
    }
}

#[cfg(test)]
pub(crate) use tests::{sym_elems, sym_shape, sym_vec};

#[cfg(test)]
mod tests {
    use super::{SymElem, SymTensor};

    /// Create a `Vec<SymElem>` from a list of symbol names and values.
    macro_rules! sym_elems {
        ($($x:expr),* $(,)?) => {
            vec![$(SymElem::from($x)),*]
        };
    }

    /// Create a symbolic vector from a list of symbol names and values.
    macro_rules! sym_vec {
        ($($x:expr),* $(,)?) => {
            SymTensor::from_vec(vec![$(SymElem::from($x)),*])
        };
    }

    /// Create a symbolic shape from a list of symbol names and values.
    macro_rules! sym_shape {
        ($($x:expr),* $(,)?) => {
            SymTensor::from_shape(vec![$(SymElem::from($x)),*])
        };
    }

    pub(crate) use {sym_elems, sym_shape, sym_vec};

    mod elem {
        use super::SymElem;

        #[test]
        fn test_range() {
            let x = SymElem::pos_var("x");
            assert_eq!(x.range(), (0, i32::MAX));

            let y = SymElem::var("y");
            assert_eq!(y.range(), (i32::MIN, i32::MAX));
        }

        #[test]
        fn test_simplify_add() {
            let x = SymElem::pos_var("x");
            let zero = SymElem::from(0);
            let one = SymElem::from(1);

            let expr = x.clone() + zero.clone();
            assert_eq!(expr, SymElem::Add((x.clone().into(), zero.clone().into())));
            assert_eq!(expr.simplify(), x);

            let expr_2 = x.clone() + one.clone();
            assert_eq!(
                expr_2.simplify(),
                SymElem::Add((x.clone().into(), one.clone().into()))
            );
        }

        // Check `C + X + D` is simplified to `S + X` where C and D are
        // constants and `S = C+D`.
        #[test]
        fn test_simplify_add_reassociate() {
            let x = SymElem::from("x");
            let c1 = SymElem::from(3);
            let c2 = SymElem::from(4);

            // C + X + D => S + X
            let expr = (x.clone() + c1.clone()) + c2.clone();
            let simplified = expr.simplify();
            assert_eq!(simplified, SymElem::from(7) + x.clone());

            // C + X + D + X => S + X + X
            let expr = (x.clone() + c1) + (x.clone() + c2);
            let simplified = expr.simplify();
            assert_eq!(simplified, SymElem::from(7) + x.clone() + x);
        }

        #[test]
        fn test_simplify_sub() {
            let x = SymElem::pos_var("x");
            let zero = SymElem::from(0);
            let one = SymElem::from(1);

            // x - 0 => x
            let expr = x.clone() - zero.clone();
            assert_eq!(expr, SymElem::Sub((x.clone().into(), zero.clone().into())));
            assert_eq!(expr.simplify(), x);

            // x - x => 0
            let expr = x.clone() - x.clone();
            assert_eq!(expr.simplify(), SymElem::Value(0));

            // x - 1 => x + (-1)
            let expr_2 = x.clone() - one.clone();
            assert_eq!(
                expr_2.simplify(),
                SymElem::Add((x.clone().into(), SymElem::from(-1).into()))
            );

            // x + y - x => y
            let y = SymElem::pos_var("y");
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
            let x = SymElem::pos_var("x");
            let one = SymElem::from(1);
            let two = SymElem::from(2);

            let expr = x.clone() * one.clone();
            assert_eq!(expr, SymElem::Mul((x.clone().into(), one.clone().into())));
            assert_eq!(expr.simplify(), x);

            let expr_2 = x.clone() * two.clone();
            assert_eq!(
                expr_2.simplify(),
                SymElem::Mul((x.clone().into(), two.clone().into()))
            );
        }

        #[test]
        fn test_simplify_div() {
            let x = SymElem::pos_var("x");
            let one = SymElem::from(1);
            let two = SymElem::from(2);

            // Constant eval
            let expr = SymElem::from(5) / SymElem::from(2);
            assert_eq!(expr.simplify(), SymElem::from(2));

            // Constant with zero divisor
            let expr = SymElem::from(5) / SymElem::from(0);
            assert_eq!(expr.simplify(), SymElem::from(5) / SymElem::from(0));

            // x / 1 => x
            let expr = x.clone() / one.clone();
            assert_eq!(expr, SymElem::Div((x.clone().into(), one.clone().into())));
            assert_eq!(expr.simplify(), x);

            // x / x => 1
            let expr = x.clone() / x.clone();
            assert_eq!(expr.simplify(), one);

            // x / 2 => x / 2
            let expr_2 = x.clone() / two.clone();
            assert_eq!(
                expr_2.simplify(),
                SymElem::Div((x.clone().into(), two.clone().into()))
            );
        }

        #[test]
        fn test_simplify_div_ceil() {
            let x = SymElem::pos_var("x");
            let one = SymElem::from(1);
            let two = SymElem::from(2);

            // Constant eval
            let expr = SymElem::from(5).div_ceil(&SymElem::from(2));
            assert_eq!(expr.simplify(), SymElem::from(3));

            // Constant with zero divisor
            let expr = SymElem::from(5).div_ceil(&SymElem::from(0));
            assert_eq!(
                expr.simplify(),
                SymElem::from(5).div_ceil(&SymElem::from(0))
            );

            // x / 1 => x
            let expr = x.clone().div_ceil(&one);
            assert_eq!(
                expr,
                SymElem::DivCeil((x.clone().into(), one.clone().into()))
            );
            assert_eq!(expr.simplify(), x);

            // x / x => 1
            let expr = x.clone().div_ceil(&x);
            assert_eq!(expr.simplify(), one);

            // x / 2 => x / 2
            let expr_2 = x.clone().div_ceil(&two);
            assert_eq!(
                expr_2.simplify(),
                SymElem::DivCeil((x.clone().into(), two.clone().into()))
            );
        }

        // Check `C * X * D` is simplified to `CD * X` where C and D are
        // constants.
        #[test]
        fn test_simplify_mul_reassociate() {
            let x = SymElem::from("x");
            let c1 = SymElem::from(3);
            let c2 = SymElem::from(4);

            // C * X * D => CD * X
            let expr = (x.clone() * c1.clone()) * c2.clone();
            let simplified = expr.simplify();
            assert_eq!(simplified, SymElem::from(12) * x.clone());

            // Same as above, but contained inside an addition expression.
            let expr = SymElem::from(5) + expr;
            let simplified = expr.simplify();
            assert_eq!(simplified, SymElem::from(5) + SymElem::from(12) * x.clone());

            // C * X * D * X => CD * X * X
            let expr = (x.clone() * c1) * (x.clone() * c2);
            let simplified = expr.simplify();
            assert_eq!(simplified, SymElem::from(12) * x.clone() * x);
        }

        #[test]
        fn test_simplify_max() {
            let one = SymElem::from(1);
            let two = SymElem::from(2);
            let expr = one.max(&two);

            assert_eq!(expr, SymElem::Max((one.clone().into(), two.clone().into())));
            assert_eq!(expr.simplify(), two.clone());
        }

        #[test]
        fn test_simplify_nested_max() {
            let expr = SymElem::from(10)
                .max(&SymElem::from(5).max(&SymElem::from(11)))
                .simplify();
            assert_eq!(expr, SymElem::from(11));
        }

        #[test]
        fn test_simplify_min() {
            let one = SymElem::from(1);
            let two = SymElem::from(2);
            let expr = one.min(&two);

            assert_eq!(expr, SymElem::Min((one.clone().into(), two.clone().into())));
            assert_eq!(expr.simplify(), one.clone());
        }

        #[test]
        fn test_simplify_nested_min() {
            let expr = SymElem::from(10)
                .min(&SymElem::from(5).min(&SymElem::from(3)))
                .simplify();
            assert_eq!(expr, SymElem::from(3));
        }

        #[test]
        fn test_simplify_broadcast() {
            let one = SymElem::from(1);
            let ten = SymElem::from(10);
            let foo = SymElem::from("foo");

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
            let foo = SymElem::from("foo");
            let ten = SymElem::from(10);
            let expr = foo.broadcast(&foo.broadcast(&ten)).simplify();
            assert_eq!(expr, SymElem::from(10));
        }

        #[test]
        fn test_simplify_neg() {
            let minus_one = -SymElem::from(1);
            assert_eq!(minus_one.simplify(), SymElem::from(-1));
        }

        #[test]
        fn test_display() {
            let expr = (SymElem::from(1) + SymElem::pos_var("foo")) * SymElem::from(3)
                + SymElem::from(4)
                - SymElem::from(5);
            assert_eq!(expr.to_string(), "(1 + foo) * 3 + 4 - 5");
        }

        #[test]
        fn test_debug() {
            let expr = (SymElem::from(1) + SymElem::pos_var("foo")) * SymElem::from(3)
                + SymElem::var("bar")
                - SymElem::from(5);
            assert_eq!(format!("{:?}", expr), "(1 + \"foo\"u) * 3 + \"bar\"i - 5");
        }

        #[test]
        fn test_exact_div() {
            // Fixed values
            assert_eq!(
                SymElem::from(15).exact_div(&SymElem::from(3)),
                Some(SymElem::from(5))
            );
            assert_eq!(SymElem::from(15).exact_div(&SymElem::from(4)), None);
            assert_eq!(SymElem::from(15).exact_div(&SymElem::from(0)), None);

            // Identities
            assert_eq!(
                SymElem::from("x").exact_div(&SymElem::from("x")),
                Some(SymElem::from(1))
            );
            assert_eq!(
                SymElem::from("x").exact_div(&SymElem::from(1)),
                Some(SymElem::from("x"))
            );

            // Products with common term in LHS and RHS
            assert_eq!(
                (SymElem::from("x") * SymElem::from("y"))
                    .exact_div(&SymElem::from("y"))
                    .map(|s| s.simplify()),
                Some(SymElem::from("x"))
            );
            assert_eq!(
                (SymElem::from("y") * SymElem::from("x"))
                    .exact_div(&SymElem::from("y"))
                    .map(|s| s.simplify()),
                Some(SymElem::from("x"))
            );

            // Cases where result is unknown
            assert_eq!(SymElem::from("x").exact_div(&SymElem::from("y")), None);
        }
    }

    mod tensor {
        use super::{SymElem, SymTensor};

        #[test]
        fn test_scalar() {
            let x = SymTensor::from_scalar("x".into());
            assert_eq!(x.ndim(), Some(0));
            assert_eq!(x.size(0), None);
            assert_eq!(x.values(), Some(["x".into()].as_slice()));
        }

        #[test]
        fn test_vector() {
            let x = SymTensor::from_vec(vec!["x".into(), 2.into()]);
            assert_eq!(x.ndim(), Some(1));
            assert_eq!(x.size(0), Some(2.into()));
            assert_eq!(x.size(1), None);
            assert_eq!(x.values(), Some(["x".into(), 2.into()].as_slice()));
        }

        #[test]
        fn test_tensor_with_shape() {
            let x = SymTensor::from_shape(vec!["x".into(), 2.into()]);
            assert_eq!(x.ndim(), Some(2));
            assert_eq!(x.size(0), Some("x".into()));
            assert_eq!(x.size(1), Some(2.into()));
            assert_eq!(x.size(2), None);
            assert_eq!(x.values(), None);
            assert_eq!(
                x.shape().unwrap().collect::<Vec<_>>(),
                vec!["x".into(), 2.into()]
            );
        }
        #[test]
        fn test_simplify() {
            // Simplify a shape
            let matrix = SymTensor::from_shape(vec![
                SymElem::pos_var("rows") + SymElem::from(0),
                SymElem::pos_var("cols") * SymElem::from(1),
            ])
            .simplify();
            assert_eq!(
                matrix.shape().unwrap().collect::<Vec<_>>(),
                vec!["rows".into(), "cols".into(),]
            );

            // Simplify a scalar
            let x = SymElem::var("x");
            let add_expr = x.clone() + SymElem::from(0);
            let scalar = SymTensor::from_scalar(add_expr.clone()).simplify();
            assert_eq!(scalar.as_scalar().unwrap(), &x);

            // Simplify a vector
            let vec = SymTensor::from_vec(vec![add_expr.clone(), add_expr.clone()]).simplify();
            assert_eq!(vec.as_vector().unwrap(), [x.clone(), x.clone()]);
        }

        #[test]
        fn test_unknown_shape() {
            let x = SymTensor::unknown("missing input shape");
            assert!(x.shape().is_none());
            assert_eq!(x.values(), None);
        }
    }
}

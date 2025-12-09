//! Tensors with symbolic shapes and values.

use std::fmt;
use std::ops::{Add, AddAssign, Mul};
use std::rc::Rc;

/// Vector or scalar with integer values.
#[derive(Clone, PartialEq)]
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
    /// Multiplication of two symbolic values
    Mul((Rc<SymElem>, Rc<SymElem>)),
    /// Maximum of two symbolic values
    Max((Rc<SymElem>, Rc<SymElem>)),
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
            Self::Add((lhs, rhs)) | Self::Mul((lhs, rhs)) | Self::Max((lhs, rhs)) => {
                let (lhs_min, lhs_max) = lhs.range();
                let (rhs_min, rhs_max) = rhs.range();
                (lhs_min.min(rhs_min), lhs_max.max(rhs_max))
            }
        }
    }

    /// Return true if the value of this expression is known to be >= 0.
    pub fn is_positive(&self) -> bool {
        match self {
            Self::Value(x) => *x >= 0,
            Self::Var(sym) => sym.positive,
            Self::Add((lhs, rhs)) => lhs.is_positive() && rhs.is_positive(),
            Self::Mul((lhs, rhs)) => lhs.is_positive() && rhs.is_positive(),
            Self::Max((lhs, rhs)) => lhs.is_positive() || rhs.is_positive(),
        }
    }

    /// Return the maximum of `self` and `other`.
    pub fn max(&self, other: &SymElem) -> SymElem {
        Self::Max((self.clone().into(), other.clone().into()))
    }

    /// Simplify an expression.
    ///
    /// This simplifies expressions such as identities (eg. `x + 0` becomes `x`).
    pub fn simplify(&self) -> SymElem {
        match self {
            Self::Value(_) | Self::Var(_) => self.clone(),
            Self::Add((lhs, rhs)) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();

                match (lhs, rhs) {
                    (SymElem::Value(0), rhs) => rhs,
                    (lhs, SymElem::Value(0)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x + y),
                    (lhs, rhs) => lhs + rhs,
                }
            }
            Self::Mul((lhs, rhs)) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();

                match (lhs, rhs) {
                    (SymElem::Value(1), rhs) => rhs,
                    (lhs, SymElem::Value(1)) => lhs,
                    (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x * y),
                    (lhs, rhs) => lhs * rhs,
                }
            }
            Self::Max((lhs, rhs)) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();

                if lhs == rhs {
                    lhs.clone()
                } else {
                    match (lhs, rhs) {
                        (SymElem::Value(x), SymElem::Value(y)) => SymElem::Value(x.max(y)),
                        (lhs, rhs) => Self::Max((lhs.into(), rhs.into())),
                    }
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
            Self::Value(_) | Self::Var(_) | Self::Max(_) => 2,
            Self::Mul(_) => 1,
            Self::Add(_) => 0,
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
}

impl PartialEq<SymElem> for SymElem {
    fn eq(&self, other: &SymElem) -> bool {
        let commutative_eq = |self_lhs, self_rhs, other_lhs, other_rhs| {
            (self_lhs == other_lhs && self_rhs == other_rhs)
                || (self_lhs == other_rhs && self_rhs == other_lhs)
        };

        // Symbols are equal if they have the same value or the same name.
        match (self, other) {
            (Self::Value(x), Self::Value(y)) => x == y,
            (Self::Var(x), Self::Var(y)) => x.name == y.name,
            (Self::Add((a, b)), Self::Add((c, d))) => commutative_eq(a, b, c, d),
            (Self::Mul((a, b)), Self::Mul((c, d))) => commutative_eq(a, b, c, d),
            (Self::Max((a, b)), Self::Max((c, d))) => commutative_eq(a, b, c, d),
            (_, _) => false,
        }
    }
}

impl Add<SymElem> for SymElem {
    type Output = SymElem;

    fn add(self, rhs: SymElem) -> Self {
        Self::Add((self.into(), rhs.into()))
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
            Self::Add((lhs, rhs)) => write_binop(f, '+', lhs, rhs),
            Self::Mul((lhs, rhs)) => write_binop(f, '*', lhs, rhs),
            Self::Max((lhs, rhs)) => write!(f, "max({:?}, {:?})", lhs, rhs),
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
            Self::Add((lhs, rhs)) => write_binop(f, '+', lhs, rhs),
            Self::Mul((lhs, rhs)) => write_binop(f, '*', lhs, rhs),
            Self::Max((lhs, rhs)) => write!(f, "max({}, {})", lhs, rhs),
        }
    }
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
        fn test_simplify_max() {
            let one = SymElem::from(1);
            let two = SymElem::from(2);
            let expr = one.max(&two);

            assert_eq!(expr, SymElem::Max((one.clone().into(), two.clone().into())));
            assert_eq!(expr.simplify(), two.clone());
        }

        #[test]
        fn test_display() {
            let expr =
                (SymElem::from(1) + SymElem::pos_var("foo")) * SymElem::from(3) + SymElem::from(4);
            assert_eq!(expr.to_string(), "(1 + foo) * 3 + 4");
        }

        #[test]
        fn test_debug() {
            let expr = (SymElem::from(1) + SymElem::pos_var("foo")) * SymElem::from(3)
                + SymElem::var("bar");
            assert_eq!(format!("{:?}", expr), "(1 + \"foo\"u) * 3 + \"bar\"i");
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

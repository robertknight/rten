//! Tensors with symbolic shapes and values.

use std::fmt;

use crate::sym_expr::SymExpr;

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

#[derive(Clone, Debug, PartialEq)]
enum SymTensorKind {
    Scalar(SymExpr),
    Vector(Vec<SymExpr>),
    Shape(Vec<SymExpr>),
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
/// use rten_shape_inference::{SymTensor, SymExpr};
///
/// // Create a matrix with `nr` rows, `nc` columns and unknown values.
/// let nr = SymExpr::from("nr");
/// let nc = SymExpr::from("nc");
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
///     SymExpr::Value(1),
///     |prod, dim| prod * dim.clone()
/// ).simplify());
/// assert_eq!(len, Some(SymExpr::Mul((nr.into(), nc.into()))));
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
    pub fn from_shape(shape: Vec<SymExpr>) -> Self {
        Self(SymTensorKind::Shape(shape))
    }

    /// Create a new symbolic tensor with the given shape and unknown values.
    pub fn from_fixed_shape(shape: &[usize]) -> Self {
        Self(SymTensorKind::Shape(
            shape
                .iter()
                .copied()
                .map(|size| SymExpr::Value(size as i32))
                .collect(),
        ))
    }

    /// Create a new symbolic vector.
    pub fn from_vec(vec: Vec<SymExpr>) -> Self {
        Self(SymTensorKind::Vector(vec))
    }

    /// Create a new symbolic scalar.
    pub fn from_scalar(item: SymExpr) -> Self {
        Self(SymTensorKind::Scalar(item))
    }

    /// Return this tensor's single element, if it is a scalar.
    pub fn as_scalar(&self) -> Option<&SymExpr> {
        match &self.0 {
            SymTensorKind::Scalar(item) => Some(item),
            _ => None,
        }
    }

    /// Return this tensor's values as a slice, if it is a vector.
    pub fn as_vector(&self) -> Option<&[SymExpr]> {
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
                SymExpr::Value(v) => Some(Constant::Scalar(*v)),
                _ => None,
            },
            SymTensorKind::Vector(vec) => {
                let values = vec
                    .iter()
                    .map(|v| match v {
                        SymExpr::Value(v) => Some(*v),
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
    pub fn size(&self, index: usize) -> Option<SymExpr> {
        match &self.0 {
            SymTensorKind::Scalar(_) => None,
            SymTensorKind::Vector(val) => {
                if index == 0 {
                    Some(SymExpr::Value(val.len() as i32))
                } else {
                    None
                }
            }
            SymTensorKind::Shape(val) => val.get(index).cloned(),
            SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Return an iterator over the dimensions or `None` if unknown.
    pub fn shape(&self) -> Option<impl ExactSizeIterator<Item = SymExpr> + Clone> {
        let ndim = self.ndim()?;
        let dims = (0..ndim).map(|d| self.size(d).unwrap());
        Some(dims)
    }

    /// Return the symbolic values in this tensor, or `None` if unknown.
    pub fn values(&self) -> Option<&[SymExpr]> {
        match &self.0 {
            SymTensorKind::Scalar(item) => Some(std::slice::from_ref(item)),
            SymTensorKind::Vector(val) => Some(val),
            SymTensorKind::Shape(_) | SymTensorKind::Unknown { .. } => None,
        }
    }

    /// Simplify symbolic expressions in this tensor.
    ///
    /// See [`SymExpr::simplify`].
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
    use super::{SymExpr, SymTensor};

    /// Create a `Vec<SymExpr>` from a list of symbol names and values.
    macro_rules! sym_elems {
        ($($x:expr),* $(,)?) => {
            vec![$(SymExpr::from($x)),*]
        };
    }

    /// Create a symbolic vector from a list of symbol names and values.
    macro_rules! sym_vec {
        ($($x:expr),* $(,)?) => {
            SymTensor::from_vec(vec![$(SymExpr::from($x)),*])
        };
    }

    /// Create a symbolic shape from a list of symbol names and values.
    macro_rules! sym_shape {
        ($($x:expr),* $(,)?) => {
            SymTensor::from_shape(vec![$(SymExpr::from($x)),*])
        };
    }

    pub(crate) use {sym_elems, sym_shape, sym_vec};

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
            SymExpr::pos_var("rows") + SymExpr::from(0),
            SymExpr::pos_var("cols") * SymExpr::from(1),
        ])
        .simplify();
        assert_eq!(
            matrix.shape().unwrap().collect::<Vec<_>>(),
            vec!["rows".into(), "cols".into(),]
        );

        // Simplify a scalar
        let x = SymExpr::var("x");
        let add_expr = x.clone() + SymExpr::from(0);
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

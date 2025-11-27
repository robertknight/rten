//! Shape inference types and traits.
//!
//! This module also provides implementations of shape inference that are
//! reused by many operators, such as unary and binary ops.

use rten_tensor::Layout;
use smallvec::SmallVec;

use crate::graph::Dimension;
use crate::ops::resolve_axes;
use crate::value::ValueView;

#[derive(Clone, Debug, PartialEq)]
pub enum ShapeInferenceError {
    /// Too many or too few inputs were provided for this operator.
    IncorrectInputCount,

    /// The input shapes are incompatible.
    ///
    /// Operator execution will fail if given inputs with these shapes.
    IncompatibleShapes,

    /// An input's rank does not match that expected by the operator.
    IncorrectRank,
}

/// Inferred size of a single output dimension.
#[derive(Clone, Debug, PartialEq)]
pub enum InferredDimension {
    Fixed(usize),
    Symbolic(String),
    /// The dimension has an unknown size.
    Unknown,
}

impl From<usize> for InferredDimension {
    fn from(size: usize) -> Self {
        InferredDimension::Fixed(size)
    }
}

impl From<String> for InferredDimension {
    fn from(name: String) -> Self {
        InferredDimension::Symbolic(name)
    }
}

impl From<&str> for InferredDimension {
    fn from(name: &str) -> Self {
        InferredDimension::Symbolic(name.to_string())
    }
}

impl From<Dimension> for InferredDimension {
    fn from(val: Dimension) -> Self {
        match val {
            Dimension::Fixed(size) => Self::Fixed(size),
            Dimension::Symbolic(name) => Self::Symbolic(name),
        }
    }
}

impl PartialEq<Dimension> for InferredDimension {
    fn eq(&self, other: &Dimension) -> bool {
        match (self, other) {
            (Self::Fixed(a), Dimension::Fixed(b)) if a == b => true,
            (Self::Symbolic(a), Dimension::Symbolic(b)) if a == b => true,
            _ => false,
        }
    }
}

/// An input to shape inference.
///
/// The input has a shape, which can be concrete (`(1, 2, 3)`) or symbolic
/// (`("batch", 2, 3)`). The input may have a known value if it is a graph
/// constant, or it may be a symbolic value (such as produced by a `Shape`
/// operator given an input with a symbolic shape).
///
/// When invoking shape inference, the most concrete value that is available
/// should be provided (ie. `Constant`, then `SymValue`, then `Value`).
#[derive(Clone, Debug)]
pub enum Input<'a> {
    /// An input with a known shape and value.
    Constant(ValueView<'a>),
    /// An input with known shape and symbolic value.
    SymValue(Vec<Dimension>),
    /// An input with a symbolic shape and unknown value.
    Value(Vec<Dimension>),
}

impl Input<'_> {
    fn ndim(&self) -> usize {
        match self {
            Self::Constant(val) => val.ndim(),
            Self::SymValue(_) => 1,
            Self::Value(val) => val.len(),
        }
    }

    fn dim(&self, index: usize) -> Dimension {
        match self {
            Self::Constant(val) => Dimension::Fixed(val.shape()[index]),
            Self::SymValue(val) => {
                assert_eq!(index, 0);
                Dimension::Fixed(val.len())
            }
            Self::Value(val) => val[index].clone(),
        }
    }

    fn dims(&self) -> impl Iterator<Item = Dimension> {
        (0..self.ndim()).map(|d| self.dim(d))
    }
}

// trait GetNodeValue {
//     fn get_node_value(&self, id: NodeId) -> Option<ValueView<'_>>;
// }

/// Information about an operator's inputs.
///
/// This includes the known shape metadata and, optionally the concrete value.
// pub struct InputInfo<'graph, 'dims> {
//     dims: &'dims [&'dims [Dimension]],
//     value_provider: Option<&'graph dyn GetNodeValue>,
//     values: Vec<(usize, NodeId)>,
// }

// impl<'graph, 'dims> InputInfo<'graph, 'dims> {
//     fn from_dims(dims: &'dims [&'dims [Dimension]]) -> Self {
//         Self {
//             dims,
//             value_provider: None,
//             values: Vec::new(),
//         }
//     }

//     fn ndim(&self) -> usize {
//         self.dims.len()
//     }

//     fn dims(&self) -> &[&[Dimension]] {
//         self.dims
//     }

//     fn value(&self, dim: usize) -> Option<ValueView<'_>> {
//         let node_id = self
//             .values
//             .iter()
//             .find(|(idx, _node)| *idx == dim)
//             .map(|(_, id)| *id)?;
//         self.value_provider
//             .and_then(|vp| vp.get_node_value(node_id))
//     }
// }

/// Infer the shapes of an operator's outputs given its inputs.
pub trait InferShapes {
    fn infer_shapes(
        &self,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<InferredDimension>>, ShapeInferenceError>;
}

// TBD - How will value inference work?
//
// TBD - What about type inference? For unary and binary ops the shape
// inference implementation is the same for each, but the output type can
// differ. From examining ops so far, looks like I could start with a general
// inference that assumes input and output types are the same, then modify the
// results. Currently only the CastElimination op checks the dtype. Other fusions
// probably should.
//
// TBD - What about partial inference (type only, rank only etc.)
// - Try to avoid it to start with and see how far I can get.
//
// Design question: Should shape and type inference be one method or separate?
// Eventually we'll want to combine the information but maybe it will be easier
// to separate them?

/// Shared shape inference implementation for unary operators.
pub const UNARY_OP: UnaryOpInfer = UnaryOpInfer;

/// Shape inference implementation for unary operators.
///
/// These operators take a single input and return an output with the same
/// shape.
pub struct UnaryOpInfer;

impl InferShapes for UnaryOpInfer {
    fn infer_shapes(
        &self,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<InferredDimension>>, ShapeInferenceError> {
        let [input] = &inputs[..] else {
            return Err(ShapeInferenceError::IncorrectInputCount);
        };
        let out_dims = input.dims().map(InferredDimension::from).collect();
        Ok(vec![out_dims])
    }
}

/// Shared shape inference implementation for binary operators which broadcast
/// their inputs together.
pub const BINARY_OP: BinaryOpInfer = BinaryOpInfer;

/// Shape inference implementation for binary operators.
///
/// These operators take two inputs and return an output whose shape is the
/// result of broadcasting the two input shapes together.
pub struct BinaryOpInfer;

impl InferShapes for BinaryOpInfer {
    fn infer_shapes(
        &self,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<InferredDimension>>, ShapeInferenceError> {
        let [a, b] = &inputs[..] else {
            return Err(ShapeInferenceError::IncorrectInputCount);
        };

        let a_pad = b.ndim().saturating_sub(a.ndim());
        let b_pad = a.ndim().saturating_sub(b.ndim());

        let a_iter = std::iter::repeat(Dimension::Fixed(1))
            .take(a_pad)
            .chain(a.dims());
        let b_iter = std::iter::repeat(Dimension::Fixed(1))
            .take(b_pad)
            .chain(b.dims());

        let mut out_shape = Vec::with_capacity(a.ndim().max(b.ndim()));
        for (a, b) in a_iter.zip(b_iter) {
            let dim: InferredDimension = match (a, b) {
                (a, b) if a == b => a.clone().into(),

                // If either size is 1, it will be broadcast against the other
                // size.
                (Dimension::Fixed(1), b) => b.clone().into(),
                (a, Dimension::Fixed(1)) => a.clone().into(),

                // If both sizes are fixed and different, we know execution
                // will fail.
                (Dimension::Fixed(_), Dimension::Fixed(_)) => {
                    return Err(ShapeInferenceError::IncompatibleShapes);
                }

                // If one dim is a fixed value other than 1 and the other
                // dim is symbolic, execution can only succeed if the symbolic
                // dim has the same size as the fixed dim.
                (Dimension::Symbolic(_a), Dimension::Fixed(b)) => InferredDimension::Fixed(b),
                (Dimension::Fixed(a), Dimension::Symbolic(_b)) => InferredDimension::Fixed(a),

                // If both dimensions are symbolic, the result can be either of
                // the dimensions.
                //
                // 1. If both sizes are equal, the result can be seen as either
                //    the first or second dim.
                // 2. If only one of the sizes is 1, the result will be the other
                //    dim.
                // 3. If the sizes are different, the op will fail.
                (Dimension::Symbolic(_), Dimension::Symbolic(_)) => InferredDimension::Unknown,
            };
            out_shape.push(dim);
        }

        Ok([out_shape].into())
    }
}

/// Shape inference for ONNX `Reduce*` operators.
#[derive(Clone, Debug, PartialEq)]
pub struct ReductionOpInfer<'a> {
    /// Axes over which the reduction is applied.
    ///
    /// Reduction ops take the axes as an attribute in ONNX opset <= 13 and an
    /// input in opset 18+.
    pub axes: Option<&'a [i32]>,

    /// True if the reduced dimension is retained as a 1-sized dimension in the
    /// output.
    pub keep_dims: bool,

    pub noop_with_empty_axes: bool,
}

impl InferShapes for ReductionOpInfer<'_> {
    fn infer_shapes(
        &self,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<InferredDimension>>, ShapeInferenceError> {
        match inputs.len() {
            1 | 2 => {}
            _ => {
                return Err(ShapeInferenceError::IncorrectInputCount);
            }
        }

        let data = &inputs[0];
        let mut axes: SmallVec<[usize; 4]> =
            if let Some(Input::Constant(ValueView::Int32Tensor(axes))) = inputs.get(2) {
                resolve_axes(data.ndim(), axes.iter())
                    .map_err(|_| ShapeInferenceError::IncorrectRank)?
            } else if let Some(axes) = self.axes {
                resolve_axes(data.ndim(), axes.iter())
                    .map_err(|_| ShapeInferenceError::IncorrectRank)?
            } else {
                (0..data.ndim()).collect()
            };
        axes.sort();
        axes.dedup();

        let out_ndim = if self.keep_dims {
            data.ndim()
        } else {
            data.ndim() - axes.len()
        };
        let mut out_shape = Vec::with_capacity(out_ndim);

        for (i, dim) in data.dims().enumerate() {
            if !axes.contains(&i) {
                out_shape.push(dim.clone().into());
                continue;
            } else if self.keep_dims {
                out_shape.push(InferredDimension::Fixed(1));
            }
        }

        Ok([out_shape].into())
    }
}

// Other:
//
//  - Concat
//  - Expand
//  - Flatten
//      - Output is 2D, where each dim is a product of dims 0..axis or
//        axis..ndim of the input
//  - Gather
//  - MatMul
//  - MatMulNBits
//      - Output
//  - Range: Output depends upon input values
//  - Reshape:
//      - Output always depends upon `shape` input
//      - Output may depend upon `data` input
//  - Shape
//      - Output is a vector of length equal to input shape rank
//      - Output of inference includes both the shape and the values. If the
//        values are all fixed, we could just replace the Shape operator with
//        a constant.
//  - Slice: Output shape depends upon inputs
//  - Transpose: Re-orders input according to `perm` attr
//  - Unsqueeze: Inserts a single size-1 axis
//  - Where: Broadcasts three input shapes together

// TODO: Need to design what the API to drive shape inference will be like.
// Something like:

// use crate::graph::NodeId;
// use crate::value::ValueMeta;
// use std::collections::HashMap;

// pub fn infer_shapes(graph: &Graph) -> HashMap<NodeId, Result<ValueMeta, ShapeInferenceError>> {
//     // Start with an initial set of nodes that have known shapes and types
//     // (inputs, constants) and iteratively expand the set.
//     todo!()
// }

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{
        BINARY_OP, InferShapes, InferredDimension, Input, ReductionOpInfer, ShapeInferenceError,
        UNARY_OP,
    };
    use crate::Dimension;

    macro_rules! dims {
        ($($x:expr),* $(,)?) => {
            vec![$(crate::Dimension::from($x)),*]
        };
    }

    macro_rules! inferred_dims {
        ($($x:expr),* $(,)?) => {
            vec![$(super::InferredDimension::from($x)),*]
        };
    }

    fn inputs(dims: impl IntoIterator<Item = Vec<Dimension>>) -> Vec<Input<'static>> {
        dims.into_iter().map(Input::Value).collect()
    }

    #[test]
    fn test_unary_op_infer() {
        let input = dims!("batch", 16, "seq", 24);
        let shape = UNARY_OP.infer_shapes(inputs([input.clone()])).unwrap();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0], input);

        let err = UNARY_OP
            .infer_shapes(inputs([input.clone(), input]))
            .err()
            .unwrap();
        assert_eq!(err, ShapeInferenceError::IncorrectInputCount);
    }

    #[test]
    fn test_binary_op_infer() {
        #[derive(Debug)]
        struct Case {
            lhs: Vec<Dimension>,
            rhs: Vec<Dimension>,
            expected: Vec<InferredDimension>,
        }

        let cases = [
            Case {
                lhs: dims!("batch"),
                rhs: dims!("batch"),
                expected: inferred_dims!("batch"),
            },
            Case {
                lhs: dims!(2, 3),
                rhs: dims!(2, 3),
                expected: inferred_dims!(2, 3),
            },
            Case {
                lhs: dims!(1, 5),
                rhs: dims!(4, 1),
                expected: inferred_dims!(4, 5),
            },
            Case {
                lhs: dims!(1, 1),
                rhs: dims!(1, 1),
                expected: inferred_dims!(1, 1),
            },
            Case {
                lhs: dims!(1, "bar"),
                rhs: dims!("foo", 1),
                expected: inferred_dims!("foo", "bar"),
            },
            Case {
                lhs: dims!("foo"),
                rhs: dims!("bar"),
                expected: inferred_dims!(InferredDimension::Unknown),
            },
        ];

        cases.test_each(|case| {
            let shape = BINARY_OP
                .infer_shapes(inputs([case.lhs.clone(), case.rhs.clone()]))
                .unwrap();
            assert_eq!(shape.len(), 1);
            assert_eq!(shape[0], case.expected);
        });
    }

    #[test]
    fn test_binary_op_infer_invalid() {
        #[derive(Clone, Debug)]
        struct Case {
            inputs: Vec<Vec<Dimension>>,
            expected: ShapeInferenceError,
        }

        let cases = [
            Case {
                inputs: [dims!(5)].into(),
                expected: ShapeInferenceError::IncorrectInputCount,
            },
            Case {
                inputs: [dims!(5), dims!(3)].into(),
                expected: ShapeInferenceError::IncompatibleShapes,
            },
        ];

        cases.test_each_clone(|case| {
            let inputs = case.inputs.into_iter().map(Input::Value).collect();
            let err = BINARY_OP.infer_shapes(inputs).err().unwrap();
            assert_eq!(err, case.expected);
        });
    }

    #[test]
    fn test_reduction_op_infer() {
        #[derive(Clone, Debug)]
        struct Case<'a> {
            inputs: Vec<Input<'a>>,
            op: ReductionOpInfer<'a>,
            expected: Vec<Dimension>,
        }

        use rten_tensor::TensorView;

        let axes = TensorView::from(&[1i32]);

        let cases = [Case {
            inputs: [
                Input::Value(dims!(3, 4, 5)),
                Input::Constant(axes.clone().into()),
            ]
            .into(),
            op: ReductionOpInfer {
                axes: None,
                keep_dims: false,
                noop_with_empty_axes: false,
            },
            expected: dims!(3, 5),
        }];

        cases.test_each(|case| {
            let shapes = case.op.infer_shapes(case.inputs.clone()).unwrap();
            assert_eq!(shapes.len(), 1);
            assert_eq!(shapes[0], case.expected);
        });
    }
}

//! Shape inference types and traits.
//!
//! This module also provides implementations of shape and type inference that
//! are commonly used by many operators.

use std::collections::{HashMap, HashSet};

use crate::env::env_flag;
use crate::graph::{Dimension, Graph, Node, NodeId, RunError, TypedConstant};
use crate::value::DataType;

pub use rten_shape_inference::{
    infer_shapes::{BinaryOp, InferShapes, InferShapesError, ReductionOp, UnaryOp},
    sym_gen::SymbolGen,
    sym_tensor::{Constant, SymElem, SymTensor, Symbol},
};

/// Impl [`InferShapes`] for a type by delegating to another type which
/// implements the trait.
///
/// This is used by operators in this crate to delegate shape inference to types
/// defined in the rten_shape_inference crate.
macro_rules! impl_infer_shapes {
    ($op:ident, $make_impl:expr) => {
        impl rten_shape_inference::infer_shapes::InferShapes for $op {
            fn infer_shapes(
                &self,
                inputs: &[rten_shape_inference::sym_tensor::SymTensor],
                sym_gen: &mut rten_shape_inference::sym_gen::SymbolGen,
            ) -> Result<
                Vec<rten_shape_inference::sym_tensor::SymTensor>,
                rten_shape_inference::infer_shapes::InferShapesError,
            > {
                $make_impl(self).infer_shapes(inputs, sym_gen)
            }
        }
    };
}
pub(crate) use impl_infer_shapes;

/// Infer the shapes of an operator's outputs given its inputs.
pub trait InferTypes {
    fn infer_types(
        &self,
        inputs: &[Option<DataType>],
    ) -> Result<Vec<Option<DataType>>, InferShapesError>;
}

/// A type inference implementation for operators which return a single output
/// whose type is the same as one of the inputs, usually the first.
pub struct SameAsInput {
    pub index: usize,
}

impl InferTypes for SameAsInput {
    fn infer_types(
        &self,
        inputs: &[Option<DataType>],
    ) -> Result<Vec<Option<DataType>>, InferShapesError> {
        let Some(input) = inputs.get(self.index) else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        Ok(vec![*input])
    }
}

/// Type inference for operators which return one output, with the same type as
/// the first input.
pub const SAME_AS_FIRST_INPUT: SameAsInput = SameAsInput { index: 0 };

pub struct FixedTypes<const N: usize> {
    pub types: [DataType; N],
}

impl<const N: usize> InferTypes for FixedTypes<N> {
    fn infer_types(
        &self,
        _inputs: &[Option<DataType>],
    ) -> Result<Vec<Option<DataType>>, InferShapesError> {
        Ok(self.types.iter().copied().map(Some).collect())
    }
}

/// Type inference for operators that always return a single output with type
/// i32.
pub const ALWAYS_INT: FixedTypes<1> = FixedTypes {
    types: [DataType::Int32],
};

/// Type inference for operators that always return a single output with type
/// f32.
pub const ALWAYS_FLOAT: FixedTypes<1> = FixedTypes {
    types: [DataType::Float],
};

/// Errors that prevent shape inference from finishing.
///
/// Shape inference can still complete if errors only happen for certain nodes.
/// In that case errors for individual operators are returned in the
/// [`InferShapesResult`] value.
pub enum InferError {
    #[allow(unused)] // Currently ignored downstream
    PlanError(RunError),
}

/// Results of shape and type inference.
pub struct InferResult {
    /// Map of value node ID to inferred shape.
    pub shapes: HashMap<NodeId, Vec<Dimension>>,

    /// Map of value node ID to inferred type.
    pub types: HashMap<NodeId, DataType>,
}

/// Infer the shapes and types of operator outputs in a graph.
///
/// Returns a map of value node ID to shape.
pub fn infer_graph(graph: &Graph) -> Result<InferResult, InferError> {
    let ops = graph
        .execution_plan(graph.input_ids(), graph.output_ids(), Default::default())
        .map_err(InferError::PlanError)?;

    let mut symbolic_values: HashMap<NodeId, SymTensor> = HashMap::new();
    let mut types: HashMap<NodeId, DataType> = HashMap::new();
    let mut all_values: HashSet<NodeId> = HashSet::new();
    let mut symbol_gen = SymbolGen::new();

    let debug = env_flag("RTEN_INFER_SHAPES_DEBUG", false);

    // Temp buffers for type and shape inference operands.
    let mut input_types: Vec<Option<DataType>> = Vec::new();
    let mut input_shapes: Vec<SymTensor> = Vec::new();

    for op_id in ops {
        let Some(Node::Operator(op)) = graph.get_node(op_id) else {
            // TODO - Return an error if the plan includes non-op nodes.
            continue;
        };

        all_values.extend(op.input_ids().iter().flatten());
        all_values.extend(op.output_ids().iter().flatten());

        // Perform type inference
        if let Some(type_infer) = op.operator().as_infer_types() {
            input_types.clear();
            input_types.extend(op.input_ids().iter().map(|&id| {
                let id = id?;

                if let Some(dtype) = types.get(&id) {
                    Some(*dtype)
                } else {
                    graph.get_node(id)?.dtype()
                }
            }));

            let output_types = type_infer.infer_types(&input_types);

            if let Ok(output_types) = output_types {
                for (id, dtype) in op.output_ids().iter().zip(output_types) {
                    if let Some(id) = id
                        && let Some(dtype) = dtype
                    {
                        types.insert(*id, dtype);
                    }
                }
            }
        }

        // Perform shape inference
        if let Some(infer) = op.operator().as_infer_shapes() {
            input_shapes.clear();
            input_shapes.extend(op.input_ids().iter().map(|input_id| {
                let Some(input_id) = input_id else {
                    return SymTensor::unknown("missing input");
                };

                match graph.get_node(*input_id) {
                    Some(Node::Constant(constant)) => {
                        if let Some(scalar) = constant.as_scalar()
                            // nb `as_scalar` will return a value if the constant
                            // is a vector with one item. Only convert to a scalar
                            // if it is actually scalar.
                            && constant.ndim() == 0
                        {
                            SymTensor::from_scalar(SymElem::Value(scalar))
                        } else if let Some(vec) = constant.as_vector() {
                            let vec = vec.to_vec().into_iter().map(SymElem::Value).collect();
                            SymTensor::from_vec(vec)
                        } else {
                            SymTensor::from_fixed_shape(constant.shape())
                        }
                    }
                    Some(Node::Value(val)) => {
                        if let Some(dims) = symbolic_values.get(input_id) {
                            dims.clone()
                        } else if let Some(shape) = val.shape() {
                            let sym_shape = shape
                                .iter()
                                .map(|dim| match dim {
                                    Dimension::Symbolic(name) => SymElem::Var(
                                        Symbol {
                                            name: name.clone(),
                                            positive: true,
                                        }
                                        .into(),
                                    ),
                                    Dimension::Fixed(size) => SymElem::Value(*size as i32),
                                })
                                .collect();
                            SymTensor::from_shape(sym_shape)
                        } else {
                            SymTensor::unknown("unknown value shape")
                        }
                    }
                    Some(Node::Operator(_)) | None => unreachable!("invalid input ID"),
                }
            }));

            let out_shapes = infer.infer_shapes(&input_shapes, &mut symbol_gen);

            if debug {
                println!(
                    "op {} inputs {:?} outputs {:?}",
                    op.name().unwrap_or(""),
                    input_shapes,
                    out_shapes
                );
            }

            match out_shapes {
                Ok(out_shapes) => {
                    for (out_id, out_shape) in op.output_ids().iter().zip(out_shapes) {
                        let Some(out_id) = out_id else {
                            continue;
                        };
                        symbolic_values.insert(*out_id, out_shape.simplify());
                    }
                }
                Err(_) => {
                    // TODO - Track error for diagnostics.
                }
            }
        }
    }

    // Extract shapes from symbolic values.
    let mut shapes = HashMap::with_capacity(symbolic_values.len());
    for (value_id, sym_value) in symbolic_values {
        if let Some(dims) = sym_value.shape() {
            let dims = dims
                .map(|dim| match dim {
                    SymElem::Value(size) => {
                        // TODO - Eliminate this panic.
                        assert!(size >= 0);
                        Dimension::Fixed(size as usize)
                    }
                    dim => Dimension::Symbolic(dim.to_string()),
                })
                .collect();
            shapes.insert(value_id, dims);
        }
    }

    Ok(InferResult { shapes, types })
}

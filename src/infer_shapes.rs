//! Shape and type inference for values in a graph.

use std::collections::HashMap;

use crate::env::env_flag;
use crate::graph::{Dimension, Graph, Node, NodeId, RunError, TypedConstant};
use crate::operator::{OutputType, OutputTypesContext};
use crate::value::ValueType;

pub use rten_shape_inference::{
    infer_shapes::{BinaryOp, InferShapes, InferShapesError, ReductionOp, UnaryOp},
    sym_gen::SymbolGen,
    sym_tensor::{SymElem, SymTensor, Symbol},
};

/// Impl [`InferShapes`] for a type by delegating to another type which
/// implements the trait.
///
/// This is used by operators in this crate to delegate shape inference to types
/// defined in the rten_shape_inference crate.
macro_rules! impl_infer_shapes {
    ($op:ident, $self:ident, $make_impl:expr) => {
        impl rten_shape_inference::infer_shapes::InferShapes for $op {
            fn infer_shapes(
                &self,
                inputs: &[rten_shape_inference::sym_tensor::SymTensor],
                sym_gen: &mut rten_shape_inference::sym_gen::SymbolGen,
            ) -> Result<
                Vec<rten_shape_inference::sym_tensor::SymTensor>,
                rten_shape_inference::infer_shapes::InferShapesError,
            > {
                let $self = self;
                let shape_op = $make_impl;
                shape_op.infer_shapes(inputs, sym_gen)
            }
        }
    };
}
pub(crate) use impl_infer_shapes;

/// Errors that prevent shape inference from finishing.
///
/// Shape inference can still complete if errors only happen for certain nodes.
/// In that case the shapes or types will be treated as unknown.
#[derive(Debug)]
pub enum InferError {
    #[allow(unused)] // Currently ignored downstream
    PlanError(RunError),
}

/// Results of shape and type inference.
pub struct InferResult {
    /// Map of value node ID to inferred shape.
    pub shapes: HashMap<NodeId, Vec<Dimension>>,

    /// Map of value node ID to inferred type.
    pub types: HashMap<NodeId, ValueType>,
}

/// Infer the shapes and types of operator outputs in a graph.
///
/// Inference works on a best-effort basis and is not guaranteed to be able
/// to determine the shape and type of every output. Reasons why inference for
/// a node can fail include:
///
/// - A graph operator does not specify its shape and type inference rules
/// - Graph inputs are missing shape or type information
/// - The shapes depend on data in the graph inputs
/// - The shape of an operator output is a function of inputs with symbolic
///   sizes and the inference infrastructure is unable to represent the
///   function.
pub fn infer_shapes(graph: &Graph) -> Result<InferResult, InferError> {
    let mut symbol_gen = SymbolGen::new();

    let ops = graph
        .execution_plan(graph.input_ids(), graph.output_ids(), Default::default())
        .map_err(InferError::PlanError)?;

    // Symbolic shapes (or values) and types of operator outputs processed so far.
    //
    // Reserve initial capacity assuming each operator produces one output,
    // which is the case for most operators.
    let mut values: HashMap<NodeId, SymTensor> = HashMap::with_capacity(ops.len());
    let mut types: HashMap<NodeId, ValueType> = HashMap::with_capacity(ops.len());

    let debug = env_flag("RTEN_INFER_SHAPES_DEBUG", false);

    // Temp buffer for shape inference operands.
    let mut input_shapes: Vec<SymTensor> = Vec::new();

    for op_id in ops {
        let Some(Node::Operator(op)) = graph.get_node(op_id) else {
            unreachable!("invalid execution plan");
        };

        // Perform type inference
        let types_ctx = OutputTypesContext {
            num_outputs: op.output_ids().len(),
        };
        if let Some(output_type_list) = op.operator().output_types(&types_ctx) {
            for (id, output_type) in op.output_ids().iter().zip(output_type_list) {
                let Some(id) = id else {
                    // Unused optional output.
                    continue;
                };

                let get_input_type = |index: u32| {
                    op.input_ids()
                        .get(index as usize)
                        .copied()
                        .flatten()
                        .and_then(|id| {
                            if let Some(dtype) = types.get(&id) {
                                Some(*dtype)
                            } else {
                                graph.get_node(id)?.dtype()
                            }
                        })
                };

                let dtype = match output_type {
                    OutputType::Fixed(dtype) => Some(dtype),
                    OutputType::CopyFromInput(index) => get_input_type(index),
                    OutputType::ElementTypeOfInputSequence(index) => {
                        get_input_type(index).map(|t| t.to_tensor_type())
                    }
                    OutputType::SequenceWithElementTypeOfInput(index) => {
                        get_input_type(index).map(|t| t.to_sequence_type())
                    }
                };
                if let Some(dtype) = dtype {
                    types.insert(*id, dtype);
                }
            }
        }

        // Perform shape inference
        if let Some(infer) = op.operator().as_infer_shapes() {
            input_shapes.clear();
            input_shapes.extend(op.input_ids().iter().map(|input_id| {
                input_id
                    .and_then(|id| {
                        let node = graph.get_node(id)?;
                        Some(sym_tensor_from_input(id, node, &values))
                    })
                    .unwrap_or_else(|| SymTensor::unknown("missing input"))
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
                        values.insert(*out_id, out_shape.simplify());
                    }
                }
                Err(_) => {
                    // TODO - Track error for diagnostics.
                }
            }
        }
    }

    // Extract shapes from symbolic values.
    let mut shapes = HashMap::with_capacity(values.len());
    for (value_id, sym_value) in values {
        if let Some(dims) = sym_value.shape() {
            let dims = dims
                .map(|dim| match dim {
                    // If a dimension size is unexpectedly inferred as a negative
                    // value, just ignore it.
                    SymElem::Value(size) if size >= 0 => Some(Dimension::Fixed(size as usize)),
                    dim => Some(Dimension::Symbolic(dim.to_string())),
                })
                .collect::<Option<Vec<_>>>();
            if let Some(dims) = dims {
                shapes.insert(value_id, dims);
            }
        }
    }

    Ok(InferResult { shapes, types })
}

/// Convert an operator input into a symbolic tensor.
///
/// If the input is a constant, we can use its shape and values directly. If
/// it is a value node and we have inferred its shape and value from shape
/// inference of previous operators then we can use that. Otherwise use
/// information about its shape that is baked into the model.
fn sym_tensor_from_input(
    input_id: NodeId,
    node: &Node,
    values: &HashMap<NodeId, SymTensor>,
) -> SymTensor {
    match node {
        Node::Constant(constant) => {
            // `as_scalar` will return a value if the constant is a vector
            // with one item. Only convert to a scalar if it is actually scalar.
            if let Some(scalar) = constant.as_scalar()
                && constant.ndim() == 0
            {
                SymTensor::from_scalar(SymElem::Value(scalar))
            } else if let Some(vec) = constant.as_vector() {
                let vec = vec.iter().copied().map(SymElem::Value).collect();
                SymTensor::from_vec(vec)
            } else {
                SymTensor::from_fixed_shape(constant.shape())
            }
        }
        Node::Value(val) => {
            if let Some(dims) = values.get(&input_id) {
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
        // If we reach here, the graph was constructed incorrectly.
        Node::Operator(_) => unreachable!("operator input is not a value or constant"),
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;

    use crate::Dimension;
    use crate::graph::builder::{Expr, OutputMeta, dims};
    use crate::ops::{MatMul, Split};
    use crate::value::{DataType, ValueType};

    use super::infer_shapes;

    #[test]
    fn test_infer_shapes() {
        let graph = {
            let x = Expr::value_with_info("data", DataType::Float, &dims!("batch", 64));
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
            let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };

        let shapes = infer_shapes(&graph).unwrap();

        let output_id = graph.output_ids()[0];
        assert_eq!(
            shapes.shapes.get(&output_id).map(|s| s.as_slice()),
            Some(dims!("batch", 12).as_slice())
        );
        assert_eq!(
            shapes.types.get(&output_id).copied(),
            Some(ValueType::Tensor(DataType::Float))
        );
    }

    #[test]
    fn test_infer_split_op_types() {
        let graph = {
            let x = Expr::value_with_info("data", DataType::Float, &dims!("batch", 64));
            let split = x.apply(
                Split {
                    axis: -1,
                    num_outputs: None,
                },
                &[],
                &[OutputMeta::NoMeta, OutputMeta::NoMeta],
            );
            let split_0 = split.output(0);
            let split_1 = split.output(1);
            Expr::make_graph(&[x], &[split_0, split_1])
        };
        assert_eq!(graph.output_ids().len(), 2);

        let result = infer_shapes(&graph).unwrap();

        for output_id in graph.output_ids() {
            assert_eq!(
                result.types.get(&output_id).copied(),
                Some(ValueType::Tensor(DataType::Float))
            );
        }
    }
}

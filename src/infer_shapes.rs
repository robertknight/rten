//! Shape and type inference for values in a graph.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use rten_base::num::AsUsize;
use rten_tensor::Tensor;
use smallvec::SmallVec;

use crate::env::env_flag;
use crate::graph;
use crate::graph::{Dimension, Graph, Node, NodeId, OperatorNode, TypedConstant};
use crate::operator::{OutputType, OutputTypesContext};
use crate::value::ValueType;

pub use rten_shape_inference::{
    BinaryOp, Constant, InferShapes, InferShapesContext, InferShapesError, ReductionOp, SymExpr,
    SymTensor, Symbol, SymbolGen, UnaryOp,
};

/// Impl [`InferShapes`] for a type by delegating to another type which
/// implements the trait.
///
/// This is used by operators in this crate to delegate shape inference to types
/// defined in the rten_shape_inference crate.
macro_rules! impl_infer_shapes {
    ($op:ident, $self:ident, $make_impl:expr) => {
        impl rten_shape_inference::InferShapes for $op {
            fn infer_shapes(
                &self,
                inputs: rten_shape_inference::InferShapesContext,
                sym_gen: &mut rten_shape_inference::SymbolGen,
            ) -> Result<
                Vec<rten_shape_inference::SymTensor>,
                rten_shape_inference::InferShapesError,
            > {
                let $self = self;
                let shape_op = $make_impl;
                shape_op.infer_shapes(inputs, sym_gen)
            }
        }
    };
}
pub(crate) use impl_infer_shapes;

/// Details of an operator which encountered a shape or type inference error.
#[derive(Debug)]
pub struct OpInfo {
    pub name: String,
    pub op_type: String,
}

/// Errors that prevent shape inference from finishing.
///
/// Depending on the settings that shape inference is run with, shape inference
/// may attempt to keep going after an error is encountered or may abort.
#[derive(Debug)]
pub enum InferError {
    /// Type inference failed for an operator.
    TypeInferenceFailed(OpInfo),
    /// Shape inference is not implemented for an operator.
    UnsupportedOperator(OpInfo),
    /// Shape inference failed for an operator.
    ///
    /// Shape inference can fail if the inputs to an operator are incorrect
    /// (wrong count, wrong rank, incompatible).
    ShapeInferenceFailed(OpInfo),
    /// Shape inference was incomplete for an operator.
    ///
    /// _Incomplete_ shape inference means that shape inference successfully
    /// ran, but at least one output has a dimension of unknown size.
    ShapeInferenceIncomplete(OpInfo),
    /// Shape inference produced a symbolic expression that exceeds the
    /// complexity limit.
    ShapeTooComplex(OpInfo),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeInferenceFailed(op_info) => write!(
                f,
                "type inference failed for {} op \"{}\"",
                op_info.op_type, op_info.name
            ),
            Self::UnsupportedOperator(op_info) => {
                write!(
                    f,
                    "shape inference unsupported for {} op \"{}\"",
                    op_info.op_type, op_info.name
                )
            }
            Self::ShapeInferenceFailed(op_info) => write!(
                f,
                "shape inference failed for {} op \"{}\"",
                op_info.op_type, op_info.name
            ),
            Self::ShapeInferenceIncomplete(op_info) => write!(
                f,
                "shape inference incomplete for {} op \"{}\"",
                op_info.op_type, op_info.name
            ),
            Self::ShapeTooComplex(op_info) => write!(
                f,
                "shape too complex for {} op \"{}\"",
                op_info.op_type, op_info.name
            ),
        }
    }
}

impl Error for InferError {}

/// Info about a value node determined by shape inference.
#[derive(Debug, PartialEq)]
pub enum Shape {
    Constant { index: usize },
    Shape(Vec<Dimension>),
}

/// Results of shape and type inference.
#[derive(Debug)]
pub struct InferResult {
    /// Unique constants.
    pub constants: Vec<Constant>,

    /// Map of value node ID to inferred shape or constant index.
    pub shapes: HashMap<NodeId, Shape>,

    /// Map of value node ID to inferred type.
    pub types: HashMap<NodeId, ValueType>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferShapeOptions {
    /// Enable strict shape inference mode.
    ///
    /// When true, [`infer_shapes`] will return an error if shape or type
    /// inference is incomplete for any operator. When false, shape inference
    /// is best-effort and will continue with remaining operators in the event
    /// of an error.
    pub strict: bool,

    /// Upper limit on the maximum complexity of symbolic expressions that
    /// shape inference may produce.
    ///
    /// The value is the maximum depth of any expression tree.
    pub max_complexity: u32,
}

impl Default for InferShapeOptions {
    fn default() -> Self {
        InferShapeOptions {
            strict: false,
            max_complexity: 10,
        }
    }
}

/// Trait for looking up nodes in a graph by ID.
pub trait GetNode {
    fn get_node(&self, id: NodeId) -> Option<&Node>;
    fn is_capture(&self, id: NodeId) -> bool;
    fn get_node_id(&self, name: &str) -> Option<NodeId>;
}

impl GetNode for Graph {
    fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.get_node(id)
    }

    fn is_capture(&self, id: NodeId) -> bool {
        self.captures().contains(&id)
    }

    fn get_node_id(&self, name: &str) -> Option<NodeId> {
        self.get_node_id(name)
    }
}

#[derive(Clone)]
pub struct ShapeEnv<'current, 'parent> {
    graph: &'current dyn GetNode,
    shapes: &'current InferredShapes,
    parent: Option<&'parent ShapeEnv<'parent, 'parent>>,
}

impl<'current, 'parent> ShapeEnv<'current, 'parent> {
    pub fn new(
        graph: &'current dyn GetNode,
        shapes: &'current InferredShapes,
        parent: Option<&'parent ShapeEnv<'parent, 'parent>>,
    ) -> Self {
        Self {
            graph,
            shapes,
            parent,
        }
    }

    fn lookup_type(&self, id: NodeId) -> Option<ValueType> {
        if self.graph.is_capture(id) {
            let name = self.graph.get_node(id).and_then(|n| n.name())?;
            let mut parent = self.parent;
            while let Some(current) = parent {
                if let Some(cap_id) = current.graph.get_node_id(name) {
                    return current.lookup_type(cap_id);
                }
                parent = current.parent;
            }
            None
        } else if let Some(dtype) = self.shapes.types.get(&id) {
            Some(dtype.clone())
        } else {
            self.graph.get_node(id)?.dtype()
        }
    }

    fn lookup_value(&self, id: NodeId) -> Option<SymTensor> {
        if self.graph.is_capture(id) {
            let name = self.graph.get_node(id).and_then(|n| n.name())?;
            let mut parent = self.parent;
            while let Some(current) = parent {
                if let Some(cap_id) = current.graph.get_node_id(name) {
                    return current.lookup_value(cap_id);
                }
                parent = current.parent;
            }
            None
        } else if let Some(value) = self.shapes.values.get(&id) {
            Some(value.clone())
        } else {
            let node = self.graph.get_node(id)?;
            Some(sym_tensor_from_input(id, node, &self.shapes.values))
        }
    }
}

/// Records inferred shapes and types for values in a graph.
pub struct InferredShapes {
    opts: InferShapeOptions,

    symbol_gen: SymbolGen,

    // Symbolic shapes (or values) and types of operator outputs processed so far.
    //
    // Reserve initial capacity assuming each operator produces one output,
    // which is the case for most operators.
    values: HashMap<NodeId, SymTensor>,
    types: HashMap<NodeId, ValueType>,

    // Temp buffer for shape inference operands.
    input_shapes: Vec<Option<SymTensor>>,

    debug: bool,
}

impl InferredShapes {
    /// Create an incremental shape inference engine.
    ///
    /// `capacity` is a hint for the number of values whose shapes/values/types
    /// will be inferred. This is typically set to the number of operators in
    /// the graph.
    ///
    /// `graph` provides access to nodes in the graph for which shape inference
    /// is being performed.
    ///
    /// `parent` is the shape inference state for the parent graph, if any. It
    /// is used to look up the inferred shapes of values captured from ancestor
    /// graphs.
    pub fn new(capacity: usize, opts: InferShapeOptions) -> Self {
        let debug = env_flag("RTEN_INFER_SHAPES_DEBUG", false);

        Self {
            opts,
            symbol_gen: SymbolGen::new(),
            values: HashMap::with_capacity(capacity),
            types: HashMap::with_capacity(capacity),
            input_shapes: Vec::new(),
            debug,
        }
    }

    /// Infer the shapes or values of an operator's outputs and persist them
    /// in the current state.
    pub fn infer(
        &mut self,
        op: &OperatorNode,
        graph: &dyn GetNode,
        parent: Option<&ShapeEnv>,
    ) -> Result<(), InferError> {
        let op_info = || OpInfo {
            name: op.name().unwrap_or_default().to_string(),
            op_type: op.operator().name().to_string(),
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

                let env = ShapeEnv::new(graph, self, parent);
                let get_input_type = |index: u32| {
                    op.input_ids()
                        .get(index.as_usize())
                        .copied()
                        .flatten()
                        .and_then(|id| env.lookup_type(id))
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
                    self.types.insert(*id, dtype);
                } else if self.opts.strict {
                    return Err(InferError::TypeInferenceFailed(op_info()));
                }
            }
        } else if self.opts.strict {
            return Err(InferError::TypeInferenceFailed(op_info()));
        }

        // Perform shape inference
        if let Some(infer) = op.operator().as_infer_shapes() {
            let mut input_shapes = std::mem::take(&mut self.input_shapes);

            let env = ShapeEnv::new(graph, self, parent);
            input_shapes.clear();
            input_shapes.extend(
                op.input_ids()
                    .iter()
                    .map(|input_id| input_id.and_then(|id| env.lookup_value(id))),
            );

            let out_shapes = infer.infer_shapes(
                InferShapesContext::new(&self.input_shapes),
                &mut self.symbol_gen,
            );

            self.input_shapes = input_shapes;

            if self.debug {
                println!(
                    "op {} inputs {:?} outputs {:?}",
                    op.name().unwrap_or(""),
                    self.input_shapes,
                    out_shapes
                );
            }

            match out_shapes {
                Ok(out_shapes) => {
                    for (out_id, out_shape) in op.output_ids().iter().zip(out_shapes) {
                        let Some(out_id) = out_id else {
                            // Ignore outputs that the model doesn't use.
                            continue;
                        };

                        // Fail inference if any output dimension has an unknown shape.
                        if self.opts.strict {
                            let has_unknown = if let Some(mut out_shape) = out_shape.shape() {
                                out_shape.any(|dims| {
                                    dims.iter().any(|expr| match expr {
                                        // If we encounter a synthetic variable, this means that
                                        // the size of a dimension could not be computed.
                                        SymExpr::Var(symbol) => symbol.synthetic,
                                        _ => false,
                                    })
                                })
                            } else {
                                // Output rank is unknown.
                                true
                            };

                            if has_unknown {
                                return Err(InferError::ShapeInferenceIncomplete(op_info()));
                            }
                        }

                        // Handle excessively complex symbolic expressions in the shape.
                        //
                        // We do this to avoid building up excessively complex expressions on which
                        // operations such as simplification become very slow. See
                        // https://github.com/robertknight/rten/issues/1298.
                        let mut out_shape = out_shape;
                        let had_complex = out_shape.replace_complex_expressions(
                            self.opts.max_complexity,
                            &mut self.symbol_gen,
                        );
                        if self.opts.strict && had_complex {
                            return Err(InferError::ShapeTooComplex(op_info()));
                        }

                        self.values.insert(*out_id, out_shape.simplify());
                    }
                }
                Err(_) => {
                    if self.opts.strict {
                        return Err(InferError::ShapeInferenceFailed(op_info()));
                    }
                }
            }
        } else if self.opts.strict {
            return Err(InferError::UnsupportedOperator(op_info()));
        }

        Ok(())
    }

    pub fn into_result(self) -> InferResult {
        let InferredShapes {
            values,
            types,
            debug,
            ..
        } = self;

        // Unique constant values.
        let mut constants = Vec::new();
        let mut constant_to_index = HashMap::new();
        let mut total_const_values = 0;

        // Map of value ID to shape.
        let mut shapes = HashMap::with_capacity(values.len());

        for (value_id, sym_value) in values {
            let shape = if let Some(val) = sym_value.to_constant() {
                total_const_values += 1;
                if let Some(&index) = constant_to_index.get(&val) {
                    Some(Shape::Constant { index })
                } else {
                    let index = constants.len();
                    constant_to_index.insert(val.clone(), index);
                    constants.push(val);
                    Some(Shape::Constant { index })
                }
            } else if let Some(dims) = sym_value.shape() {
                let dims = dims
                    .map(|dim| match dim {
                        // If a dimension size is unexpectedly inferred as a negative
                        // value, just ignore it.
                        SymExpr::Value(size) if size >= 0 => Some(Dimension::Fixed(size as usize)),
                        dim => Some(Dimension::Symbolic(dim.to_string())),
                    })
                    .collect::<Option<Vec<_>>>();
                dims.map(Shape::Shape)
            } else {
                None
            };

            if let Some(shape) = shape {
                shapes.insert(value_id, shape);
            }
        }

        if debug {
            println!(
                "Shape inference: {} constant values, {} unique",
                total_const_values,
                constants.len()
            );
        }

        InferResult {
            constants,
            shapes,
            types,
        }
    }
}

/// Apply the results of shape inference to a graph.
pub fn apply_shapes(graph: &mut Graph, shapes: InferResult) {
    let const_ids: Vec<NodeId> = shapes
        .constants
        .into_iter()
        .map(|constant| {
            let tensor = match constant {
                rten_shape_inference::Constant::Scalar(x) => Tensor::from(x),
                rten_shape_inference::Constant::Vector(vec) => Tensor::from(vec),
            };
            graph.add_constant(None, tensor.into_arc())
        })
        .collect();

    let replace_value = |graph: &mut Graph, old_value_id, new_value_id| {
        // Replace `old_value_id` in operator inputs.
        let Some(consumer_ids) = graph.get_consumers(old_value_id) else {
            return;
        };
        let consumer_ids: SmallVec<[NodeId; 1]> = SmallVec::from_slice(consumer_ids);

        for op_id in consumer_ids {
            graph.replace_input(op_id, old_value_id, new_value_id);
        }
    };

    for (value_id, shape) in shapes.shapes {
        match shape {
            Shape::Constant { index } => {
                let const_id = const_ids[index];
                replace_value(graph, value_id, const_id);
            }
            Shape::Shape(shape) => {
                graph.update_value_shape(value_id, shape);
            }
        }
    }
    for (value_id, value_type) in shapes.types {
        graph.update_value_type(value_id, value_type);
    }
}

/// Convert a `f32` value to `i32` if it represents an exact integer that
/// fits in the `i32` range.
fn f32_to_int_checked(x: f32) -> Option<i32> {
    // `i32::MIN as f32` preserves the exact value. `i32::MAX as f32` rounds up
    // by one. Hence we use an exclusive upper bound.
    if x.is_finite() && x.fract() == 0.0 && x >= (i32::MIN as f32) && x < (i32::MAX as f32) {
        Some(x as i32)
    } else {
        None
    }
}

/// Extract a constant's scalar value as a symbolic values.
///
/// This supports `i32` constants and `f32` constants whose value is an
/// exact integer.
fn const_to_sym_scalar(constant: &graph::Constant) -> Option<SymExpr> {
    let int_val: Option<i32> = constant.as_scalar();
    if let Some(val) = int_val {
        return Some(SymExpr::Value(val));
    }

    let float_val: Option<f32> = constant.as_scalar();
    if let Some(val) = float_val.and_then(f32_to_int_checked) {
        return Some(SymExpr::Value(val));
    }

    None
}

/// Extract a constant's 1D values as symbolic values.
///
/// This supports `i32` vectors and `f32` vectors whose values are all exact
/// integers.
fn const_to_sym_vector(constant: &graph::Constant) -> Option<Vec<SymExpr>> {
    let int_vec: Option<&[i32]> = constant.as_vector();
    if let Some(int_vec) = int_vec {
        return Some(int_vec.iter().copied().map(SymExpr::Value).collect());
    }

    let float_vec: Option<&[f32]> = constant.as_vector();
    if let Some(float_vec) = float_vec {
        return float_vec
            .iter()
            .map(|&f| f32_to_int_checked(f).map(SymExpr::Value))
            .collect();
    }

    None
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
            // `const_to_sym_scalar` will return a value if the constant is a
            // vector with one item. Only convert to a scalar if it is actually
            // scalar.
            if let Some(scalar) = const_to_sym_scalar(constant)
                && constant.ndim() == 0
            {
                SymTensor::from_scalar(scalar)
            } else if let Some(vec) = const_to_sym_vector(constant) {
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
                        Dimension::Symbolic(name) => SymExpr::Var(
                            Symbol {
                                name: name.clone(),
                                positive: true,
                                synthetic: false,
                            }
                            .into(),
                        ),
                        Dimension::Fixed(size) => SymExpr::Value(*size as i32),
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
pub fn infer_shapes(graph: &Graph, opts: InferShapeOptions) -> Result<InferResult, InferError> {
    let ops = graph
        .execution_plan(graph.input_ids(), graph.output_ids(), Default::default())
        .unwrap();

    let mut ctx = InferredShapes::new(ops.len(), opts);
    for op_id in ops {
        let Some(Node::Operator(op)) = graph.get_node(op_id) else {
            unreachable!("invalid execution plan");
        };
        ctx.infer(op, graph, None)?;
    }
    Ok(ctx.into_result())
}

#[cfg(test)]
mod tests {
    use rten_tensor::NdTensor;

    use crate::Dimension;
    use crate::graph::TypedConstant;
    use crate::graph::builder::{Expr, OutputMeta, dims};
    use crate::ops::{Concat, Gather, Gemm, MatMul, Shape as ShapeOp, Split, Unsqueeze};
    use crate::value::{DataType, ValueType};

    use super::{Constant, InferError, InferShapeOptions, Shape, apply_shapes, infer_shapes};

    #[test]
    fn test_infer_shapes() {
        let graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
            let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };

        let shapes = infer_shapes(&graph, Default::default()).unwrap();

        let output_id = graph.output_ids()[0];
        let Some(Shape::Shape(shape)) = shapes.shapes.get(&output_id) else {
            panic!("output is not a shape");
        };
        assert_eq!(shape.as_slice(), dims!("batch", 12).as_slice());
        assert_eq!(
            shapes.types.get(&output_id).copied(),
            Some(ValueType::Tensor(DataType::Float))
        );
    }

    #[test]
    fn test_infer_shapes_strict() {
        let opts = InferShapeOptions {
            strict: true,
            ..Default::default()
        };

        // Successful strict shape inference
        let graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
            let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };
        let result = infer_shapes(&graph, opts.clone());
        assert!(result.is_ok());

        // Incomplete shape inference.
        let graph = {
            let x = Expr::value("data"); // Missing type, shape
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
            let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };
        let result = infer_shapes(&graph, opts.clone());
        assert!(
            matches!(&result, Err(InferError::ShapeInferenceIncomplete(op_info)) if op_info.name == "MatMul"),
            "{:?} is not expected error",
            result
        );

        // Failed shape inference.
        let graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );
            // RHS input to Gemm with too few dims.
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64]));
            let out = x.apply(
                Gemm {
                    alpha: 1.,
                    beta: 0.,
                    transpose_a: false,
                    transpose_b: false,
                },
                &[w],
                &[OutputMeta::NoMeta],
            );
            out.build_graph(&["data"])
        };
        let result = infer_shapes(&graph, opts.clone());
        assert!(
            matches!(&result, Err(InferError::ShapeInferenceFailed(op_info)) if op_info.name == "Gemm"),
            "{:?} is not expected error",
            result
        );

        // Unsuccessful type inference
        let graph = {
            let x = Expr::value("data");
            let out = x.clone() + x;
            out.build_graph(&["data"])
        };
        let result = infer_shapes(&graph, opts.clone());
        assert!(
            matches!(&result, Err(InferError::TypeInferenceFailed(op_info)) if op_info.name == "Add"),
            "{:?} is not expected error",
            result
        );
    }

    #[test]
    fn test_infer_split_op_types() {
        let graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );
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

        let result = infer_shapes(&graph, Default::default()).unwrap();

        for output_id in graph.output_ids() {
            assert_eq!(
                result.types.get(&output_id).copied(),
                Some(ValueType::Tensor(DataType::Float))
            );
        }
    }

    #[test]
    fn test_infer_constants() {
        // Create graph that extracts and concatenates the last two dims of
        // an input shape. Since these are fixed, the final output is a constant.
        let graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64, 32),
            );
            let shape = x.apply(
                ShapeOp {
                    start: None,
                    end: None,
                },
                &[],
                &[OutputMeta::NoMeta],
            );
            let dim1 = shape.apply(
                Gather { axis: 0 },
                &[Expr::constant(1)],
                &[OutputMeta::NoMeta],
            );
            let dim2 = shape.apply(
                Gather { axis: 0 },
                &[Expr::constant(2)],
                &[OutputMeta::NoMeta],
            );
            let axes = Expr::constant(NdTensor::from([0i32]));
            let dim1_vec = dim1.apply(Unsqueeze {}, &[axes.clone()], &[OutputMeta::NoMeta]);
            let dim2_vec = dim2.apply(Unsqueeze {}, &[axes], &[OutputMeta::NoMeta]);
            let dims_vec = dim1_vec.apply(Concat { axis: 0 }, &[dim2_vec], &[OutputMeta::NoMeta]);
            dims_vec.build_graph(&["data"])
        };

        let output_id = graph.output_ids()[0];
        let result = infer_shapes(&graph, Default::default()).unwrap();

        let shape = result.shapes.get(&output_id).unwrap();
        let Shape::Constant { index } = shape else {
            panic!("{:?} is not a constant", shape);
        };
        assert_eq!(result.constants[*index], Constant::Vector(vec![64, 32]));
    }

    #[test]
    fn test_apply_shapes() {
        // Build a graph that has input shape and type metadata, but no output
        // metadata.
        let mut graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );
            let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
            let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };

        // Infer shapes
        let shapes = infer_shapes(&graph, Default::default()).unwrap();
        apply_shapes(&mut graph, shapes);

        // Verify that values were updated with inferred shapes and types.
        let output = graph.get_node(graph.output_ids()[0]).unwrap();
        assert_eq!(
            output.shape().as_deref(),
            Some(dims!("batch", 12).as_slice())
        );
        assert_eq!(output.dtype(), Some(ValueType::Tensor(DataType::Float)));
    }

    #[test]
    fn test_apply_shapes_replaces_values_with_constants() {
        let mut graph = {
            let x = Expr::value_with_info(
                "data",
                ValueType::Tensor(DataType::Float),
                &dims!("batch", 64),
            );

            // Extract second dimension of input via `Gather<axis=0>(Shape(X), indices=[1])`.
            let indices = Expr::constant(1);
            let out = x
                .unary(ShapeOp {
                    start: None,
                    end: None,
                })
                .apply(Gather { axis: 0 }, &[indices], &[OutputMeta::NoMeta]);
            out.build_graph(&["data"])
        };

        // Infer shapes
        let shapes = infer_shapes(&graph, Default::default()).unwrap();
        apply_shapes(&mut graph, shapes);

        // The output should be replaced with a constant as it doesn't depend on
        // model inputs.
        let output = graph.get_node(graph.output_ids()[0]).unwrap();
        assert_eq!(output.as_constant().and_then(|c| c.as_scalar()), Some(64));
    }
}

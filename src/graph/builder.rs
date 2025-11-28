//! Tools to simplify building graphs in tests.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

use crate::graph::{Graph, NodeId};
use crate::operator::Operator;
use crate::{DataType, Dimension, Value};

enum ExprKind {
    /// Expression representing a value node.
    Value(ValueExpr),
    /// Expression representing a constant node.
    Constant(Value),
    /// Expression representing an operator node.
    Operator(OperatorExpr),
    /// Expression representing a specific output of an operator node.
    OperatorOutput(OperatorOutputExpr),
}

/// An expression describing a [`Graph`].
///
/// Expressions are constructed using constructor methods and math operators.
/// They are then converted into a model graph using [`Expr::build_graph`].
///
/// The following builds a graph for the widely-used GELU activation function
/// which has the equation `x * 0.5 * (1 + Erf(X / Sqrt(2)))`:
///
/// ```
/// use crate::graph::builder::Expr;
///
/// let x = Expr::value("x");
/// let sqrt_2 = Expr::constant((2.0f32).sqrt());
/// let one = Expr::constant(1.0);
/// let half = Expr::constant(0.5);
/// let expr = x.clone() * ((x / sqrt_2).unary(Erf {}) + one) * half;
/// let graph: Graph = expr.build_graph(["x"]);
/// ```
///
/// This graph has a single input, a value node with the name "x", and one
/// output that corresponds to the output of the final `Mul` operator.
#[derive(Clone)]
pub struct Expr {
    kind: Rc<ExprKind>,
}

impl From<ExprKind> for Expr {
    fn from(kind: ExprKind) -> Expr {
        Expr { kind: kind.into() }
    }
}

impl Expr {
    /// Create an expression representing a runtime-computed value (eg. model
    /// inputs).
    pub fn value(name: &str) -> Expr {
        Expr::from(ExprKind::Value(ValueExpr {
            name: name.to_string(),
            dtype: None,
            shape: None,
        }))
    }

    /// Create an expression representing a runtime-computed value (eg. model
    /// inputs), with shape and dtype information.
    pub fn value_with_info(name: &str, dtype: DataType, shape: &[Dimension]) -> Expr {
        Expr::from(ExprKind::Value(ValueExpr {
            name: name.to_string(),
            dtype: Some(dtype),
            shape: Some(shape.to_vec()),
        }))
    }

    /// Create an expression representing a constant value.
    pub fn constant<V>(value: V) -> Expr
    where
        V: Into<Value>,
    {
        Expr::from(ExprKind::Constant(value.into()))
    }

    /// Create an expression which applies a unary operator to this expression.
    pub fn unary<Op: Operator + Send + Sync>(&self, op: Op) -> Expr {
        self.apply(op, &[], &[OutputMeta::NoMeta])
    }

    /// Create an expression which applies a binary operator to this expression.
    pub fn binary<Op: Operator + Send + Sync>(&self, op: Op, rhs: Expr) -> Expr {
        self.apply(op, &[rhs], &[OutputMeta::NoMeta])
    }

    /// Create an expression which applies an operator to this expression.
    pub fn apply<Op: Operator + Send + Sync>(
        &self,
        op: Op,
        operands: &[Expr],
        outputs: &[OutputMeta],
    ) -> Expr {
        let mut inputs: Vec<_> = [self.clone()].into();
        inputs.extend(operands.iter().map(|opr| opr.clone()));
        Expr::from(ExprKind::Operator(OperatorExpr {
            op: Arc::new(op),
            inputs,
            outputs: outputs.to_vec(),
        }))
    }

    /// Create an expression which refers to the index'th output of the `self`
    /// operator expression.
    pub fn output(&self, index: usize) -> Expr {
        let ExprKind::Operator(op_info) = self.kind.as_ref() else {
            panic!("can only call `output` on an operator expression");
        };
        assert!(
            index < op_info.outputs.len(),
            "can't get output {} for operator with {} outputs",
            index,
            op_info.outputs.len()
        );
        Expr::from(ExprKind::OperatorOutput(OperatorOutputExpr {
            op: self.clone(),
            output_index: index,
        }))
    }

    /// Convert this expression into a graph.
    ///
    /// The inputs of the graph are values with names listed in `inputs`. The
    /// output is the node that corresponds to the result of the `self`
    /// expression.
    ///
    /// This function only supports creating graphs with a single output. To
    /// create graphs with multiple outputs, use [`make_graph`](Self::make_graph).
    pub fn build_graph<'a, I: AsRef<[&'a str]>>(self, inputs: I) -> Graph {
        let mut graph = Graph::new();
        let mut expr_output_ids = HashMap::new();
        let mut name_gen = NodeNameGenerator::new();
        let output_ids = self.add_to_graph(&mut graph, &mut name_gen, &mut expr_output_ids);

        let input_ids: Vec<NodeId> = inputs
            .as_ref()
            .iter()
            .map(|name| {
                graph
                    .get_node_id(name)
                    .expect("input name passed to `build_graph` not found in graph")
            })
            .collect();
        graph.set_input_ids(&input_ids);
        graph.set_output_ids(&output_ids);

        graph
    }

    /// Create a graph with the given inputs and outputs.
    pub fn make_graph<I: AsRef<[Expr]>, O: AsRef<[Expr]>>(inputs: I, outputs: O) -> Graph {
        Self::make_graph_impl(inputs.as_ref(), outputs.as_ref())
    }

    fn make_graph_impl(inputs: &[Expr], outputs: &[Expr]) -> Graph {
        let mut graph = Graph::new();
        let mut expr_output_ids = HashMap::new();
        let mut name_gen = NodeNameGenerator::new();

        let extend_unique = |output: &mut Vec<NodeId>, new_ids: Vec<NodeId>| {
            for id in new_ids {
                if !output.contains(&id) {
                    output.push(id);
                }
            }
        };

        let mut output_ids = Vec::new();
        for output in outputs {
            let new_output_ids =
                output.add_to_graph(&mut graph, &mut name_gen, &mut expr_output_ids);
            extend_unique(&mut output_ids, new_output_ids);
        }

        let mut input_ids = Vec::new();
        for input in inputs {
            let new_input_ids = input.add_to_graph(&mut graph, &mut name_gen, &mut expr_output_ids);
            extend_unique(&mut input_ids, new_input_ids);
        }

        graph.set_input_ids(&input_ids);
        graph.set_output_ids(&output_ids);

        graph
    }

    fn add_to_graph(
        &self,
        graph: &mut Graph,
        name_gen: &mut NodeNameGenerator,
        expr_output_ids: &mut HashMap<ExprRef, Vec<NodeId>>,
    ) -> Vec<NodeId> {
        if let Some(node_ids) = expr_output_ids.get(&ExprRef(self.clone())) {
            return node_ids.clone();
        }

        let output_ids: Vec<NodeId> = match self.kind.as_ref() {
            ExprKind::Value(value_info) => [graph.add_value(
                Some(value_info.name.as_str()),
                value_info.shape.clone(),
                value_info.dtype,
            )]
            .into(),
            ExprKind::Constant(value) => {
                let name = name_gen.generate("const");
                let const_id = match value {
                    Value::FloatTensor(value) => {
                        graph.add_constant(Some(name.as_str()), value.clone().into_arc())
                    }
                    Value::Int32Tensor(value) => {
                        graph.add_constant(Some(name.as_str()), value.clone().into_arc())
                    }
                    Value::Int8Tensor(value) => {
                        graph.add_constant(Some(name.as_str()), value.clone().into_arc())
                    }
                    _ => unimplemented!("constant type not supported"),
                };
                [const_id].into()
            }
            ExprKind::Operator(op_info) => {
                let op_inputs: Vec<_> = op_info
                    .inputs
                    .iter()
                    .flat_map(|input_expr| {
                        input_expr.add_to_graph(graph, name_gen, expr_output_ids)
                    })
                    .map(Some)
                    .collect();

                let op_outputs: Vec<NodeId> = op_info
                    .outputs
                    .iter()
                    .map(|output_info| {
                        let output_name = name_gen.generate(&format!("{}_out", op_info.op.name()));
                        let (output_dtype, output_shape) = match output_info {
                            OutputMeta::NoMeta => (None, None),
                            OutputMeta::Meta((dtype, shape)) => (Some(*dtype), Some(shape.clone())),
                        };
                        graph.add_value(Some(output_name.as_str()), output_shape, output_dtype)
                    })
                    .collect();

                let op_outputs_opt: Vec<_> = op_outputs.iter().copied().map(Some).collect();

                let op_name = name_gen.generate(op_info.op.name());
                graph.add_op(
                    Some(op_name.as_str()),
                    op_info.op.clone(),
                    &op_inputs,
                    &op_outputs_opt,
                );

                op_outputs
            }
            ExprKind::OperatorOutput(output_info) => {
                let output_ids = output_info
                    .op
                    .add_to_graph(graph, name_gen, expr_output_ids);
                [output_ids[output_info.output_index]].into()
            }
        };
        expr_output_ids.insert(ExprRef(self.clone()), output_ids.clone());

        output_ids
    }
}

/// Wrapper around an `Expr` which uses reference-equality.
struct ExprRef(Expr);

impl PartialEq for ExprRef {
    fn eq(&self, other: &ExprRef) -> bool {
        Rc::ptr_eq(&self.0.kind, &other.0.kind)
    }
}

impl Eq for ExprRef {}

impl Hash for ExprRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0.kind).hash(state)
    }
}

/// Metadata about an operator output value.
#[derive(Clone)]
pub enum OutputMeta {
    /// Value without dtype or shape info.
    NoMeta,
    /// Value with dtype and shape info.
    Meta((DataType, Vec<Dimension>)),
}

struct OperatorExpr {
    op: Arc<dyn Operator + Send + Sync>,
    inputs: Vec<Expr>,
    outputs: Vec<OutputMeta>,
}

struct ValueExpr {
    name: String,
    dtype: Option<DataType>,
    shape: Option<Vec<Dimension>>,
}

struct OperatorOutputExpr {
    op: Expr,
    output_index: usize,
}

struct NodeNameGenerator {
    used_names: HashSet<String>,
}

impl NodeNameGenerator {
    fn new() -> NodeNameGenerator {
        NodeNameGenerator {
            used_names: HashSet::new(),
        }
    }

    fn generate(&mut self, prefix: &str) -> String {
        let mut name = prefix.to_string();
        let mut suffix = 0;
        while self.used_names.contains(&name) {
            suffix += 1;
            name = format!("{}_{}", prefix, suffix);
        }
        self.used_names.insert(name.clone());
        name
    }
}

macro_rules! impl_binary_op {
    ($op_trait:ident, $op_method:ident, $op_struct: ident) => {
        impl $op_trait for Expr {
            type Output = Expr;

            fn $op_method(self, rhs: Expr) -> Expr {
                self.binary(crate::ops::$op_struct {}, rhs)
            }
        }

        impl<V> $op_trait<V> for Expr
        where
            V: Into<Value>,
        {
            type Output = Expr;

            fn $op_method(self, rhs: V) -> Expr {
                self.binary(crate::ops::$op_struct {}, Expr::constant(rhs))
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Div, div, Div);
impl_binary_op!(Sub, sub, Sub);

/// Create a [`Dimension`] array from a list of symbolic names and fixed sizes.
macro_rules! dims {
    ($($x:expr),* $(,)?) => {
        [$(Dimension::from($x)),*]
    };
}
pub(crate) use dims;

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;

    use super::Expr;

    #[test]
    fn test_build_graph() {
        // Build expression featuring values, operators and constants, including
        // re-use of the same expression (`x_sqr`) and generate a graph from it.
        let x = Expr::value("x");
        let x_sqr = x.clone() * x.clone();
        let x_4_plus_2 = x_sqr.clone() * x_sqr.clone() + 2.0;
        let graph = x_4_plus_2.build_graph(["x"]);

        // Verify graph generates expected value from input when run.
        let in_id = graph.input_ids()[0];
        let out_id = graph.output_ids()[0];
        let x_val = Tensor::from(4.);
        let mut result = graph
            .run([(in_id, x_val.into())].into(), &[out_id], None, None)
            .unwrap();

        let expected = (4.0f32).powf(4.0) + 2.0;
        let result: Tensor<f32> = result.remove(0).try_into().unwrap();
        assert_eq!(result, Tensor::from(expected));
    }
}

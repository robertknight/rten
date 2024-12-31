//! Tools to simplify building graphs in tests.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

use rten_tensor::Tensor;

use crate::graph::{Graph, NodeId};
use crate::ops::Operator;

enum ExprKind {
    Value(String),
    // Constants are limited to f32 just to keep the initial implementation
    // simple. Expand as needed.
    Constant(Tensor<f32>),
    Operator(OperatorExpr),
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

struct OperatorExpr {
    // `Operator`s are not cloneable, so when we construct a graph from the
    // expression we need to take ownership of the operator and pass it to the
    // graph. However there may be multiple references to operator
    // sub-expressions. Consider:
    //
    //   let x = Expr::value("x");
    //   let x_sqr = x.clone() * x.clone();
    //   let x_4 = x_sqr.clone() * x_sqr.clone();
    //   x_4.build_graph() // Encounters `x_sqr` twice
    //
    // To handle this we put the operator in a cell. When we first
    // encounter it during graph generation we take it out and add it to the
    // graph. For subsequent references to the operator we will use the output
    // node ID of the already-added operator.
    op: Cell<Option<Box<dyn Operator + Send + Sync>>>,
    inputs: Vec<Expr>,
}

struct NodeNameGenerator {
    next_id: u32,
}

impl NodeNameGenerator {
    fn new(next_id: u32) -> NodeNameGenerator {
        NodeNameGenerator { next_id }
    }

    fn generate(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.next_id);
        self.next_id += 1;
        name
    }
}

impl Expr {
    /// Create an expression representing a runtime-computed value (eg. model
    /// inputs).
    pub fn value(name: &str) -> Expr {
        Expr::from(ExprKind::Value(name.to_string()))
    }

    /// Create an expression representing a constant value.
    pub fn constant<V>(value: V) -> Expr
    where
        V: Into<Tensor<f32>>,
    {
        Expr::from(ExprKind::Constant(value.into()))
    }

    /// Create an expression which applies a unary operator to this expression.
    pub fn unary<Op: Operator + Send + Sync>(&self, op: Op) -> Expr {
        Expr::from(ExprKind::Operator(OperatorExpr {
            op: Cell::new(Some(Box::new(op))),
            inputs: [self.clone()].into(),
        }))
    }

    /// Create an expression which applies a binary operator to this expression.
    pub fn binary<Op: Operator + Send + Sync>(&self, op: Op, rhs: Expr) -> Expr {
        Expr::from(ExprKind::Operator(OperatorExpr {
            op: Cell::new(Some(Box::new(op))),
            inputs: [self.clone(), rhs].into(),
        }))
    }

    /// Convert this expression into a graph.
    ///
    /// The inputs of the graph are values with names listed in `inputs`. The
    /// output is the node that corresponds to the result of the `self`
    /// expression.
    pub fn build_graph<'a, I: AsRef<[&'a str]>>(self, inputs: I) -> Graph {
        let mut graph = Graph::new();
        let mut expr_output_ids = HashMap::new();
        let mut name_gen = NodeNameGenerator::new(1);
        let output_id = self.add_to_graph(&mut graph, &mut name_gen, &mut expr_output_ids);

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
        graph.set_output_ids(&[output_id]);

        graph
    }

    fn add_to_graph(
        &self,
        graph: &mut Graph,
        name_gen: &mut NodeNameGenerator,
        expr_output_ids: &mut HashMap<ExprRef, NodeId>,
    ) -> NodeId {
        if let Some(node_id) = expr_output_ids.get(&ExprRef(self.clone())) {
            return *node_id;
        }

        let output_id = match self.kind.as_ref() {
            ExprKind::Value(name) => graph.add_value(Some(name.as_str()), None, None),
            ExprKind::Constant(value) => {
                let name = name_gen.generate("const");
                graph.add_constant(Some(name.as_str()), value.clone())
            }
            ExprKind::Operator(op_info) => {
                let op_inputs: Vec<_> = op_info
                    .inputs
                    .iter()
                    .map(|input_expr| {
                        Some(input_expr.add_to_graph(graph, name_gen, expr_output_ids))
                    })
                    .collect();

                let op = op_info
                    .op
                    .take()
                    .expect("operator has already been added to graph");

                let output_name = name_gen.generate(&format!("{}_out", op.name()));
                let op_output = graph.add_value(Some(output_name.as_str()), None, None);

                let op_name = name_gen.generate(op.name());
                graph.add_op(Some(op_name.as_str()), op, &op_inputs, &[Some(op_output)]);
                op_output
            }
        };
        expr_output_ids.insert(ExprRef(self.clone()), output_id);

        output_id
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
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Div, div, Div);
impl_binary_op!(Sub, sub, Sub);

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
        let x_4_plus_2 = x_sqr.clone() * x_sqr.clone() + Expr::constant(2.0);
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

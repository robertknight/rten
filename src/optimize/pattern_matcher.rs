use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::graph::{Constant, Graph, Node, NodeId, OperatorNode};
use crate::ops::ValueView;

/// Tracks an association between named symbols (variables) in a pattern and
/// the node IDs they have been resolved to.
struct SymbolMap {
    // Map of `(name, node_id)` for resolved symbols. This is modified only
    // by extending and truncating it.
    symbols: Vec<(&'static str, NodeId)>,

    // Stack of checkpoints. Each is the length of `symbols` at the time of
    // the checkpoint.
    checkpoints: Vec<usize>,
}

impl SymbolMap {
    fn new() -> SymbolMap {
        SymbolMap {
            symbols: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    /// Save the current state of the map.
    ///
    /// This is useful if we need to backtrack during pattern matching.
    fn checkpoint(&mut self) {
        self.checkpoints.push(self.symbols.len());
    }

    /// Discard any new symbols recorded since the last call to `checkpoint`.
    fn revert(&mut self) {
        if let Some(checkpoint) = self.checkpoints.pop() {
            self.symbols.truncate(checkpoint);
        }
    }

    /// Add a new symbol-node association.
    fn add(&mut self, name: &'static str, node_id: NodeId) {
        self.symbols.push((name, node_id));
    }

    /// Find the node ID that a symbol has been resolved to.
    fn find(&self, name: &str) -> Option<NodeId> {
        self.symbols.iter().find_map(|(sym_name, node_id)| {
            if *sym_name == name {
                Some(*node_id)
            } else {
                None
            }
        })
    }
}

/// The result of matching a [`Pattern`] against a graph node.
pub struct Match {
    symbols: SymbolMap,
}

impl Match {
    /// Return the node ID that a symbol or operator was resolved to.
    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.symbols.find(name)
    }
}

/// Absolute tolerance for matching float constants against constant patterns.
const CONST_TOLERANCE: f32 = 1e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantPattern {
    value: f32,
}

impl ConstantPattern {
    fn matches(&self, node: &Constant) -> bool {
        match node.as_view() {
            ValueView::FloatTensor(t) => t
                .item()
                .is_some_and(|x| (x - self.value).abs() <= CONST_TOLERANCE),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpPattern {
    /// Name of the operator (eg. "MatMul")
    name: &'static str,

    /// Patterns that the inputs must match.
    inputs: Vec<Pattern>,

    /// Identifier which can be used to look up the operator node ID after a
    /// successful match.
    key: Option<&'static str>,
}

impl OpPattern {
    fn matches(&self, node: &OperatorNode, graph: &Graph, symbols: &mut SymbolMap) -> bool {
        if node.operator().name() != self.name {
            return false;
        }
        if self.inputs.len() != node.input_ids().len() {
            return false;
        }

        // For commutative binary operators, we allow the pattern to match
        // either way around.
        if let (true, [pat_a, pat_b], [Some(input_a), Some(input_b)]) = (
            node.operator().is_commutative(),
            &self.inputs[..],
            node.input_ids(),
        ) {
            symbols.checkpoint();

            if pat_a.test_impl(*input_a, graph, symbols)
                && pat_b.test_impl(*input_b, graph, symbols)
            {
                return true;
            }

            symbols.revert();

            pat_b.test_impl(*input_a, graph, symbols) && pat_a.test_impl(*input_b, graph, symbols)
        } else {
            self.inputs
                .iter()
                .zip(node.input_ids())
                .all(|(input_expr, input_id)| {
                    input_id.map(|input_id| input_expr.test_impl(input_id, graph, symbols))
                        == Some(true)
                })
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SymbolPattern {
    name: &'static str,

    /// True if this symbol can only match a constant.
    constant: bool,
}

/// Specifies a pattern for a subgraph within a [`Graph`].
///
/// Patterns consist of matchers for operators, constants and symbols
/// (variables). These are matched against a node in in a [`Graph`]. The node
/// matches if it is the output of a subgraph that matches the pattern.
///
/// Patterns are created using constructor methods and combined to form patterns
/// that can match sub-graphs within a graph. For example
/// `Pattern::constant(1.0) + Pattern::symbol("x")` describes a graph with an
/// `Add` operator that takes the float constant `1.0` and a free variable `x`
/// as inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Matches an operator.
    Operator(OpPattern),
    /// Matches a constant value.
    Constant(ConstantPattern),
    /// Matches either a constant or a value.
    Symbol(SymbolPattern),
}

impl Pattern {
    /// Create a pattern that matches an operator.
    pub fn operator<I: Into<Vec<Pattern>>>(name: &'static str, inputs: I) -> Pattern {
        Pattern::Operator(OpPattern {
            name,
            inputs: inputs.into(),
            key: None,
        })
    }

    /// Create a pattern that matches a binary operator.
    pub fn binary_op<A: Into<Pattern>, B: Into<Pattern>>(
        name: &'static str,
        input_a: A,
        input_b: B,
    ) -> Pattern {
        let inputs: [Pattern; 2] = [input_a.into(), input_b.into()];
        Pattern::operator(name, inputs)
    }

    /// Create a pattern that matches a unary operator.
    pub fn unary_op<I: Into<Pattern>>(name: &'static str, input: I) -> Pattern {
        let inputs: [Pattern; 1] = [input.into()];
        Pattern::operator(name, inputs)
    }

    /// Set the identifier for a pattern, used to look up the node ID in a
    /// match using [`Match::node_id`].
    pub fn with_name(self, name: &'static str) -> Pattern {
        match self {
            Pattern::Operator(mut op) => {
                op.key = Some(name);
                Pattern::Operator(op)
            }
            Pattern::Symbol(mut symbol) => {
                symbol.name = name;
                Pattern::Symbol(symbol)
            }
            // Constants don't currently support keys.
            Pattern::Constant(constant) => Pattern::Constant(constant),
        }
    }

    /// Create a pattern that matches a constant node with a given value.
    pub fn constant(value: f32) -> Pattern {
        Pattern::Constant(ConstantPattern { value })
    }

    /// Create a pattern that matches any value.
    ///
    /// In order for a pattern to match a node, all symbols with the same name
    /// must resolve to the same node.
    pub fn symbol(name: &'static str) -> Pattern {
        Pattern::Symbol(SymbolPattern {
            name,
            constant: false,
        })
    }

    /// Create a pattern that matches a constant.
    ///
    /// Unlike [`constant`](Self::constant), the value of the constant is not specified.
    pub fn const_symbol(name: &'static str) -> Pattern {
        Pattern::Symbol(SymbolPattern {
            name,
            constant: true,
        })
    }

    /// Test this pattern is a subgraph of a graph.
    ///
    /// The matching starts from `node_id`, which specifies the output of
    /// the subgraph.
    ///
    /// If the pattern matches, this returns a [`Match`] which allows looking
    /// up the node IDs that any symbols in the pattern were resolved to.
    pub fn test(&self, node_id: NodeId, graph: &Graph) -> Option<Match> {
        let mut symbols = SymbolMap::new();
        if self.test_impl(node_id, graph, &mut symbols) {
            Some(Match { symbols })
        } else {
            None
        }
    }

    /// Match this pattern against a subgraph with output `node_id` and record
    /// symbol-node associations in `symbols`.
    fn test_impl(&self, node_id: NodeId, graph: &Graph, symbols: &mut SymbolMap) -> bool {
        let Some(node) = graph.get_node(node_id) else {
            return false;
        };

        match (self, node) {
            // Operator patterns can match either an operator node or an
            // operator output.
            (Pattern::Operator(op_pat), Node::Operator(op_node)) => {
                if op_pat.matches(op_node, graph, symbols) {
                    if let Some(key) = op_pat.key {
                        symbols.add(key, node_id);
                    }
                    true
                } else {
                    false
                }
            }
            (Pattern::Operator(op_pat), Node::Value(_)) => {
                let Some((op_node_id, op_node)) = graph.get_source_node(node_id) else {
                    return false;
                };
                if op_pat.matches(op_node, graph, symbols) {
                    if let Some(key) = op_pat.key {
                        symbols.add(key, op_node_id);
                    }
                    true
                } else {
                    false
                }
            }
            (Pattern::Constant(const_pat), Node::Constant(const_node)) => {
                const_pat.matches(const_node)
            }
            (Pattern::Symbol(sym_pat), Node::Constant(_) | Node::Value(_)) => {
                if sym_pat.constant && !matches!(node, Node::Constant(_)) {
                    return false;
                }

                // If we have seen this symbol before, it must resolve to the
                // same node. Otherwise it always matches.
                if let Some(resolved_id) = symbols.find(sym_pat.name) {
                    resolved_id == node_id
                } else {
                    symbols.add(sym_pat.name, node_id);
                    true
                }
            }
            _ => false,
        }
    }
}

impl From<f32> for Pattern {
    fn from(val: f32) -> Pattern {
        Pattern::constant(val)
    }
}

macro_rules! impl_binop_for_pattern {
    ($trait:ident, $method:ident, $op_name:expr) => {
        impl<I: Into<Pattern>> $trait<I> for Pattern {
            type Output = Pattern;

            fn $method(self, rhs: I) -> Pattern {
                Pattern::binary_op($op_name, self, rhs.into())
            }
        }

        impl $trait<Pattern> for f32 {
            type Output = Pattern;

            fn $method(self, rhs: Pattern) -> Pattern {
                Pattern::binary_op($op_name, Pattern::constant(self), rhs)
            }
        }
    };
}
impl_binop_for_pattern!(Add, add, "Add");
impl_binop_for_pattern!(Mul, mul, "Mul");
impl_binop_for_pattern!(Div, div, "Div");
impl_binop_for_pattern!(Sub, sub, "Sub");

impl Neg for Pattern {
    type Output = Pattern;

    fn neg(self) -> Pattern {
        Pattern::unary_op("Neg", self)
    }
}

#[cfg(test)]
mod tests {
    use super::Pattern;
    use crate::graph::builder::Expr;
    use crate::graph::{Graph, Node};
    use crate::ops::Abs;

    /// Create a graph that implements the softsign function `x / 1 + |x|`.
    fn softsign_graph() -> Graph {
        let x = Expr::value("x");
        let expr = x.clone() / (Expr::constant(1.0) + x.unary(Abs {}));
        expr.build_graph(["x"])
    }

    #[test]
    fn test_pattern_match() {
        struct Case {
            graph: Graph,
            pattern: Pattern,
            expect_match: bool,
        }

        let x = Pattern::symbol("x");
        let c = Pattern::const_symbol("c");
        let unary_op = Pattern::unary_op;

        let cases = [
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (1.0 + unary_op("Abs", x.clone())),
                expect_match: true,
            },
            // Pattern with constant symbol instead of fixed constant.
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (c.clone() + unary_op("Abs", x.clone())),
                expect_match: true,
            },
            // Pattern with operands of a non-commutative operator ("/") swapped.
            Case {
                graph: softsign_graph(),
                pattern: (1.0 + unary_op("Abs", x.clone())) / x.clone(),
                expect_match: false,
            },
            // Pattern with operands of a commutative operator ("+") swapped around.
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (unary_op("Abs", x.clone()) + 1.0),
                expect_match: true,
            },
            // Pattern with "+" operator swapped for "-".
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (1.0 - unary_op("Abs", x.clone())),
                expect_match: false,
            },
            // Pattern with modified constant value.
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (1.1 + unary_op("Abs", x.clone())),
                expect_match: false,
            },
            // Pattern with constant value which doesn't exactly match graph,
            // but is within the allowed tolerance.
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (1.00001 + unary_op("Abs", x.clone())),
                expect_match: true,
            },
            // Pattern where a symbol ("x") does not resolve to the same node
            // in all positions.
            Case {
                graph: softsign_graph(),
                pattern: x.clone() / (x.clone() + unary_op("Abs", x.clone())),
                expect_match: false,
            },
            // Pattern which has the right structure, except that a dynamic
            // input is matched against a constant symbol.
            Case {
                graph: softsign_graph(),
                pattern: c.clone() / (1.0 + unary_op("Abs", x.clone())),
                expect_match: false,
            },
        ];

        for (
            i,
            Case {
                graph,
                pattern,
                expect_match,
            },
        ) in cases.into_iter().enumerate()
        {
            let input = graph.input_ids()[0];
            let output = graph.output_ids()[0];
            let pat_match = pattern.test(output, &graph);

            assert_eq!(pat_match.is_some(), expect_match, "mismatch for case {}", i);
            if let Some(pat_match) = pat_match {
                assert_eq!(pat_match.node_id("x"), Some(input));
            }
        }
    }

    #[test]
    fn test_operator_with_key() {
        let graph = softsign_graph();
        let output = graph.output_ids()[0];
        let x = Pattern::symbol("x");
        let pat = x.clone() / (1.0 + Pattern::unary_op("Abs", x.clone()).with_name("abs_op"));
        let pat_match = pat.test(output, &graph).unwrap();
        let abs_node_id = pat_match.node_id("abs_op").unwrap();
        let abs_op = graph.get_node(abs_node_id).unwrap();
        assert!(matches!(abs_op, Node::Operator(op) if op.operator().name() == "Abs"));
    }
}

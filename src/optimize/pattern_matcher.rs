use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use smallvec::SmallVec;

use crate::graph::{Constant, Graph, Node, NodeId, OperatorNode};
use crate::value::ValueView;

/// Inline capacity for chains of associative+commutative operands.
const SMALL_VEC_CAP: usize = 4;

type PatternVec<'a> = SmallVec<[&'a Pattern; SMALL_VEC_CAP]>;
type NodeIdVec = SmallVec<[NodeId; SMALL_VEC_CAP]>;

/// Tells [`SymbolMap::transaction`] what to do with the symbol bindings added
/// during the transaction body.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SymbolsAction {
    /// Keep the bindings added during the transaction.
    Keep,
    /// Roll back the bindings added during the transaction.
    Discard,
}

impl SymbolsAction {
    fn is_keep(self) -> bool {
        matches!(self, SymbolsAction::Keep)
    }
}

/// Tracks an association between named symbols (variables) in a pattern and
/// the node IDs they have been resolved to.
struct SymbolMap {
    // Map of `(name, node_id)` for resolved symbols. This is modified only
    // by extending and truncating it.
    symbols: Vec<(&'static str, NodeId)>,
}

impl SymbolMap {
    fn new() -> SymbolMap {
        SymbolMap {
            symbols: Vec::new(),
        }
    }

    /// Run `f`, which may extend the symbol map, then either keep or discard
    /// any bindings it added based on the [`SymbolsAction`] it returns.
    ///
    /// Returns the result of `f`.
    fn transaction<F>(&mut self, f: F) -> SymbolsAction
    where
        F: FnOnce(&mut Self) -> SymbolsAction,
    {
        let saved_len = self.symbols.len();
        let action = f(self);
        if action == SymbolsAction::Discard {
            self.symbols.truncate(saved_len);
        }
        action
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
    tolerance: f32,
}

impl ConstantPattern {
    fn new(value: f32) -> Self {
        ConstantPattern {
            value,
            tolerance: CONST_TOLERANCE,
        }
    }

    fn exact(value: f32) -> Self {
        ConstantPattern {
            value,
            tolerance: 0.,
        }
    }

    fn matches(&self, node: &Constant) -> bool {
        match node.as_view() {
            ValueView::FloatTensor(t) => t
                .item()
                .is_some_and(|x| (x - self.value).abs() <= self.tolerance),
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

        let op = node.operator();

        // For binary operators that are both associative and commutative
        // (eg. `Add`, `Mul`), match against the flattened chain of operands.
        // This makes patterns insensitive to the bracketing of nested chains:
        // a pattern of the form `Op(Op(a, b), c)` will match a graph of the
        // form `Op(a, Op(b, c))` and vice versa.
        //
        // The multiset match only adds value when at least one side has a
        // nested chain (so its flattened length is >= 3). If both sides
        // flatten to two inputs, the strict commutative matcher below handles
        // the case identically.
        if op.is_associative() && op.is_commutative() && self.inputs.len() == 2 {
            let patterns = self.flatten_associative_chain();
            if patterns.len() >= 3 {
                let nodes = flatten_graph_associative_chain(node, graph, self.name);
                if patterns.len() == nodes.len()
                    && match_pattern_set(&patterns, &nodes, graph, symbols)
                {
                    return true;
                }
            }
        }

        // For commutative binary operators, we allow the pattern to match
        // either way around.
        if op.is_commutative()
            && let [pat_a, pat_b] = &self.inputs[..]
            && let [Some(input_a), Some(input_b)] = node.input_ids()
        {
            let action = symbols.transaction(|s| {
                if pat_a.test_impl(*input_a, graph, s) && pat_b.test_impl(*input_b, graph, s) {
                    SymbolsAction::Keep
                } else {
                    SymbolsAction::Discard
                }
            });
            if action.is_keep() {
                return true;
            }
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

    /// Flatten operands of a nested associative pattern.
    ///
    /// eg. `Add(A, Add(B, C))` is flattened to `[A, B, C]`.
    fn flatten_associative_chain(&self) -> PatternVec<'_> {
        let mut patterns = PatternVec::new();
        for input in &self.inputs {
            flatten_associative_pattern_impl(input, self.name, &mut patterns);
        }
        patterns
    }
}

fn flatten_associative_pattern_impl<'a>(
    pat: &'a Pattern,
    op_name: &'static str,
    patterns: &mut PatternVec<'a>,
) {
    if let PatternKind::Operator(op_pat) = &*pat.kind
        && op_pat.name == op_name
        && op_pat.inputs.len() == 2
        // Don't flatten through patterns that have a key, since we need to
        // preserve them as a single unit so the matching node ID can be
        // recorded.
        && op_pat.key.is_none()
    {
        for input in &op_pat.inputs {
            flatten_associative_pattern_impl(input, op_name, patterns);
        }
    } else {
        patterns.push(pat);
    }
}

/// Flatten the graph subtree rooted at `node`, descending recursively through
/// any operator with the same name as `op_name` that has two inputs. Returns
/// the list of sub-graph nodes that form the chain.
fn flatten_graph_associative_chain(node: &OperatorNode, graph: &Graph, op_name: &str) -> NodeIdVec {
    let mut nodes = NodeIdVec::new();
    for input in node.input_ids() {
        match input {
            Some(input_id) => flatten_associative_graph_impl(*input_id, graph, op_name, &mut nodes),
            None => {
                // Bail out: a missing input means we can't represent the chain
                // faithfully. Return an empty list so the multiset matcher
                // falls back to the strict matcher.
                return NodeIdVec::new();
            }
        }
    }
    nodes
}

fn flatten_associative_graph_impl(
    node_id: NodeId,
    graph: &Graph,
    op_name: &str,
    nodes: &mut NodeIdVec,
) {
    if let Some((_, op_node)) = graph.get_source_node(node_id)
        && op_node.operator().name() == op_name
        && let [Some(lhs), Some(rhs)] = op_node.input_ids()
    {
        flatten_associative_graph_impl(*lhs, graph, op_name, nodes);
        flatten_associative_graph_impl(*rhs, graph, op_name, nodes);
    } else {
        nodes.push(node_id);
    }
}

/// Try to match each pattern against a distinct graph node, allowing any
/// permutation. Returns true if a successful assignment is found, in which case
/// symbol bindings are added to `symbols`.
fn match_pattern_set(
    patterns: &[&Pattern],
    nodes: &[NodeId],
    graph: &Graph,
    symbols: &mut SymbolMap,
) -> bool {
    debug_assert_eq!(patterns.len(), nodes.len());
    let mut used = SmallVec::<[bool; SMALL_VEC_CAP]>::from_elem(false, nodes.len());
    match_pattern_set_recursive(patterns, nodes, &mut used, graph, symbols)
}

fn match_pattern_set_recursive(
    patterns: &[&Pattern],
    nodes: &[NodeId],
    used: &mut [bool],
    graph: &Graph,
    symbols: &mut SymbolMap,
) -> bool {
    let Some((pat, rest)) = patterns.split_first() else {
        return true;
    };

    for (i, &graph_input) in nodes.iter().enumerate() {
        if used[i] {
            continue;
        }
        let action = symbols.transaction(|s| {
            if !pat.test_impl(graph_input, graph, s) {
                return SymbolsAction::Discard;
            }
            used[i] = true;
            if match_pattern_set_recursive(rest, nodes, used, graph, s) {
                SymbolsAction::Keep
            } else {
                used[i] = false;
                SymbolsAction::Discard
            }
        });
        if action.is_keep() {
            return true;
        }
    }

    false
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SymbolPattern {
    name: &'static str,

    /// True if this symbol can only match a constant.
    constant: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum PatternKind {
    /// Matches an operator.
    Operator(OpPattern),
    /// Matches a constant value.
    Constant(ConstantPattern),
    /// Matches either a constant or a value.
    Symbol(SymbolPattern),
    /// Matches any pattern from a set.
    AnyOf(Vec<Pattern>),
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
pub struct Pattern {
    kind: Rc<PatternKind>,
}

impl Pattern {
    /// Create a pattern that matches an operator.
    pub fn operator<I: Into<Vec<Pattern>>>(name: &'static str, inputs: I) -> Pattern {
        PatternKind::Operator(OpPattern {
            name,
            inputs: inputs.into(),
            key: None,
        })
        .into()
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
        let mut kind = self.kind.clone();

        match Rc::make_mut(&mut kind) {
            PatternKind::Operator(op) => {
                op.key = Some(name);
            }
            PatternKind::Symbol(symbol) => {
                symbol.name = name;
            }
            PatternKind::Constant(_) | PatternKind::AnyOf(_) => {}
        }

        Self { kind }
    }

    /// Create a pattern that matches a constant node with a given value.
    pub fn constant(value: f32) -> Pattern {
        PatternKind::Constant(ConstantPattern::new(value)).into()
    }

    /// Create a pattern that matches a constant node with a given value.
    ///
    /// Unlike [`constant`](Self::constant) the value must match exactly with
    /// no tolerance.
    pub fn exact_constant(value: f32) -> Pattern {
        PatternKind::Constant(ConstantPattern::exact(value)).into()
    }

    /// Create a pattern that matches any value.
    ///
    /// In order for a pattern to match a node, all symbols with the same name
    /// must resolve to the same node.
    pub fn symbol(name: &'static str) -> Pattern {
        PatternKind::Symbol(SymbolPattern {
            name,
            constant: false,
        })
        .into()
    }

    /// Create a pattern that matches a constant.
    ///
    /// Unlike [`constant`](Self::constant), the value of the constant is not specified.
    pub fn const_symbol(name: &'static str) -> Pattern {
        PatternKind::Symbol(SymbolPattern {
            name,
            constant: true,
        })
        .into()
    }

    /// Create a pattern that matches any pattern from a list.
    ///
    /// Patterns are matched from left-to-right. This means that if one pattern
    /// is an extension of another, the extension should be listed first.
    pub fn any_of(patterns: Vec<Pattern>) -> Pattern {
        PatternKind::AnyOf(patterns).into()
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

        match (&*self.kind, node) {
            // Operator patterns can match either an operator node or an
            // operator output.
            (PatternKind::Operator(op_pat), Node::Operator(op_node))
                if op_pat.matches(op_node, graph, symbols) =>
            {
                if let Some(key) = op_pat.key {
                    symbols.add(key, node_id);
                }
                true
            }
            (PatternKind::Operator(op_pat), Node::Value(_)) => {
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
            (PatternKind::Constant(const_pat), Node::Constant(const_node)) => {
                const_pat.matches(const_node)
            }
            (PatternKind::Symbol(sym_pat), Node::Constant(_) | Node::Value(_)) => {
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
            (PatternKind::AnyOf(patterns), _) => patterns.iter().any(|pattern| {
                symbols
                    .transaction(|s| {
                        if pattern.test_impl(node_id, graph, s) {
                            SymbolsAction::Keep
                        } else {
                            SymbolsAction::Discard
                        }
                    })
                    .is_keep()
            }),
            _ => false,
        }
    }

    /// Return true if this pattern contains a symbol with a given name.
    pub fn contains_symbol(&self, name: &str) -> bool {
        match &*self.kind {
            PatternKind::Operator(op) => {
                op.name == name || op.inputs.iter().any(|pat| pat.contains_symbol(name))
            }
            PatternKind::Constant(_) => false,
            PatternKind::Symbol(sym_pat) => sym_pat.name == name,
            PatternKind::AnyOf(patterns) => patterns.iter().any(|pat| pat.contains_symbol(name)),
        }
    }
}

impl From<PatternKind> for Pattern {
    fn from(kind: PatternKind) -> Pattern {
        Pattern {
            kind: Rc::new(kind),
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
    use crate::ops::{Abs, Pow, Reciprocal, Sqrt};

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

    #[test]
    fn test_shared_sub_pattern() {
        let sqrt_expr = Expr::value("x").unary(Sqrt {});
        let rsqrt_div_graph = (Expr::constant(1.) / sqrt_expr.clone()).build_graph(["x"]);
        let rsqrt_rcp_graph = sqrt_expr.unary(Reciprocal {}).build_graph(["x"]);

        // Pattern which wraps a common inner pattern in multiple alternative
        // outer patterns.
        let sqrt_pat = Pattern::unary_op("Sqrt", Pattern::symbol("x"));
        let rsqrt_pat = Pattern::any_of(
            [
                1. / sqrt_pat.clone(),
                Pattern::unary_op("Reciprocal", sqrt_pat),
            ]
            .into(),
        );

        let div_match = rsqrt_pat
            .test(rsqrt_div_graph.output_ids()[0], &rsqrt_div_graph)
            .unwrap();
        assert_eq!(
            div_match.node_id("x").unwrap(),
            rsqrt_div_graph.input_ids()[0]
        );

        let rcp_match = rsqrt_pat
            .test(rsqrt_rcp_graph.output_ids()[0], &rsqrt_rcp_graph)
            .unwrap();
        assert_eq!(
            rcp_match.node_id("x").unwrap(),
            rsqrt_rcp_graph.input_ids()[0]
        );
    }

    /// A pattern of the form `(a op b) op c` for an associative+commutative
    /// `op` should match any bracketing of an equivalent chain in the graph.
    #[test]
    fn test_pattern_match_associative_chain() {
        let make_graph = |build: fn(Expr, Expr, Expr) -> Expr| -> Graph {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let c = Expr::value("c");
            let pow_ab = a.clone().binary(Pow {}, b);
            build(a, pow_ab, c).build_graph(["a", "b", "c"])
        };

        // Pattern is `(x * pow(x, y)) * z` (left-associated). Each input is
        // distinct: `x` and `z` are symbols, the middle input is a `Pow`.
        let x = Pattern::symbol("x");
        let y = Pattern::symbol("y");
        let z = Pattern::symbol("z");
        let pat = x.clone() * Pattern::binary_op("Pow", x.clone(), y.clone()) * z.clone();

        let matching = [
            // `(a * pow(a,b)) * c` — same shape as the pattern.
            (|a, p, c| a * p * c) as fn(Expr, Expr, Expr) -> Expr,
            // `a * (pow(a,b) * c)` — right-associated.
            |a, p, c| a * (p * c),
            // `(c * a) * pow(a,b)` — operands re-ordered.
            |a, p, c| c * a * p,
            // `pow(a,b) * (a * c)` — re-ordered and re-associated.
            |a, p, c| p * (a * c),
        ];

        for (i, build) in matching.iter().enumerate() {
            let graph = make_graph(*build);
            let m = pat.test(graph.output_ids()[0], &graph);
            assert!(m.is_some(), "expected match for case {}", i);
        }

        // The chain in the graph only has two `Mul` operands (the pattern
        // expects three), so the pattern must not match.
        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let pow_ab = a.clone().binary(Pow {}, b);
            (a * pow_ab).build_graph(["a", "b"])
        };
        assert!(pat.test(graph.output_ids()[0], &graph).is_none());
    }

    /// `Equal` is commutative (`a == b` iff `b == a`) but not associative
    /// (`(a == b) == c` differs from `a == (b == c)` in general). A pattern
    /// with a chain of `Equal` ops should NOT match a re-associated graph:
    /// the associative-chain path is gated on `is_associative()` being true.
    #[test]
    fn test_pattern_match_skips_non_associative_ops() {
        use crate::ops::Equal;

        let w = Pattern::symbol("w");
        let x = Pattern::symbol("x");
        let y = Pattern::symbol("y");
        let z = Pattern::symbol("z");
        let eq = |a, b| Pattern::binary_op("Equal", a, b);
        let pat = eq(eq(w, x), eq(y, z));

        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let c = Expr::value("c");
            let d = Expr::value("d");
            let eq_ab = a.binary(Equal {}, b);
            let eq_abc = eq_ab.binary(Equal {}, c);
            let eq_abcd = eq_abc.binary(Equal {}, d);
            eq_abcd.build_graph(["a", "b", "c", "d"])
        };

        assert!(pat.test(graph.output_ids()[0], &graph).is_none());
    }

    /// Flattening an associative chain must not descend through a keyed
    /// pattern: doing so would prevent the key from being recorded.
    #[test]
    fn test_pattern_match_associative_preserves_keyed_pattern() {
        let x = Pattern::symbol("x");
        let y = Pattern::symbol("y");
        let z = Pattern::symbol("z");
        let inner = Pattern::binary_op("Mul", x, y).with_name("inner");
        let pat = inner * z;

        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let c = Expr::value("c");
            ((a * b) * c).build_graph(["a", "b", "c"])
        };

        let m = pat.test(graph.output_ids()[0], &graph).unwrap();
        let inner_id = m
            .node_id("inner")
            .expect("expected key 'inner' to be recorded");
        let inner_node = graph.get_node(inner_id).unwrap();
        assert!(matches!(inner_node, Node::Operator(op) if op.operator().name() == "Mul"));
    }

    /// An associative pattern with more inputs than the graph chain should not
    /// match. An associative pattern with fewer inputs than the graph chain
    /// should still match by binding a single symbol to a `Mul` sub-chain.
    #[test]
    fn test_pattern_match_associative_partial() {
        let x = Pattern::symbol("x");
        let y = Pattern::symbol("y");
        let pat = x.clone() * y.clone();

        let graph = {
            let a = Expr::value("a");
            let b = Expr::value("b");
            let c = Expr::value("c");
            (a * b * c).build_graph(["a", "b", "c"])
        };

        let m = pat.test(graph.output_ids()[0], &graph).unwrap();
        let x_id = m.node_id("x").unwrap();
        let y_id = m.node_id("y").unwrap();
        assert_ne!(x_id, y_id);
    }
}

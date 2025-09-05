use std::error::Error;
use std::sync::Arc;

use rten_base::byte_cast::cast_pod_slice;
use rten_tensor::Tensor;
use rten_testing::TestCases;

use super::{GraphOptimizer, OptimizeError, OptimizeOptions};
use crate::constant_storage::{ArcSlice, ArcTensorView, ConstantStorage};
use crate::graph::builder::{Expr, OutputMeta};
use crate::graph::{
    CaptureEnv, Constant, Graph, Node, NodeId, OperatorNode, PlanOptions, TypedConstant,
};
use crate::ops::{
    Add, Cast, DynamicQuantizeLinear, Erf, FusedMatMul, Gelu, Identity, LayerNormalization, MatMul,
    MatMulInteger, Neg, Pow, ReduceMean, RmsNormalization, Shape, Sigmoid, Slice, Softmax, Sqrt,
    Swish, Tanh, Transpose,
};
use crate::{DataType, Dimension};

fn optimize_graph(graph: Graph) -> Result<Graph, OptimizeError> {
    let optimizer = GraphOptimizer::new();
    optimizer.optimize(graph, None, OptimizeOptions::default())
}

fn arc_tensor_view(val: f32) -> ArcTensorView<f32> {
    let const_data = Vec::from(val.to_le_bytes());
    let const_storage = Arc::new(ConstantStorage::Buffer(const_data));
    let slice = ArcSlice::new(
        const_storage.clone(),
        cast_pod_slice(const_storage.data()).unwrap(),
    )
    .unwrap();
    ArcTensorView::from_data(&[], slice)
}

/// Extends [`Expr`] with methods to create expressions for specific
/// operations.
///
/// For example `a.matmul(b)` returns an expression for a `MatMul` graph
/// node with `a` and `b` as inputs.
trait OpExprs {
    fn cast(&self, to: DataType) -> Expr;
    fn erf(&self) -> Expr;
    fn identity(&self) -> Expr;
    fn pow(&self, rhs: Expr) -> Expr;
    fn matmul(&self, rhs: Expr) -> Expr;
    fn mean(&self) -> Expr;
    fn mean_axes(&self, axes: Expr) -> Expr;
    fn shape(&self) -> Expr;
    fn sigmoid(&self) -> Expr;
    fn slice(&self, starts: Expr, ends: Expr) -> Expr;
    fn square(&self) -> Expr;
    fn sqrt(&self) -> Expr;
    fn softmax(&self, axis: isize) -> Expr;
    fn tanh(&self) -> Expr;
    fn transpose(&self) -> Expr;
}

impl OpExprs for Expr {
    fn cast(&self, to: DataType) -> Expr {
        self.unary(Cast { to })
    }

    fn erf(&self) -> Expr {
        self.unary(Erf {})
    }

    fn identity(&self) -> Expr {
        self.unary(Identity {})
    }

    fn matmul(&self, rhs: Expr) -> Expr {
        self.binary(MatMul {}, rhs)
    }

    fn mean(&self) -> Expr {
        self.unary(ReduceMean {
            axes: Some(vec![-1]),
            keep_dims: false,
        })
    }

    fn mean_axes(&self, axes: Expr) -> Expr {
        self.binary(
            ReduceMean {
                axes: None,
                keep_dims: false,
            },
            axes,
        )
    }

    fn pow(&self, rhs: Expr) -> Expr {
        self.binary(Pow {}, rhs)
    }

    fn shape(&self) -> Expr {
        self.unary(Shape {
            start: None,
            end: None,
        })
    }

    fn slice(&self, starts: Expr, ends: Expr) -> Expr {
        self.apply(Slice {}, &[starts, ends], &[OutputMeta::NoMeta])
    }

    fn sigmoid(&self) -> Expr {
        self.unary(Sigmoid {})
    }

    fn square(&self) -> Expr {
        self.binary(Pow {}, Expr::constant(2.0))
    }

    fn sqrt(&self) -> Expr {
        self.unary(Sqrt {})
    }

    fn softmax(&self, axis: isize) -> Expr {
        self.unary(Softmax { axis })
    }

    fn tanh(&self) -> Expr {
        self.unary(Tanh {})
    }

    fn transpose(&self) -> Expr {
        self.unary(Transpose { perm: None })
    }
}

trait GetConsumingOp {
    /// Get the operator node that consumes a value.
    fn get_consuming_op(&self, value: NodeId) -> Option<&OperatorNode>;
}

impl GetConsumingOp for Graph {
    fn get_consuming_op(&self, value: NodeId) -> Option<&OperatorNode> {
        self.get_consumers(value)
            .and_then(|c| c.first())
            .and_then(|consumer_id| self.get_node(*consumer_id))
            .and_then(|node| node.as_operator())
    }
}

#[test]
fn test_convert_captured_values_to_constants() -> Result<(), Box<dyn Error>> {
    let mut graph = Graph::new();

    // Add a ref-counted constant value that can be cheaply cloned.
    let const_tensor = arc_tensor_view(42.);
    graph.add_constant(Some("const_a"), const_tensor);

    // Capture the constant in the subgraph as a value.
    let mut subgraph = Graph::new();
    let sg_val = subgraph.add_value(Some("const_a"), None, None);
    subgraph.set_captures(&[sg_val]);
    subgraph.set_output_ids(&[sg_val]);

    // Run optimizations on the subgraph. This should replace the captured
    // value with a local constant that references the same data.
    let optimizer = GraphOptimizer::new();
    let capture_env = CaptureEnv::top_level_static(&graph);
    let optimized_subgraph =
        optimizer.optimize(subgraph, Some(&capture_env), OptimizeOptions::default())?;

    let outputs = optimized_subgraph.output_ids();
    assert!(optimized_subgraph.captures().is_empty());
    assert_eq!(outputs.len(), 1);
    let node = optimized_subgraph.get_node(outputs[0]).unwrap();
    assert_eq!(node.name(), Some("const_a"));
    assert!(matches!(node, Node::Constant(_)));

    Ok(())
}

#[test]
fn test_constant_propagation() -> Result<(), Box<dyn Error>> {
    let mut graph = Graph::new();

    // Add an operator with constant inputs.
    let const_a = graph.add_constant(Some("const_a"), Tensor::from([1, 2, 3]));
    let const_b = graph.add_constant(Some("const_b"), Tensor::from([4, 5, 6]));
    let (_, add_out) = graph.add_simple_op("add_1", Add {}, &[const_a, const_b]);

    // Add an operator with a dynamic input and the output of the previous operator.
    let input = graph.add_value(Some("input"), None, None);
    let (add_op_2, add_2_out) = graph.add_simple_op("add_2", Add {}, &[add_out, input]);
    graph.set_input_ids(&[input]);
    graph.set_output_ids(&[add_out, add_2_out]);

    // Optimize the graph. This should replace the first operator's output
    // with a constant value.
    let optimizer = GraphOptimizer::new();
    let optimized_graph = optimizer.optimize(graph, None, OptimizeOptions::default())?;

    // Check that we got the expected inputs and outputs. The optimizer
    // does not promise to preserve IDs for unmodified parts of the graph,
    // but the current implementation does.
    assert_eq!(optimized_graph.input_ids(), &[input]);
    assert_ne!(optimized_graph.output_ids()[0], add_out);
    assert_eq!(optimized_graph.output_ids()[1], add_2_out);

    // Check first output was replaced with constant.
    let replaced_node = optimized_graph
        .get_node(optimized_graph.output_ids()[0])
        .and_then(|n| match &n {
            Node::Constant(c) => Some(c),
            _ => None,
        })
        .unwrap();
    let Constant::Int32(const_int) = replaced_node else {
        return Err("constant not an int".into());
    };
    assert_eq!(const_int.view(), Tensor::from([5, 7, 9]));

    // Check input to second operator was replaced with constant.
    let op = optimized_graph
        .get_node(add_op_2)
        .and_then(|n| match &n {
            Node::Operator(op) => Some(op),
            _ => None,
        })
        .unwrap();
    let input_ids: Vec<_> = op.input_ids().iter().map(|id| id.unwrap()).collect();
    assert_eq!(input_ids.len(), 2);
    assert_ne!(input_ids[0], add_out);
    assert_eq!(input_ids[0], optimized_graph.output_ids()[0]);
    assert_eq!(input_ids[1], input);

    Ok(())
}

#[test]
fn test_fuse_op_with_duplicate_inputs() {
    // We use MatMul + Add fusion for this test, but any fused op that
    // takes multiple non-constant inputs would work.
    let graph = {
        let x = Expr::value("x");
        let bias = Tensor::from([1., 2., 3.]);
        let expr = x.matmul(x.clone()) + bias;
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "FusedMatMul");
}

#[test]
fn test_fuse_op_with_captured_input() {
    // Create subgraph for Silu operation which takes input from capture
    // rather than a regular input.
    let mut subgraph = {
        let x = Expr::value("x");
        let expr = x.clone() * x.sigmoid();
        expr.build_graph([])
    };
    let x_id = subgraph.get_node_id("x").unwrap();
    subgraph.set_captures(&[x_id]);

    let graph = Graph::new();
    let capture_env = CaptureEnv::top_level_static(&graph);
    let optimized_subgraph = GraphOptimizer::new()
        .optimize(subgraph, Some(&capture_env), OptimizeOptions::default())
        .unwrap();

    let (_, op) = optimized_subgraph
        .get_source_node(optimized_subgraph.output_ids()[0])
        .unwrap();
    assert_eq!(op.operator().name(), "Silu");
}

#[test]
fn test_fuse_transpose() {
    let graph = {
        let x = Expr::value("x");
        let y = Expr::value("y");
        x.transpose().matmul(y.transpose()).build_graph(["x", "y"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "TransformInputs(MatMul)");
    assert_eq!(
        op.input_ids(),
        graph
            .input_ids()
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_fuse_silu() {
    let graph = {
        let x = Expr::value("x");
        let expr = x.clone() * x.sigmoid();
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "Silu");
}

#[test]
fn test_fuse_swish() {
    let graph = {
        let x = Expr::value("x");
        let beta = 1.7;
        let expr = x.clone() * (x.clone() * beta).sigmoid();
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let swish_op = op.operator().downcast_ref::<Swish>().unwrap();
    assert_eq!(swish_op.beta, 1.7);
}

#[test]
fn test_fuse_matmul_add() {
    let graph = {
        let a = Expr::value("a");
        let b = Expr::value("b");
        let bias = Tensor::from([1., 2., 3.]);
        let expr = a.matmul(b) + bias;
        expr.build_graph(["a", "b"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "FusedMatMul");
}

#[test]
fn test_fuse_matmul_scaled() {
    // Pattern 1: MatMul(Mul(A, c), Mul(B, d)). This has scale applied to
    // inputs via `Mul` ops.
    let graph = {
        let a = Expr::value("a");
        let b = Expr::value("b");
        let expr = (a * 0.5).matmul(b * 0.3);
        expr.build_graph(["a", "b"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "FusedMatMul");
    let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
    assert_eq!(fused_matmul_op.alpha, Some(0.5 * 0.3));

    // Pattern 2: Div(MatMul(A, B), c). This has scale applied to outputs
    // via `Div` ops.
    let graph = {
        let a = Expr::value("a");
        let b = Expr::value("b");
        let expr = a.matmul(b) / 0.5;
        expr.build_graph(["a", "b"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "FusedMatMul");
    let fused_matmul_op = op.operator().downcast_ref::<FusedMatMul>().unwrap();
    assert_eq!(fused_matmul_op.alpha, Some(1. / 0.5));
}

#[test]
fn test_chained_fused_ops() {
    // Two consecutive decomposed Silu operations
    let graph = {
        let x = Expr::value("x");
        let y = x.clone() * x.sigmoid();
        let z = y.clone() * y.sigmoid();
        z.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    // Check that both ops were fused. This requires that the inputs to the
    // second group of fused nodes are updated after fusing the first.
    let (_, fused_op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(fused_op.operator().name(), "Silu");
    let (_, fused_op_2) = graph
        .get_source_node(fused_op.input_ids()[0].unwrap())
        .unwrap();
    assert_eq!(fused_op_2.operator().name(), "Silu");
}

#[test]
fn test_fuse_gelu() {
    let graph = {
        let x = Expr::value("x");
        let sqrt_2 = (2.0f32).sqrt();
        let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let gelu = op.operator().downcast_ref::<Gelu>().unwrap();
    assert_eq!(gelu.approximate, false);
}

#[test]
fn test_fuse_approx_gelu() {
    let graph = {
        let x = Expr::value("x");
        let sqrt_2_pi = Expr::constant((2.0f32 / std::f32::consts::PI).sqrt());
        let expr = x.clone()
            * 0.5
            * (Expr::constant(1.)
                + (sqrt_2_pi * (x.clone() + x.pow(Expr::constant(3.0)) * 0.044715)).tanh());
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let gelu = op.operator().downcast_ref::<Gelu>().unwrap();
    assert_eq!(gelu.approximate, true);
}

fn layer_norm_graph(with_bias: bool) -> Graph {
    // Center mean
    let epsilon = 1e-6;
    let x = Expr::value("x");
    let x_mean = x.mean();
    let x_sub_mean = x.clone() - x_mean;

    // Normalize variance
    let normalized = x_sub_mean.clone() / (x_sub_mean.square().mean() + epsilon).sqrt();

    // Shift and scale result
    let scale = Tensor::from([3., 4., 5.]);
    let expr = if with_bias {
        let bias = Tensor::from([1., 2., 3.]);
        normalized * scale + bias
    } else {
        normalized * scale
    };
    expr.build_graph(["x"])
}

#[test]
fn test_fuse_layer_norm() {
    #[derive(Debug)]
    struct Case {
        with_bias: bool,
    }

    let cases = [Case { with_bias: true }, Case { with_bias: false }];

    cases.test_each(|&Case { with_bias }| {
        let graph = layer_norm_graph(with_bias);
        let graph = optimize_graph(graph).unwrap();
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "LayerNormalization");

        let layer_norm = op.operator().downcast_ref::<LayerNormalization>().unwrap();
        assert_eq!(layer_norm.epsilon, Some(1e-6));
        let bias_input = op.input_ids().get(2).copied().flatten();
        assert_eq!(bias_input.is_some(), with_bias);
    })
}

#[test]
fn test_fuse_rms_norm() {
    // See https://arxiv.org/pdf/1910.07467
    let graph = {
        let x = Expr::value("x");
        let epsilon = 1e-6;
        let rms = (x.square().mean() + epsilon).sqrt();
        let scale = Tensor::from([3., 4., 5.]);
        let expr = x * (Expr::constant(1.) / rms) * scale;
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let rms_norm = op.operator().downcast_ref::<RmsNormalization>().unwrap();
    assert_eq!(rms_norm.epsilon, Some(1e-6));
}

#[test]
fn test_fuse_rms_norm_with_positive_axes() {
    let graph = {
        let dims = &[
            Dimension::Symbolic("batch".to_string()),
            Dimension::Symbolic("seq".to_string()),
            Dimension::Fixed(16),
        ];
        let x = Expr::value_with_info("x", DataType::Float, dims);

        // axes is specifies as a positive value, so must be resolved
        // against the input's shape information.
        let axes = Expr::constant(Tensor::from([2i32]));

        let epsilon = 1e-6;
        let x_square = x.apply(
            Pow {},
            &[Expr::constant(2.0)],
            // Add shape info to Pow(X, 2) output so ReduceMean can verify that
            // `axes` refers to the last axis.
            &[OutputMeta::Meta((DataType::Float, dims.to_vec()))],
        );
        let rms = (x_square.mean_axes(axes) + epsilon).sqrt();
        let scale = Tensor::from([3., 4., 5.]);
        let expr = x * (Expr::constant(1.) / rms) * scale;
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let rms_norm = op.operator().downcast_ref::<RmsNormalization>().unwrap();
    assert_eq!(rms_norm.epsilon, Some(1e-6));
}

#[test]
fn test_fuse_add_softmax() {
    let graph = {
        let qk = Expr::value("qk");
        let m = Expr::value("m");
        let expr = (qk + m).softmax(-1);
        expr.build_graph(["qk", "m"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "AddSoftmax");
}

#[test]
fn test_fuse_add_softmax_positive_axes() {
    let graph = {
        let dims = [
            Dimension::Symbolic("batch".to_string()),
            Dimension::Fixed(768),
        ];
        let qk = Expr::value_with_info("qk", DataType::Float, &dims);
        let m = Expr::value("m");
        let expr = qk
            // Add shape info so optimizer can determine softmax is applied
            // to last axis.
            .apply(
                Add {},
                &[m],
                &[OutputMeta::Meta((DataType::Float, dims.to_vec()))],
            )
            .softmax(1);
        expr.build_graph(["qk", "m"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "AddSoftmax");
}

#[test]
fn test_fuse_reciprocal() {
    let graph = {
        let x = Expr::value("x");
        let expr = Expr::constant(1.) / x;
        expr.build_graph(["x"])
    };
    let graph = optimize_graph(graph).unwrap();
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "Reciprocal");
}

#[test]
fn test_fuse_reduce_mean_axes() {
    let graph = {
        let x = Expr::value("x");
        let axes = Expr::constant(Tensor::from([-1i32]));
        x.mean_axes(axes).build_graph(["x"])
    };
    let graph = optimize_graph(graph).unwrap();
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let mean_op = op.operator().downcast_ref::<ReduceMean>().unwrap();
    assert_eq!(mean_op.axes.as_deref(), Some([-1].as_slice()));
}

#[test]
fn test_fuse_identity_op() {
    struct Case {
        expr: Expr,
    }

    let cases = [
        Case {
            expr: (Expr::value("x") + 0.),
        },
        Case {
            expr: (Expr::value("x") - 0.),
        },
        Case {
            expr: (Expr::value("x") * 1.),
        },
        Case {
            expr: (Expr::value("x") / 1.),
        },
    ];

    for case in cases {
        let graph = optimize_graph(case.expr.build_graph(["x"])).unwrap();
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();

        // No-op connected to graph output should be replaced by `Identity` op.
        assert_eq!(op.operator().name(), "Identity");
    }
}

#[test]
fn test_eliminate_binary_identity_pattern() {
    let graph = {
        let x = Expr::value("x");
        let expr = x * 1. + 2.;
        expr.build_graph(["x"])
    };
    let input_id = graph.input_ids()[0];
    let output_id = graph.output_ids()[0];
    let mul_op = graph.get_consumers(input_id).unwrap()[0];
    assert_eq!(
        graph
            .get_node(mul_op)
            .unwrap()
            .as_operator()
            .unwrap()
            .operator()
            .name(),
        "Mul"
    );

    let graph = optimize_graph(graph).unwrap();

    // Optimization should not change input/output IDs.
    assert_eq!(graph.input_ids(), [input_id]);
    assert_eq!(graph.output_ids(), [output_id]);

    let (_, op) = graph.get_source_node(output_id).unwrap();
    assert_eq!(op.operator().name(), "Add");

    // Outputs of identity nodes should be replaced by inputs to identity nodes
    // in other operators.
    assert_eq!(op.input_ids()[0], Some(input_id));

    // Identity nodes should be removed from the graph.
    assert!(graph.get_node(mul_op).is_none());
}

#[test]
fn test_eliminate_unary_identity_pattern() {
    let graph = {
        let x = Expr::value("x");
        x.identity().sqrt().build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    // Verify that Identity operation was eliminated.
    let input_id = graph.input_ids()[0];
    assert_eq!(
        graph.get_consuming_op(input_id).unwrap().operator().name(),
        "Sqrt"
    );
}

#[test]
fn test_eliminate_noop_cast() {
    let graph = {
        let x = Expr::value_with_info(
            "x",
            DataType::Float,
            &[Dimension::Symbolic("x".to_string())],
        );
        x.cast(DataType::Float).erf().build_graph(["x"])
    };
    let graph = optimize_graph(graph).unwrap();

    let input_id = graph.input_ids()[0];

    // Verify that Cast operation was eliminated.
    assert_eq!(
        graph.get_consuming_op(input_id).unwrap().operator().name(),
        "Erf"
    );
}

#[test]
fn test_fuse_matmulinteger_cast_scale() {
    let graph = {
        let x = Expr::value("x");
        let weights = Expr::constant(Tensor::<i8>::zeros(&[4, 4]));
        let weights_zero = Expr::constant(Tensor::<i8>::zeros(&[4]));

        let quant = x.apply(
            DynamicQuantizeLinear {},
            &[],
            &[OutputMeta::NoMeta, OutputMeta::NoMeta, OutputMeta::NoMeta],
        );
        let quant_x = quant.output(0);
        let quant_scale = quant.output(1);
        let quant_zero = quant.output(2);
        let const_scale = Expr::constant(Tensor::from([0.1, 0.2, 0.3]));

        let expr = quant_x
            .apply(
                MatMulInteger {},
                &[weights, quant_zero, weights_zero],
                &[OutputMeta::NoMeta],
            )
            .unary(Cast {
                to: DataType::Float,
            })
            * (quant_scale * const_scale);
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();

    assert_eq!(op.operator().name(), "MatMulIntegerToFloat");
}

#[test]
fn test_slice_shape_to_constant() {
    // `Slice(Shape(X), starts, ends)` which can be simplified to a constant.
    let graph = {
        let x = Expr::value_with_info(
            "x",
            DataType::Float,
            &[Dimension::Symbolic("batch".into()), Dimension::Fixed(64)],
        );
        let starts = Expr::constant(Tensor::from([-1i32]));
        let ends = Expr::constant(Tensor::from([i32::MAX]));
        let expr = x.shape().slice(starts, ends);
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let id_input = graph.output_ids()[0];
    let const_node = graph
        .get_node(id_input)
        .and_then(|n| n.as_constant())
        .unwrap();
    assert_eq!(const_node.as_scalar(), Some(64i32));
}

#[test]
fn test_optimize_preserves_input_output_nodes() {
    // Fuse-able Transpose + MatMul
    let graph = {
        let x = Expr::value("x");
        let y = Expr::value("y");
        x.transpose().matmul(y).build_graph(["x", "y"])
    };
    let orig_input_ids = graph.input_ids().to_vec();
    let orig_output_ids = graph.output_ids().to_vec();

    let graph = optimize_graph(graph).unwrap();

    // Verify that optimizer did change the graph
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "TransformInputs(MatMul)");

    // The IDs of the input and output nodes should be the same after
    // optimization.
    //
    // The optimizer could have created new output nodes instead, but it
    // would need to ensure that the new outputs preserved value node
    // metadata (name, shape) from the original outputs.
    assert_eq!(graph.input_ids(), orig_input_ids);
    assert_eq!(graph.output_ids(), orig_output_ids);
}

#[test]
fn test_optimize_error() {
    let mut graph = Graph::new();
    let optimizer = GraphOptimizer::new();
    let invalid_id = NodeId::from_u32(123);
    graph.set_input_ids(&[invalid_id]);
    graph.set_output_ids(&[invalid_id]);
    let result = optimizer.optimize(graph, None, OptimizeOptions::default());
    assert!(matches!(result, Err(OptimizeError::RunError(_))));
}

#[test]
fn test_optimize_removes_unfused_ops() {
    let graph = {
        let x = Expr::value("x");
        let sqrt_2 = (2.0f32).sqrt();
        let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
        expr.build_graph(["x"])
    };

    // Intermediate nodes of Gelu op. These should be removed after fusion.
    let ops = ["Mul", "Erf", "Div", "Add"];
    for op in ops {
        assert!(graph.get_node_id(op).is_some());
    }

    let optimized = optimize_graph(graph).unwrap();
    for op in ops {
        assert!(optimized.get_node_id(op).is_none());
    }
    let fused_op = optimized
        .get_node_id("Mul_1")
        .and_then(|id| optimized.get_node(id))
        .and_then(|n| n.as_operator())
        .unwrap();
    assert_eq!(fused_op.operator().name(), "Gelu");
}

#[test]
fn test_optimize_does_not_fuse_if_intermediate_outputs_reused() {
    let mut graph = {
        let x = Expr::value("x");
        let sqrt_2 = (2.0f32).sqrt();
        let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
        expr.build_graph(["x"])
    };

    // Add an operator which reuses an intermediate value from the Gelu
    // subgraph. This should prevent fusion.
    let erf_out = graph.get_node_id("Erf_out").unwrap();
    graph.add_simple_op("neg", Neg {}, &[erf_out]);

    let optimized = optimize_graph(graph).unwrap();
    let fused_op = optimized
        .get_node_id("Mul_1")
        .and_then(|id| optimized.get_node(id))
        .and_then(|n| n.as_operator())
        .unwrap();
    assert_eq!(fused_op.operator().name(), "Mul");
}

#[test]
fn test_fuse_transpose_matmul_scaled() {
    // Div(MatMul(Transpose(X), Transpose(Y)), C) subgraph.
    //
    // This should be fused into a single operation in two stages:
    //  1. Fuse Div + MatMul into `FusedMatMul(Transpose(X), Transpose(Y), alpha=C)`
    //  2. Fuse FusedMatMul + Transpose into `TransformInputs(FusedMatMul(X, Y, alpha=C))`.
    let x = Expr::value("x");
    let y = Expr::value("y");
    let xy = x.transpose().matmul(y.transpose());
    let xy_scaled = xy / 8.;
    let graph = xy_scaled.build_graph(["x", "y"]);

    let input_ids = graph.input_ids().to_vec();
    let output_ids = graph.output_ids().to_vec();
    let optimized = optimize_graph(graph).unwrap();
    let plan = optimized
        .execution_plan(&input_ids, &output_ids, PlanOptions::default())
        .unwrap();

    let op_name = |node_id| {
        optimized
            .get_node(node_id)
            .and_then(|n| n.as_operator())
            .map(|op| op.operator().name())
    };

    assert_eq!(plan.len(), 1);
    assert_eq!(op_name(plan[0]), Some("TransformInputs(FusedMatMul)"));
}

#[test]
fn test_optimize_does_not_fuse_if_intermediate_output_is_graph_output() {
    let mut graph = {
        let x = Expr::value("x");
        let sqrt_2 = (2.0f32).sqrt();
        let expr = x.clone() * ((x / sqrt_2).erf() + 1.0) * 0.5;
        expr.build_graph(["x"])
    };

    // Add intermediate output as an output of the graph. This should
    // prevent fusion.
    let erf_out = graph.get_node_id("Erf_out").unwrap();
    let mut output_ids = graph.output_ids().to_vec();
    output_ids.push(erf_out);
    graph.set_output_ids(&output_ids);

    let optimized = optimize_graph(graph).unwrap();

    let fused_op = optimized
        .get_node_id("Mul_1")
        .and_then(|id| optimized.get_node(id))
        .and_then(|n| n.as_operator())
        .unwrap();
    assert_eq!(fused_op.operator().name(), "Mul");
}

use std::error::Error;
use std::sync::Arc;

use rten_base::byte_cast::cast_slice;
use rten_shape_inference::{SymExpr, Symbol};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor, TensorView};
use rten_testing::TestCases;

use super::{
    DynamicQuantConfig, DynamicQuantOp, DynamicQuantOptions, GraphOptimizer, NamePattern,
    OptimizeError, OptimizeOptions,
};
use crate::Dimension;
use crate::constant_storage::{ArcSlice, ArcTensorView, ConstantStorage};
use crate::graph::builder::{Expr, OutputMeta, dims};
use crate::graph::{
    CaptureEnv, Constant, Graph, Node, NodeId, OperatorNode, PlanOptions, TypedConstant,
};
use crate::infer_shapes::InferShapeOptions;
use crate::ops::{
    Add, BatchNormalization, Cast, ComputeShape, Conv, ConvInteger, ConvTranspose, Cos,
    DequantizeLinear, Dst, DynamicQuantizeLinear, Erf, Expand, FusedMatMul, Gather, Gelu,
    GroupedQueryAttentionMatMul, Identity, If, IsNaN, LayerNormalization, MatMul, MatMulInteger,
    Neg, Padding, Pass, Pow, RMSNormalization, ReduceMean, RegId, RepeatInterleave, Reshape,
    ScalarOp, Shape, Sigmoid, Sin, Slice, Softmax, Softplus, Split, Sqrt, Src, Swish, Tanh, TileOp,
    TileProgram, Transpose, Unsqueeze, Where,
};
use crate::value::{DataType, Value, ValueType};

fn optimize_graph(graph: Graph) -> Result<Graph, OptimizeError> {
    let optimizer = GraphOptimizer::new();
    optimizer.optimize(graph, None, OptimizeOptions::default())
}

fn optimize_graph_infer_shapes(graph: Graph) -> Result<Graph, OptimizeError> {
    let optimizer = GraphOptimizer::new();
    optimizer.optimize(
        graph,
        None,
        OptimizeOptions {
            infer_shapes: Some(InferShapeOptions {
                strict: false,
                ..Default::default()
            }),
            ..Default::default()
        },
    )
}

fn arc_tensor_view(val: f32) -> ArcTensorView<f32> {
    let const_data = Vec::from(val.to_le_bytes());
    let const_storage = Arc::new(ConstantStorage::Buffer(const_data));
    let slice = ArcSlice::new(
        const_storage.clone(),
        cast_slice(const_storage.data()).unwrap(),
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
    fn is_nan(&self) -> Expr;
    fn permute(&self, perm: &[usize]) -> Expr;
    fn pow(&self, rhs: Expr) -> Expr;
    fn matmul(&self, rhs: Expr) -> Expr;
    fn mean(&self) -> Expr;
    fn mean_axes(&self, axes: Expr) -> Expr;
    fn shape(&self) -> Expr;
    fn sigmoid(&self) -> Expr;
    fn slice(&self, starts: Expr, ends: Expr) -> Expr;
    fn square(&self) -> Expr;
    fn sin(&self) -> Expr;
    fn sqrt(&self) -> Expr;
    fn softmax(&self, axis: isize) -> Expr;
    fn tanh(&self) -> Expr;
    fn transpose(&self) -> Expr;
    fn where_(&self, if_true: Expr, if_false: Expr) -> Expr;
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

    fn is_nan(&self) -> Expr {
        self.unary(IsNaN {})
    }

    fn matmul(&self, rhs: Expr) -> Expr {
        self.binary(MatMul {}, rhs)
    }

    fn mean(&self) -> Expr {
        self.unary(ReduceMean {
            axes: Some(vec![-1]),
            keep_dims: false,
            noop_with_empty_axes: false,
        })
    }

    fn mean_axes(&self, axes: Expr) -> Expr {
        self.binary(
            ReduceMean {
                axes: None,
                keep_dims: false,
                noop_with_empty_axes: false,
            },
            axes,
        )
    }

    fn permute(&self, perm: &[usize]) -> Expr {
        self.unary(Transpose {
            perm: Some(perm.to_vec()),
        })
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

    fn sin(&self) -> Expr {
        self.unary(Sin {})
    }

    fn sqrt(&self) -> Expr {
        self.unary(Sqrt {})
    }

    fn softmax(&self, axis: isize) -> Expr {
        self.unary(Softmax {
            axis,
            flush_nans_to_zero: false,
        })
    }

    fn tanh(&self) -> Expr {
        self.unary(Tanh {})
    }

    fn transpose(&self) -> Expr {
        self.unary(Transpose { perm: None })
    }

    fn where_(&self, if_true: Expr, if_false: Expr) -> Expr {
        self.apply(Where {}, &[if_true, if_false], &[OutputMeta::NoMeta])
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
    let const_a = graph.add_constant(Some("const_a"), Tensor::from([1, 2, 3]).into_arc());
    let const_b = graph.add_constant(Some("const_b"), Tensor::from([4, 5, 6]).into_arc());
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
fn test_remove_dead_nodes_after_constant_propagation() {
    let mut graph = Graph::new();

    // Add an operator with constant inputs, which will be replaced by a
    // constant.
    let const_a = graph.add_constant(Some("const_a"), Tensor::from([1, 2, 3]).into_arc());
    let const_b = graph.add_constant(Some("const_b"), Tensor::from([4, 5, 6]).into_arc());
    let (add_op_1, add_out) = graph.add_simple_op("add_1", Add {}, &[const_a, const_b]);

    // Add an operator with a dynamic input.
    let input = graph.add_value(Some("input"), None, None);
    let (add_op_2, add_2_out) = graph.add_simple_op("add_2", Add {}, &[add_out, input]);

    // Add a constant which is unused in the original graph.
    let unused_const = graph.add_constant(Some("unused"), Tensor::from([7, 8, 9]).into_arc());

    graph.set_input_ids(&[input]);
    graph.set_output_ids(&[add_2_out]);

    let optimized_graph = optimize_graph(graph).unwrap();

    // The constant-folded operator, its output value and its constant inputs
    // should be removed.
    for id in [const_a, const_b, add_op_1, add_out] {
        assert!(optimized_graph.get_node(id).is_none());
    }

    // The replacement constant should be findable by the name of the value
    // it replaced.
    let folded_id = optimized_graph.get_node_id("add_1_out").unwrap();
    assert_ne!(folded_id, add_out);
    assert!(matches!(
        optimized_graph.get_node(folded_id),
        Some(Node::Constant(_))
    ));

    // Live operators, graph inputs and constants that were unused in the
    // original graph should be preserved.
    for id in [add_op_2, add_2_out, input, unused_const] {
        assert!(optimized_graph.get_node(id).is_some());
    }
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
fn test_fuse_tile_program() {
    // Operations which compute `x + beta * sin(alpha * x)^2` using one
    // register.
    let snake_program = |inner_dims: usize| {
        let reg = Dst::Reg(RegId(0));
        let reg_src = Src::Reg(RegId(0));
        let ops = vec![
            TileOp::MulScalar {
                dst: reg,
                src: Src::Input(0),
                scalar: 0,
            },
            TileOp::SinSquared {
                dst: reg,
                src: reg_src,
            },
            TileOp::MulScalarAdd {
                dst: Dst::Output(0),
                src: reg_src,
                scalar: 1,
                addend: Src::Input(0),
            },
        ];
        TileProgram::new(vec![ops.into()], 1, 2, inner_dims)
    };

    #[derive(Debug)]
    struct Case {
        // Shape of the `alpha` and `beta` constants.
        param_shape: Vec<usize>,
        // Shape of the input, if known.
        input_shape: Option<Vec<Dimension>>,
        // True if `sin(alpha * x)` is also a graph output.
        sin_is_output: bool,
        // Expected program, or `None` if the output should not be fused.
        expected: Option<TileProgram>,
    }

    let cases = [
        // Per-channel parameters
        Case {
            param_shape: [1, 3, 1].into(),
            input_shape: Some(dims!("batch", 3, "time").into()),
            sin_is_output: false,
            expected: Some(snake_program(1)),
        },
        // Per-channel parameters with fewer dimensions than the input
        Case {
            param_shape: [3, 1].into(),
            input_shape: Some(dims!("batch", 3, "time").into()),
            sin_is_output: false,
            expected: Some(snake_program(1)),
        },
        // Parameters which are the same for all channels
        Case {
            param_shape: [].into(),
            input_shape: Some(dims!("batch", 3, "time").into()),
            sin_is_output: false,
            expected: Some(snake_program(3)),
        },
        // Input with unknown shape
        Case {
            param_shape: [1, 3, 1].into(),
            input_shape: None,
            sin_is_output: false,
            expected: None,
        },
        // Parameters which vary along the time axis
        Case {
            param_shape: [1, 1, 5].into(),
            input_shape: Some(dims!("batch", 3, 5).into()),
            sin_is_output: false,
            expected: None,
        },
        // Rows which are known to be short
        Case {
            param_shape: [1, 3, 1].into(),
            input_shape: Some(dims!("batch", 3, 1).into()),
            sin_is_output: false,
            expected: None,
        },
        // Sine which is also a graph output. It becomes a second output of
        // the program, and is not combined with the squaring.
        Case {
            param_shape: [1, 3, 1].into(),
            input_shape: Some(dims!("batch", 3, "time").into()),
            sin_is_output: true,
            expected: Some(TileProgram::new(
                vec![
                    vec![
                        TileOp::MulScalar {
                            dst: Dst::Reg(RegId(0)),
                            src: Src::Input(0),
                            scalar: 0,
                        },
                        TileOp::Sin {
                            dst: Dst::Output(1),
                            src: Src::Reg(RegId(0)),
                        },
                        TileOp::Square {
                            dst: Dst::Reg(RegId(0)),
                            src: Src::Output(1),
                        },
                        TileOp::MulScalarAdd {
                            dst: Dst::Output(0),
                            src: Src::Reg(RegId(0)),
                            scalar: 1,
                            addend: Src::Input(0),
                        },
                    ]
                    .into(),
                ],
                1,
                2,
                1,
            )),
        },
    ];

    cases.test_each(|case| {
        let n_params: usize = case.param_shape.iter().product();
        let alpha = Tensor::from_data(
            &case.param_shape,
            (0..n_params).map(|i| 0.5 + i as f32).collect::<Vec<_>>(),
        );
        let beta = alpha.map(|a| 1. / a);

        let graph = {
            let x = match &case.input_shape {
                Some(shape) => {
                    Expr::value_with_info("x", ValueType::Tensor(DataType::Float), shape)
                }
                None => Expr::value("x"),
            };
            let sin = (Expr::constant(alpha) * x.clone()).sin();
            let expr = x.clone() + Expr::constant(beta) * sin.square();
            if case.sin_is_output {
                Expr::make_graph([x], [expr, sin])
            } else {
                Expr::make_graph([x], [expr])
            }
        };

        let graph = optimize_graph_infer_shapes(graph).unwrap();
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();

        if let Some(expected) = &case.expected {
            let op = op
                .operator()
                .downcast_ref::<TileProgram>()
                .expect("output should be fused");
            assert_eq!(op, expected);
        } else {
            assert_eq!(op.operator().name(), "Add");
        }
    })
}

#[test]
fn test_fuse_tile_program_instance_norm() {
    let graph = {
        let float = ValueType::Tensor(DataType::Float);
        let x = Expr::value_with_info("x", float, &dims!("batch", 3, "time"));
        let gamma = Expr::value_with_info("gamma", float, &dims!("batch", 3, 1));
        let beta = Expr::value_with_info("beta", float, &dims!("batch", 3, 1));
        let mean = |x: &Expr| {
            x.unary(ReduceMean {
                axes: Some(vec![-1]),
                keep_dims: true,
                noop_with_empty_axes: false,
            })
        };
        let centered = x.clone() - mean(&x);
        let std = (mean(&centered.square()) + Expr::constant(1e-5)).sqrt();
        let expr = centered / std * gamma.clone() + beta.clone();
        Expr::make_graph([x, gamma, beta], [expr])
    };

    let graph = optimize_graph_infer_shapes(graph).unwrap();
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let op = op
        .operator()
        .downcast_ref::<TileProgram>()
        .expect("output should be fused");

    // Scalars are the inputs (epsilon, gamma, beta) followed by the mean,
    // variance, variance + epsilon, standard deviation and its reciprocal.
    let reg = Dst::Reg(RegId(0));
    let reg_src = Src::Reg(RegId(0));
    let x = Src::Input(0);
    let expected = TileProgram::new(
        vec![
            vec![TileOp::Sum { src: x, scalar: 3 }].into(),
            Pass {
                scalar_ops: vec![ScalarOp::DivRowLen { scalar: 3 }].into(),
                tile_ops: vec![
                    TileOp::SubScalar {
                        dst: reg,
                        src: x,
                        scalar: 3,
                    },
                    TileOp::Square {
                        dst: reg,
                        src: reg_src,
                    },
                    TileOp::Sum {
                        src: reg_src,
                        scalar: 4,
                    },
                ]
                .into(),
            },
            Pass {
                scalar_ops: vec![
                    ScalarOp::DivRowLen { scalar: 4 },
                    ScalarOp::Add { dst: 5, a: 4, b: 0 },
                    ScalarOp::Sqrt { dst: 6, src: 5 },
                    ScalarOp::Reciprocal { dst: 7, src: 6 },
                ]
                .into(),
                tile_ops: vec![
                    TileOp::SubScalar {
                        dst: reg,
                        src: x,
                        scalar: 3,
                    },
                    TileOp::MulScalar {
                        dst: reg,
                        src: reg_src,
                        scalar: 7,
                    },
                    TileOp::MulScalar {
                        dst: reg,
                        src: reg_src,
                        scalar: 1,
                    },
                    TileOp::AddScalar {
                        dst: Dst::Output(0),
                        src: reg_src,
                        scalar: 2,
                    },
                ]
                .into(),
            },
        ],
        1,
        3,
        1,
    );
    assert_eq!(*op, expected);
}

#[test]
fn test_fuse_tile_program_multiple_outputs() {
    // Convert magnitudes and phases to real and imaginary parts.
    let graph = {
        let float = ValueType::Tensor(DataType::Float);
        let shape = dims!("batch", 11, "time");
        let mag_in = Expr::value_with_info("mag", float, &shape);
        let phase_in = Expr::value_with_info("phase", float, &shape);
        let mag = mag_in.unary(Softplus {});
        let phase = phase_in.sin();
        let real = mag.clone() * phase.unary(Cos {});
        let imag = mag * phase.sin();
        Expr::make_graph([mag_in, phase_in], [real, imag])
    };

    let graph = optimize_graph_infer_shapes(graph).unwrap();
    let ops: Vec<_> = graph
        .output_ids()
        .iter()
        .map(|id| graph.get_source_node(*id).unwrap())
        .collect();
    assert_eq!(ops[0].0, ops[1].0);
    let op = ops[0]
        .1
        .operator()
        .downcast_ref::<TileProgram>()
        .expect("outputs should be fused");

    // The phase is input 0 and the magnitude is input 1. The imaginary part
    // is output 0, which is written last, and the real part is output 1.
    let expected = TileProgram::new(
        vec![
            vec![
                TileOp::Sin {
                    dst: Dst::Reg(RegId(0)),
                    src: Src::Input(0),
                },
                TileOp::Cos {
                    dst: Dst::Reg(RegId(1)),
                    src: Src::Reg(RegId(0)),
                },
                TileOp::Softplus {
                    dst: Dst::Reg(RegId(2)),
                    src: Src::Input(1),
                },
                TileOp::Mul {
                    dst: Dst::Output(1),
                    a: Src::Reg(RegId(2)),
                    b: Src::Reg(RegId(1)),
                },
                TileOp::Sin {
                    dst: Dst::Reg(RegId(0)),
                    src: Src::Reg(RegId(0)),
                },
                TileOp::Mul {
                    dst: Dst::Output(0),
                    a: Src::Reg(RegId(2)),
                    b: Src::Reg(RegId(0)),
                },
            ]
            .into(),
        ],
        2,
        0,
        3,
    );
    assert_eq!(*op, expected);

    // Check that each graph output is produced by the expected program
    // output.
    let op_node = ops[0].1;
    assert_eq!(
        op_node.output_ids(),
        [Some(graph.output_ids()[1]), Some(graph.output_ids()[0])]
    );
}

#[test]
fn test_fuse_tile_program_excludes_self_dependency() {
    // `Add` depends on `Sin` both directly and via an operator which can't
    // be part of a program. If both were fused, the program would depend on
    // its own output.
    let graph = {
        let float = ValueType::Tensor(DataType::Float);
        let x = Expr::value_with_info("x", float, &dims!("batch", 3, "time"));
        let sin = (x * Expr::constant(2.0)).sin();
        let expr = sin.clone() + sin.tanh();
        expr.build_graph(["x"])
    };

    let graph = optimize_graph_infer_shapes(graph).unwrap();

    let (_, add_op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(add_op.operator().name(), "Add");
    let add_inputs: Vec<_> = add_op.input_ids().iter().copied().flatten().collect();
    let has_program_input = add_inputs.iter().any(|id| {
        graph
            .get_source_node(*id)
            .is_some_and(|(_, op)| op.operator().name() == "TileProgram")
    });
    assert!(has_program_input);
}

#[test]
fn test_fuse_swish() {
    let graph = {
        let x = Expr::value("x");
        let alpha = 1.7;
        let expr = x.clone() * (x.clone() * alpha).sigmoid();
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let swish_op = op.operator().downcast_ref::<Swish>().unwrap();
    assert_eq!(swish_op.alpha, 1.7);
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
fn test_fuse_conv_add() {
    #[derive(Debug)]
    struct Case {
        // Number of spatial dimensions: 1 for a 1D conv (`[N, C, L]`), 2 for a
        // 2D conv (`[N, C, H, W]`).
        spatial_dims: usize,
    }

    let cases = [Case { spatial_dims: 1 }, Case { spatial_dims: 2 }];

    cases.test_each(|&Case { spatial_dims }| {
        let out_channels = 4;
        let graph = {
            let x = Expr::value("x");

            // Weight shape is `[out_channels, in_channels, ...kernel]`.
            let mut weight_shape = vec![out_channels, 3];
            weight_shape.extend(std::iter::repeat_n(3, spatial_dims));
            let weight = Expr::constant(Tensor::<f32>::zeros(&weight_shape));

            let conv = x.apply(
                Conv {
                    groups: 1,
                    dilations: vec![1; spatial_dims],
                    padding: Padding::Same,
                    strides: vec![1; spatial_dims],
                },
                &[weight],
                &[OutputMeta::NoMeta],
            );

            // Per-channel bias broadcast over the conv output: `[1, C, 1, ...]`.
            let mut bias_shape = vec![1, out_channels];
            bias_shape.extend(std::iter::repeat_n(1, spatial_dims));
            let bias = Tensor::<f32>::zeros(&bias_shape);

            (conv + bias).build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "Conv");

        // The fused Conv should have a 1D bias input of shape `[out_channels]`.
        let bias_id = op.input_ids()[2].expect("conv should have a bias input");
        let bias = graph
            .get_node(bias_id)
            .and_then(|n| n.as_constant())
            .expect("bias should be a constant");
        assert_eq!(bias.shape(), [out_channels]);
    });
}

/// Create options which enable dynamic quantization for operators of type
/// `op` which match `pattern`.
fn dynamic_quant_rule(op: DynamicQuantOp, pattern: NamePattern) -> DynamicQuantOptions {
    DynamicQuantOptions::default().with_rule(op, pattern, DynamicQuantConfig::Int8)
}

#[test]
fn test_glob_match() {
    #[derive(Debug)]
    struct Case {
        pattern: &'static str,
        name: &'static str,
        expected: bool,
    }

    let case = |pattern, name, expected| Case {
        pattern,
        name,
        expected,
    };

    let cases = [
        // No wildcards
        case("", "", true),
        case("", "a", false),
        case("abc", "abc", true),
        case("abc", "abcd", false),
        case("abc", "xabc", false),
        // Wildcard only
        case("*", "", true),
        case("*", "abc", true),
        case("**", "abc", true),
        // Prefix, suffix and substring
        case("ab*", "abc", true),
        case("ab*", "ab", true),
        case("ab*", "xab", false),
        case("*bc", "abc", true),
        case("*bc", "bcd", false),
        case("*b*", "abc", true),
        case("*b*", "b", true),
        case("*b*", "ac", false),
        // Prefix and suffix may not overlap
        case("ab*bc", "abc", false),
        case("ab*bc", "abbc", true),
        // Multiple wildcards
        case("a*b*c", "abc", true),
        case("a*b*c", "axxbxxc", true),
        case("a*b*c", "acb", false),
        case("*a*a*", "a", false),
        case("*a*a*", "aa", true),
        // Suffix which also occurs earlier in the name
        case("a*bc", "abcbc", true),
        case("*.conv*.weight", "m.conv1.conv2.weight", true),
        // Wildcards match separators
        case("/decoder/*/Conv", "/decoder/block.0/conv1/Conv", true),
    ];

    cases.test_each(|case| {
        assert_eq!(super::glob_match(case.pattern, case.name), case.expected);
    });
}

#[test]
fn test_dynamic_quant_options() {
    use DynamicQuantConfig::{Int8, Off};
    use DynamicQuantOp::{Conv, ConvTranspose, MatMul};

    let opts = DynamicQuantOptions::default()
        .with_rule(Conv, NamePattern::weight("predictor.*"), Off)
        .with_rule(Conv, NamePattern::op("/conv_post/*"), Off)
        .with_rule(Conv, NamePattern::op("*"), Int8)
        .with_rule(MatMul, NamePattern::weight("encoder.*"), Int8)
        .with_rule(MatMul, NamePattern::op("/proj/*"), Int8);

    #[derive(Debug)]
    struct Case {
        op: DynamicQuantOp,
        op_name: Option<&'static str>,
        weight_name: Option<&'static str>,
        expected: DynamicQuantConfig,
    }

    let cases = [
        // First matching rule takes priority.
        Case {
            op: Conv,
            op_name: Some("/conv1/Conv"),
            weight_name: Some("predictor.conv1.weight"),
            expected: Off,
        },
        Case {
            op: Conv,
            op_name: Some("/conv_post/Conv"),
            weight_name: Some("decoder.conv_post.weight"),
            expected: Off,
        },
        Case {
            op: Conv,
            op_name: Some("/conv2/Conv"),
            weight_name: Some("decoder.conv2.weight"),
            expected: Int8,
        },
        // Operator patterns are not matched against weight names, and vice
        // versa.
        Case {
            op: Conv,
            op_name: Some("predictor.conv"),
            weight_name: Some("/conv_post/weight"),
            expected: Int8,
        },
        // Missing names are treated as empty.
        Case {
            op: Conv,
            op_name: None,
            weight_name: None,
            expected: Int8,
        },
        Case {
            op: MatMul,
            op_name: None,
            weight_name: None,
            expected: Off,
        },
        // Rules apply only to their operator type.
        Case {
            op: MatMul,
            op_name: Some("/layer/MatMul"),
            weight_name: Some("encoder.weight"),
            expected: Int8,
        },
        Case {
            op: MatMul,
            op_name: Some("/proj/MatMul"),
            weight_name: Some("proj.weight"),
            expected: Int8,
        },
        Case {
            op: MatMul,
            op_name: Some("/layer/MatMul"),
            weight_name: Some("decoder.weight"),
            expected: Off,
        },
        Case {
            op: ConvTranspose,
            op_name: Some("/conv2/Conv"),
            weight_name: Some("decoder.conv2.weight"),
            expected: Off,
        },
    ];

    cases.test_each(|case| {
        assert_eq!(
            opts.config(case.op, case.op_name, case.weight_name),
            &case.expected
        );
    });
}

#[test]
fn test_fuse_matmul_dynamic_quant() {
    #[derive(Debug)]
    struct Case {
        pattern: NamePattern,
        zero_point: i8,
        // Also use the dequantized weights in a non-`MatMul` operator.
        extra_consumer: bool,
        // Whether each of the two `MatMul`s is expected to be fused.
        expect_fused: [bool; 2],
    }

    let cases = [
        Case {
            pattern: NamePattern::op("matmul_*"),
            zero_point: 0,
            extra_consumer: false,
            expect_fused: [true, true],
        },
        // Weights which are not only used by `MatMul`s
        Case {
            pattern: NamePattern::op("matmul_*"),
            zero_point: 0,
            extra_consumer: true,
            expect_fused: [true, true],
        },
        // Selected by weight name
        Case {
            pattern: NamePattern::weight("layer.*"),
            zero_point: 0,
            extra_consumer: false,
            expect_fused: [true, true],
        },
        // Only one of the `MatMul`s which share the weights is selected
        Case {
            pattern: NamePattern::op("matmul_x"),
            zero_point: 0,
            extra_consumer: false,
            expect_fused: [true, false],
        },
        // Operator not selected
        Case {
            pattern: NamePattern::op("other*"),
            zero_point: 0,
            extra_consumer: false,
            expect_fused: [false, false],
        },
        // Non-zero weight zero point
        Case {
            pattern: NamePattern::op("*"),
            zero_point: 1,
            extra_consumer: false,
            expect_fused: [false, false],
        },
    ];

    cases.test_each(|case| {
        let n = 3;

        // Create a graph where blocked int8 weights are dequantized, cast to
        // f32 and shared by two `MatMul`s, one of which is followed by a bias.
        let make_graph = || {
            let mut graph = Graph::new();
            let x = graph.add_value(Some("x"), None, None);
            let y = graph.add_value(Some("y"), None, None);
            let weight_q = graph.add_constant(
                Some("layer.weight_q"),
                Tensor::<i8>::from([[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12]]).into_arc(),
            );
            let scale = graph.add_constant(
                Some("scale"),
                Tensor::from([[0.5f32, 0.25, 1.], [2., 0.5, 0.25]]).into_arc(),
            );
            let zero_point = graph.add_constant(
                Some("zero_point"),
                Tensor::<i8>::full(&[2, n], case.zero_point).into_arc(),
            );
            let (_, dequantized) = graph.add_simple_op(
                "dequantize",
                DequantizeLinear {
                    axis: 0,
                    block_size: 2,
                },
                &[weight_q, scale, zero_point],
            );
            let (_, weight) = graph.add_simple_op(
                "cast",
                Cast {
                    to: DataType::Float,
                },
                &[dequantized],
            );
            let (_, x_out) = graph.add_simple_op("matmul_x", MatMul {}, &[x, weight]);
            let bias = graph.add_constant(Some("bias"), Tensor::from([1f32, 2., 3.]).into_arc());
            let (_, x_out) = graph.add_simple_op("add_bias", Add {}, &[x_out, bias]);
            let (_, y_out) = graph.add_simple_op("matmul_y", MatMul {}, &[y, weight]);
            graph.set_input_ids(&[x, y]);
            if case.extra_consumer {
                let (_, extra_out) = graph.add_simple_op("extra", Add {}, &[weight, weight]);
                graph.set_output_ids(&[x_out, y_out, extra_out]);
            } else {
                graph.set_output_ids(&[x_out, y_out]);
            }
            graph
        };

        let graph = GraphOptimizer::new()
            .optimize(
                make_graph(),
                None,
                OptimizeOptions {
                    dynamic_quant: dynamic_quant_rule(DynamicQuantOp::MatMul, case.pattern.clone()),
                    ..Default::default()
                },
            )
            .unwrap();

        let op_names = [0, 1].map(|i| {
            let (_, op) = graph.get_source_node(graph.output_ids()[i]).unwrap();
            op.operator().name()
        });
        let is_fused = op_names.map(|name| name == "MatMulDynamicQuant");
        assert_eq!(is_fused, case.expect_fused);

        let (_, x_op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let (_, y_op) = graph.get_source_node(graph.output_ids()[1]).unwrap();
        if is_fused[0] {
            // The bias should be fused into the first `MatMul`.
            assert_eq!(x_op.input_ids().len(), 4);
        }
        if is_fused == [true, true] {
            // The requantized weights should be shared, unless the original
            // weights are also used elsewhere.
            assert_eq!(
                x_op.input_ids()[1..3] == y_op.input_ids()[1..3],
                !case.extra_consumer
            );
        }

        // Compare the fused and unfused graphs on concrete input.
        let x = Tensor::from([[0.5f32, -1., 2., 0.25], [3., 1., -0.5, 0.]]);
        let y = Tensor::from([[1f32, 2., 3., 4.]]);
        let run = |graph: &Graph| -> Vec<Tensor<f32>> {
            let inputs = vec![
                (graph.input_ids()[0], x.view().into()),
                (graph.input_ids()[1], y.view().into()),
            ];
            graph
                .run(inputs, graph.output_ids(), None, None)
                .unwrap()
                .into_iter()
                .map(|out| out.try_into().unwrap())
                .collect()
        };
        let expected = run(&make_graph());
        let actual = run(&graph);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.shape(), expected.shape());
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!((a - e).abs() < 0.05 * e.abs().max(1.), "{a} != {e}");
            }
        }
    })
}

#[test]
fn test_fuse_conv_dynamic_quant() {
    #[derive(Debug)]
    struct Case {
        pattern: NamePattern,
        // Use a scalar weight scale instead of a per-channel one.
        scalar_scale: bool,
        zero_point: i8,
        groups: usize,
        // Use blocked quantization, which requires requantizing the weights.
        blocked: bool,
        // Insert a `Cast` between the `DequantizeLinear` and `Conv`.
        cast: bool,
        expect_fused: bool,
    }

    let cases = [
        Case {
            pattern: NamePattern::op("conv"),
            scalar_scale: false,
            zero_point: 0,
            groups: 1,
            blocked: false,
            cast: false,
            expect_fused: true,
        },
        Case {
            pattern: NamePattern::op("*"),
            scalar_scale: true,
            zero_point: 0,
            groups: 1,
            blocked: false,
            cast: false,
            expect_fused: true,
        },
        // Operator not selected
        Case {
            pattern: NamePattern::op("other*"),
            scalar_scale: false,
            zero_point: 0,
            groups: 1,
            blocked: false,
            cast: false,
            expect_fused: false,
        },
        // Non-zero weight zero point
        Case {
            pattern: NamePattern::op("*"),
            scalar_scale: false,
            zero_point: 1,
            groups: 1,
            blocked: false,
            cast: false,
            expect_fused: false,
        },
        // Grouped convolution
        Case {
            pattern: NamePattern::op("*"),
            scalar_scale: false,
            zero_point: 0,
            groups: 2,
            blocked: false,
            cast: false,
            expect_fused: false,
        },
        // Blocked quantization with a cast, as used by int4 models
        Case {
            pattern: NamePattern::op("*"),
            scalar_scale: false,
            zero_point: 0,
            groups: 1,
            blocked: true,
            cast: true,
            expect_fused: true,
        },
    ];

    cases.test_each(|case| {
        let out_chans = 4;
        let in_chans = 4 / case.groups;
        let mut graph = Graph::new();
        let x = graph.add_value(Some("x"), None, None);
        let weight_q = graph.add_constant(
            Some("block.weight_q"),
            Tensor::<i8>::full(&[out_chans, in_chans, 3], 2).into_arc(),
        );
        let scale = if case.blocked {
            // Scale for blocks of 2 input channels.
            graph.add_constant(
                Some("scale"),
                Tensor::<f32>::full(&[out_chans, in_chans / 2, 3], 0.5).into_arc(),
            )
        } else if case.scalar_scale {
            graph.add_constant(Some("scale"), Tensor::from(0.5f32).into_arc())
        } else {
            graph.add_constant(
                Some("scale"),
                Tensor::from([0.5f32, 0.25, 1., 2.]).into_arc(),
            )
        };
        let zero_point = graph.add_constant(
            Some("zero_point"),
            Tensor::<i8>::full(&[out_chans], case.zero_point).into_arc(),
        );
        let dequantized = graph.add_value(Some("dequantized"), None, None);
        graph.add_op(
            Some("dequantize"),
            Arc::new(if case.blocked {
                DequantizeLinear {
                    axis: 1,
                    block_size: 2,
                }
            } else {
                DequantizeLinear {
                    axis: 0,
                    block_size: 0,
                }
            }),
            &[Some(weight_q), Some(scale), Some(zero_point)],
            &[Some(dequantized)],
        );
        let weight = if case.cast {
            let (_, cast_out) = graph.add_simple_op(
                "cast",
                Cast {
                    to: DataType::Float,
                },
                &[dequantized],
            );
            cast_out
        } else {
            dequantized
        };
        let bias = graph.add_constant(Some("bias"), Tensor::<f32>::zeros(&[out_chans]).into_arc());
        let (_, output) = graph.add_simple_op(
            "conv",
            Conv {
                groups: case.groups,
                dilations: vec![1],
                padding: Padding::Same,
                strides: vec![1],
            },
            &[x, weight, bias],
        );
        graph.set_input_ids(&[x]);
        graph.set_output_ids(&[output]);

        let graph = GraphOptimizer::new()
            .optimize(
                graph,
                None,
                OptimizeOptions {
                    dynamic_quant: dynamic_quant_rule(DynamicQuantOp::Conv, case.pattern.clone()),
                    ..Default::default()
                },
            )
            .unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        if !case.expect_fused {
            assert_eq!(op.operator().name(), "Conv");
            return;
        }
        assert_eq!(op.operator().name(), "ConvDynamicQuant");

        // Inputs should be the input, int8 weights, per-channel scale and bias.
        let input_ids: Vec<_> = op.input_ids().iter().map(|id| id.unwrap()).collect();
        assert_eq!(input_ids[0], x);
        if case.blocked {
            // Weights are requantized per output channel.
            let weight = graph
                .get_node(input_ids[1])
                .and_then(|n| n.as_constant())
                .unwrap();
            assert_eq!(weight.shape(), [out_chans, in_chans, 3]);

            // All dequantized weights are `2 * 0.5 = 1.0`, so each channel's
            // values map to the maximum quantized value.
            let weight: TensorView<i8> = TypedConstant::as_typed_view(weight).unwrap();
            assert!(weight.iter().all(|&w| w == 127));
            let scale = graph
                .get_node(input_ids[2])
                .and_then(|n| n.as_constant())
                .unwrap();
            let scale: TensorView<f32> = TypedConstant::as_typed_view(scale).unwrap();
            assert!(scale.iter().all(|&s| s == 1. / 127.));
        } else {
            assert_eq!(input_ids[1], weight_q);
        }
        let scale = graph
            .get_node(input_ids[2])
            .and_then(|n| n.as_constant())
            .unwrap();
        assert_eq!(scale.shape(), [out_chans]);
        assert_eq!(input_ids[3], bias);
    })
}

#[test]
fn test_fuse_conv_transpose_dynamic_quant() {
    #[derive(Debug)]
    struct Case {
        pattern: NamePattern,
        groups: usize,
        // Use blocked quantization along the input channel axis, followed by
        // a `Cast`, as used by int4 models.
        blocked: bool,
        expect_fused: bool,
    }

    let cases = [
        Case {
            pattern: NamePattern::op("conv_*"),
            groups: 1,
            blocked: false,
            expect_fused: true,
        },
        Case {
            pattern: NamePattern::op("*"),
            groups: 1,
            blocked: true,
            expect_fused: true,
        },
        // Operator not selected
        Case {
            pattern: NamePattern::op("other*"),
            groups: 1,
            blocked: false,
            expect_fused: false,
        },
        // Grouped transposed convolution
        Case {
            pattern: NamePattern::op("*"),
            groups: 2,
            blocked: false,
            expect_fused: false,
        },
    ];

    cases.test_each(|case| {
        let (in_chans, out_chans_per_group, kernel) = (4, 3, 4);
        let make_graph = || {
            let mut graph = Graph::new();
            let x = graph.add_value(Some("x"), None, None);
            let weight_q = graph.add_constant(
                Some("ups.weight_q"),
                Tensor::<i8>::from_fn(&[in_chans, out_chans_per_group, kernel], |idx| {
                    (idx[0] * 7 + idx[1] * 3 + idx[2]) as i8 - 10
                })
                .into_arc(),
            );
            let (scale, dq) = if case.blocked {
                // Scale for blocks of 2 input channels.
                let scale = Tensor::<f32>::from_fn(&[2, out_chans_per_group, kernel], |idx| {
                    0.01 * (1 + idx[0] + idx[1]) as f32
                });
                let dq = DequantizeLinear {
                    axis: 0,
                    block_size: 2,
                };
                (scale, dq)
            } else {
                let scale = Tensor::from([0.01f32, 0.02, 0.03]);
                let dq = DequantizeLinear {
                    axis: 1,
                    block_size: 0,
                };
                (scale, dq)
            };
            let scale = graph.add_constant(Some("scale"), scale.into_arc());
            let (_, mut weight) = graph.add_simple_op("dequantize", dq, &[weight_q, scale]);
            if case.blocked {
                (_, weight) = graph.add_simple_op(
                    "cast",
                    Cast {
                        to: DataType::Float,
                    },
                    &[weight],
                );
            }
            let bias = graph.add_constant(
                Some("bias"),
                Tensor::from_fn(&[out_chans_per_group * case.groups], |idx| idx[0] as f32)
                    .into_arc(),
            );
            let (_, output) = graph.add_simple_op(
                "conv_transpose",
                ConvTranspose {
                    groups: case.groups,
                    dilations: vec![1],
                    padding: Padding::Fixed([1, 1].as_slice().into()),
                    strides: vec![2],
                    output_padding: None,
                },
                &[x, weight, bias],
            );
            graph.set_input_ids(&[x]);
            graph.set_output_ids(&[output]);
            graph
        };

        let graph = GraphOptimizer::new()
            .optimize(
                make_graph(),
                None,
                OptimizeOptions {
                    dynamic_quant: dynamic_quant_rule(
                        DynamicQuantOp::ConvTranspose,
                        case.pattern.clone(),
                    ),
                    ..Default::default()
                },
            )
            .unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        if !case.expect_fused {
            assert_eq!(op.operator().name(), "ConvTranspose");
            return;
        }
        assert_eq!(op.operator().name(), "ConvTransposeDynamicQuant");

        // The weights are requantized with a scale for each output channel and
        // kernel position.
        let input_ids: Vec<_> = op.input_ids().iter().map(|id| id.unwrap()).collect();
        assert_eq!(input_ids[0], graph.input_ids()[0]);
        let weight = graph
            .get_node(input_ids[1])
            .and_then(|n| n.as_constant())
            .unwrap();
        assert_eq!(weight.shape(), [in_chans, out_chans_per_group, kernel]);
        let scale = graph
            .get_node(input_ids[2])
            .and_then(|n| n.as_constant())
            .unwrap();
        assert_eq!(scale.shape(), [out_chans_per_group * kernel]);

        // Compare the fused and unfused graphs on concrete input.
        let x = Tensor::from_fn(&[1, in_chans, 6], |idx| {
            ((idx[1] * 5 + idx[2] * 3) % 7) as f32 * 0.5 - 1.
        });
        let run = |graph: &Graph| -> Tensor<f32> {
            let inputs = vec![(graph.input_ids()[0], x.view().into())];
            graph
                .run(inputs, graph.output_ids(), None, None)
                .unwrap()
                .remove(0)
                .try_into()
                .unwrap()
        };
        let expected = run(&make_graph());
        let actual = run(&graph);
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 0.05 * e.abs().max(1.), "{a} != {e}");
        }
    })
}

#[test]
fn test_fuse_conv_dynamic_quant_residual() {
    #[derive(Debug)]
    struct Case {
        // Put the residual on the left side of the `Add`.
        residual_first: bool,
        // Add the convolution output to itself instead of to the input.
        add_to_self: bool,
        // Also use the convolution output as a graph output.
        reuse_conv_output: bool,
        expect_fused: bool,
    }

    let cases = [
        Case {
            residual_first: false,
            add_to_self: false,
            reuse_conv_output: false,
            expect_fused: true,
        },
        Case {
            residual_first: true,
            add_to_self: false,
            reuse_conv_output: false,
            expect_fused: true,
        },
        Case {
            residual_first: false,
            add_to_self: true,
            reuse_conv_output: false,
            expect_fused: false,
        },
        Case {
            residual_first: false,
            add_to_self: false,
            reuse_conv_output: true,
            expect_fused: false,
        },
    ];

    cases.test_each(|case| {
        let make_graph = || {
            let mut graph = Graph::new();
            let x = graph.add_value(Some("x"), None, None);
            let weight_q = graph.add_constant(
                Some("weight_q"),
                Tensor::<i8>::from_fn(&[4, 4, 3], |idx| (idx[0] + 2 * idx[1]) as i8 - idx[2] as i8)
                    .into_arc(),
            );
            let scale = graph.add_constant(
                Some("scale"),
                Tensor::from([0.5f32, 0.25, 1., 2.]).into_arc(),
            );
            let (_, weight) = graph.add_simple_op(
                "dequantize",
                DequantizeLinear {
                    axis: 0,
                    block_size: 0,
                },
                &[weight_q, scale],
            );
            let (_, conv_out) = graph.add_simple_op(
                "conv",
                Conv {
                    groups: 1,
                    dilations: vec![1],
                    padding: Padding::Same,
                    strides: vec![1],
                },
                &[x, weight],
            );
            let add_inputs = if case.add_to_self {
                [conv_out, conv_out]
            } else if case.residual_first {
                [x, conv_out]
            } else {
                [conv_out, x]
            };
            let (_, output) = graph.add_simple_op("add", Add {}, &add_inputs);
            graph.set_input_ids(&[x]);
            if case.reuse_conv_output {
                graph.set_output_ids(&[output, conv_out]);
            } else {
                graph.set_output_ids(&[output]);
            }
            graph
        };

        let graph = GraphOptimizer::new()
            .optimize(
                make_graph(),
                None,
                OptimizeOptions {
                    dynamic_quant: dynamic_quant_rule(DynamicQuantOp::Conv, NamePattern::op("*")),
                    ..Default::default()
                },
            )
            .unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        if !case.expect_fused {
            assert_eq!(op.operator().name(), "Add");
            return;
        }
        assert_eq!(op.operator().name(), "ConvDynamicQuant");
        assert_eq!(op.input_ids().len(), 5);
        assert_eq!(op.input_ids()[4], Some(graph.input_ids()[0]));

        // Compare the fused and unfused graphs on concrete input.
        let x = Tensor::from([[
            [0.5f32, -1., 2., 0.25, 1.5],
            [3., 1., -0.5, 0., 2.],
            [1., 1., 1., 1., 1.],
            [-2., 0.5, 0.25, 1., 0.],
        ]]);
        let run = |graph: &Graph| -> Tensor<f32> {
            let inputs = vec![(graph.input_ids()[0], x.view().into())];
            graph
                .run(inputs, &graph.output_ids()[..1], None, None)
                .unwrap()
                .remove(0)
                .try_into()
                .unwrap()
        };
        let expected = run(&make_graph());
        let actual = run(&graph);
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 0.05 * e.abs().max(1.), "{a} != {e}");
        }
    })
}

#[test]
fn test_fuse_batch_normalization() {
    #[derive(Debug)]
    struct Case {
        /// Rank of the input and of the scale and bias constants.
        rank: usize,
    }

    let cases = [Case { rank: 2 }, Case { rank: 3 }, Case { rank: 4 }];

    cases.test_each(|&Case { rank }| {
        let channels = 4;
        let graph = {
            let mut x_shape = vec![Dimension::from("batch"), Dimension::from(channels)];
            x_shape.extend(std::iter::repeat_n(Dimension::from(8), rank - 2));
            let x = Expr::value_with_info("x", ValueType::Tensor(DataType::Float), &x_shape);

            // Per-channel scale and bias broadcast over `x`: `[1, C, 1, ...]`.
            let mut const_shape = vec![1, channels];
            const_shape.extend(std::iter::repeat_n(1, rank - 2));
            let scale = Tensor::<f32>::full(const_shape.as_slice(), 2.);
            let bias = Tensor::<f32>::full(const_shape.as_slice(), 3.);

            (x * scale + bias).build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        assert_eq!(op.operator().name(), "BatchNormalization");

        // The scale, bias, mean and variance inputs should all be `[C]`
        // vectors. Mean and variance are chosen so that the normalization
        // reduces to `x * scale + bias`.
        let input_value = |index: usize| -> Vec<f32> {
            let input_id = op.input_ids()[index].expect("input should be present");
            let constant = graph
                .get_node(input_id)
                .and_then(|n| n.as_constant())
                .expect("input should be a constant");
            assert_eq!(constant.shape(), [channels]);
            TypedConstant::<f32>::as_typed_view(constant)
                .expect("input should be an f32 constant")
                .iter()
                .copied()
                .collect()
        };
        assert_eq!(input_value(1), [2.; 4]);
        assert_eq!(input_value(2), [3.; 4]);
        assert_eq!(input_value(3), [0.; 4]);
        assert_eq!(input_value(4), [1.; 4]);
    });
}

#[test]
fn test_fuse_conv_batch_norm() {
    #[derive(Debug)]
    struct Case {
        /// Bias of the `Conv`, or `None` if it has no bias input.
        conv_bias: Option<[f32; 2]>,
        expected_bias: [f32; 2],
    }

    // Weights are `[1., 2.]`, and the batch norm constants below give a scale
    // of `[3. / 2., 4. / 2.]` per output channel.
    let cases = [
        Case {
            conv_bias: Some([10., 20.]),
            // `(10. - 1.) * 1.5 + 5.`, `(20. - 2.) * 2. + 6.`
            expected_bias: [18.5, 42.],
        },
        Case {
            conv_bias: None,
            // `(0. - 1.) * 1.5 + 5.`, `(0. - 2.) * 2. + 6.`
            expected_bias: [3.5, 2.],
        },
    ];

    cases.test_each(
        |&Case {
             conv_bias,
             expected_bias,
         }| {
            let graph = {
                let x = Expr::value("x");

                // Weight shape is `[out_channels, in_channels, kernel]`.
                let weight = Expr::constant(Tensor::from([[[1.]], [[2.]]]));
                let mut conv_inputs = vec![weight];
                if let Some(bias) = conv_bias {
                    conv_inputs.push(Expr::constant(Tensor::from(bias)));
                }
                let conv = x.apply(
                    Conv {
                        groups: 1,
                        dilations: vec![1],
                        padding: Padding::Same,
                        strides: vec![1],
                    },
                    &conv_inputs,
                    &[OutputMeta::NoMeta],
                );

                // `(var + epsilon).sqrt()` is 2, so the per-channel scale
                // applied to the weights is `scale / 2`.
                conv.apply(
                    BatchNormalization { epsilon: 1. },
                    &[
                        Expr::constant(Tensor::from([3., 4.])),
                        Expr::constant(Tensor::from([5., 6.])),
                        Expr::constant(Tensor::from([1., 2.])),
                        Expr::constant(Tensor::from([3., 3.])),
                    ],
                    &[OutputMeta::NoMeta],
                )
                .build_graph(["x"])
            };

            let graph = optimize_graph(graph).unwrap();

            let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
            assert_eq!(op.operator().name(), "Conv");

            let constant_value = |index: usize| -> Vec<f32> {
                let input_id = op.input_ids()[index].expect("input should be present");
                let constant = graph
                    .get_node(input_id)
                    .and_then(|n| n.as_constant())
                    .expect("input should be a constant");
                TypedConstant::<f32>::as_typed_view(constant)
                    .expect("input should be an f32 constant")
                    .iter()
                    .copied()
                    .collect()
            };
            assert_eq!(constant_value(1), [1.5, 4.]);
            assert_eq!(constant_value(2), expected_bias);
        },
    );
}

#[test]
fn test_no_fuse_conv_add_non_channel_bias() {
    // A value added to the Conv output which broadcasts over a non-channel
    // axis (here the last/width axis, shape `[1, 1, 1, 4]`) is a valid add but
    // not a per-channel bias, so it must not be fused into the Conv.
    let graph = {
        let x = Expr::value("x");
        let weight = Expr::constant(Tensor::<f32>::zeros(&[4, 3, 3, 3]));
        let conv = x.apply(
            Conv {
                groups: 1,
                dilations: vec![1, 1],
                padding: Padding::zero::<2>(),
                strides: vec![1, 1],
            },
            &[weight],
            &[OutputMeta::NoMeta],
        );
        // Non-1 dimension is at index 3, not the channel index (1).
        let addend = Tensor::<f32>::zeros(&[1, 1, 1, 4]);
        (conv + addend).build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    assert_eq!(op.operator().name(), "Add");
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
    #[derive(Debug)]
    struct Case {
        build: fn(Expr) -> Expr,
    }

    // Test all bracketings of the GELU expression. The optimizer should fuse
    // each form, regardless of how the chain of `Mul` ops is associated.
    let cases = [
        // PyTorch's `nn.GELU` form.
        Case {
            build: |x| x.clone() * ((x / (2.0f32).sqrt()).erf() + 1.0) * 0.5,
        },
        // Erf argument expressed as `x * 1/sqrt(2)` instead of `x / sqrt(2)`
        Case {
            build: |x| x.clone() * ((x * 1. / (2.0f32).sqrt()).erf() + 1.0) * 0.5,
        },
        // Scale-first form: `(x * 0.5) * (1 + erf(x / sqrt(2)))`
        Case {
            build: |x| x.clone() * 0.5 * ((x / (2.0f32).sqrt()).erf() + 1.0),
        },
        // Right-associated form: `x * ((1 + erf(x / sqrt(2))) * 0.5)`
        Case {
            build: |x| {
                let half = Expr::constant(0.5);
                x.clone() * (((x / (2.0f32).sqrt()).erf() + 1.0) * half)
            },
        },
        // Right-associated, scale-first form: `x * (0.5 * (1 + erf(...)))`
        Case {
            build: |x| {
                let half = Expr::constant(0.5);
                x.clone() * (half * ((x / (2.0f32).sqrt()).erf() + 1.0))
            },
        },
    ];

    cases.test_each(|case| {
        let graph = {
            let x = Expr::value("x");
            let expr = (case.build)(x);
            expr.build_graph(["x"])
        };

        let graph = optimize_graph(graph).unwrap();

        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
        let gelu = op
            .operator()
            .downcast_ref::<Gelu>()
            .expect("expected fused Gelu op");
        assert!(!gelu.approximate);
    });
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
    assert!(gelu.approximate);
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
    let rms_norm = op.operator().downcast_ref::<RMSNormalization>().unwrap();
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
        let x = Expr::value_with_info("x", ValueType::Tensor(DataType::Float), dims);

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
    let rms_norm = op.operator().downcast_ref::<RMSNormalization>().unwrap();
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
        let qk = Expr::value_with_info("qk", ValueType::Tensor(DataType::Float), &dims);
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
fn test_fuse_safe_softmax() {
    let graph = {
        let x = Expr::value("x");
        let y = x.softmax(-1);
        let expr = y.is_nan().where_(Expr::constant(0.), y);
        expr.build_graph(["x"])
    };

    let graph = optimize_graph(graph).unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let softmax = op.operator().downcast_ref::<Softmax>().unwrap();
    assert_eq!(softmax.axis, -1);
    assert!(softmax.flush_nans_to_zero);
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
            ValueType::Tensor(DataType::Float),
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
fn test_no_eliminate_value_captured_by_subgraph() {
    let mut graph = Graph::new();
    let input = graph.add_value(Some("x"), None, Some(ValueType::Tensor(DataType::Float)));
    let cond = graph.add_value(Some("cond"), None, Some(ValueType::Tensor(DataType::Int32)));
    graph.set_input_ids(&[input, cond]);

    // No-op cast
    let (_cast_op, _cast_out) = graph.add_simple_op(
        "cast",
        Cast {
            to: DataType::Float,
        },
        &[input],
    );

    // Build then/else branches which each capture "cast_out" and return it.
    let make_branch = || {
        let mut sg = Graph::new();
        let cap = sg.add_value(Some("cast_out"), None, None);
        sg.set_captures(&[cap]);
        sg.set_output_ids(&[cap]);
        sg
    };

    let if_out = graph.add_value(Some("if_out"), None, None);
    graph.add_op(
        Some("If"),
        Arc::new(If {
            then_branch: make_branch(),
            else_branch: make_branch(),
        }),
        &[Some(cond)],
        &[Some(if_out)],
    );
    graph.set_output_ids(&[if_out]);

    let graph = optimize_graph(graph).unwrap();

    let cast_out = graph
        .get_node_id("cast_out")
        .expect("captured value should still exist");
    let (_, op) = graph
        .get_source_node(cast_out)
        .expect("captured value should still have a producer");
    assert_eq!(op.operator().name(), "Cast");
}

/// Create a graph containing a `DynamicQuantizeLinear` operator with three
/// outputs, of which only those in `used_outputs` are graph outputs.
fn quantize_graph(used_outputs: &[usize]) -> Graph {
    let x = Expr::value("x");
    let quant = x.apply(
        DynamicQuantizeLinear {},
        &[],
        &[OutputMeta::NoMeta, OutputMeta::NoMeta, OutputMeta::NoMeta],
    );
    let outputs: Vec<_> = used_outputs.iter().map(|&i| quant.output(i)).collect();
    Expr::make_graph([x], outputs)
}

/// Return the operator which produces the first output of `graph`.
fn first_output_source(graph: &Graph) -> &OperatorNode {
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    op
}

#[test]
fn test_remove_unused_outputs() {
    let graph = quantize_graph(&[0]);

    let unused_ids: Vec<NodeId> = first_output_source(&graph).output_ids()[1..]
        .iter()
        .flatten()
        .copied()
        .collect();
    assert_eq!(unused_ids.len(), 2);

    let graph = optimize_graph(graph).unwrap();

    let op = first_output_source(&graph);
    assert_eq!(op.operator().name(), "DynamicQuantizeLinear");
    assert_eq!(op.output_ids()[1..], [None, None]);

    for id in unused_ids {
        assert!(graph.get_node(id).is_none());
    }
}

#[test]
fn test_does_not_remove_output_used_as_graph_output() {
    let graph = optimize_graph(quantize_graph(&[0, 2])).unwrap();

    let op = first_output_source(&graph);
    assert!(op.output_ids()[1].is_none());
    assert_eq!(op.output_ids()[2], Some(graph.output_ids()[1]));
}

#[test]
fn test_does_not_remove_output_used_by_operator() {
    let x = Expr::value("x");
    let quant = x.apply(
        DynamicQuantizeLinear {},
        &[],
        &[OutputMeta::NoMeta, OutputMeta::NoMeta, OutputMeta::NoMeta],
    );
    let neg_scale = quant.output(1).unary(Neg {});
    let graph = optimize_graph(Expr::make_graph([x], [quant.output(0), neg_scale])).unwrap();

    let op = first_output_source(&graph);
    let scale_id = op.output_ids()[1].expect("scale output should be preserved");
    assert_eq!(
        graph
            .get_consuming_op(scale_id)
            .map(|op| op.operator().name()),
        Some("Neg")
    );
    assert!(op.output_ids()[2].is_none());
}

#[test]
fn test_does_not_remove_output_captured_by_subgraph() {
    let mut graph = Graph::new();
    let input = graph.add_value(Some("x"), None, Some(ValueType::Tensor(DataType::Float)));
    let cond = graph.add_value(Some("cond"), None, Some(ValueType::Tensor(DataType::Int32)));
    graph.set_input_ids(&[input, cond]);

    let quant_out = graph.add_value(Some("quant_out"), None, None);
    let scale = graph.add_value(Some("scale"), None, None);
    let zero_point = graph.add_value(Some("zero_point"), None, None);
    graph.add_op(
        Some("quant"),
        Arc::new(DynamicQuantizeLinear {}),
        &[Some(input)],
        &[quant_out, scale, zero_point].map(Some),
    );

    // Build then/else branches which each capture "scale" and return it.
    let make_branch = || {
        let mut sg = Graph::new();
        let cap = sg.add_value(Some("scale"), None, None);
        sg.set_captures(&[cap]);
        sg.set_output_ids(&[cap]);
        sg
    };

    let if_out = graph.add_value(Some("if_out"), None, None);
    graph.add_op(
        Some("If"),
        Arc::new(If {
            then_branch: make_branch(),
            else_branch: make_branch(),
        }),
        &[Some(cond)],
        &[Some(if_out)],
    );
    graph.set_output_ids(&[quant_out, if_out]);

    let graph = optimize_graph(graph).unwrap();

    let op = first_output_source(&graph);
    assert_eq!(op.output_ids()[1], Some(scale));
    // Un-captured zero-point output should be removed.
    assert!(op.output_ids()[2].is_none());
}

#[test]
fn test_removing_unused_output_preserves_output_count() {
    // When the optimizer removes unused operator outputs, it must preserve
    // the length of the output list as this can affect the behavior of the
    // `Split` operator.
    let x = Expr::value("x");
    let split = x.apply(
        Split {
            axis: 0,
            num_outputs: None,
        },
        &[],
        &[OutputMeta::NoMeta, OutputMeta::NoMeta, OutputMeta::NoMeta],
    );
    let graph = optimize_graph(Expr::make_graph([x.clone()], [split.output(0)])).unwrap();

    let op = first_output_source(&graph);
    assert_eq!(op.output_ids().len(), 3);
    assert_eq!(op.output_ids()[1..], [None, None]);
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
            &[
                OutputMeta::NoMeta,
                OutputMeta::Meta((DataType::Float, vec![])),
                OutputMeta::NoMeta,
            ],
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

    let graph = optimize_graph_infer_shapes(graph).unwrap();
    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();

    assert_eq!(op.operator().name(), "MatMulIntegerToFloat");
}

#[test]
fn test_fuse_convinteger_cast_scale() {
    #[derive(Debug)]
    struct Case {
        scale: Tensor<f32>,
        expected_op: &'static str,
    }

    let cases = [
        // Scalar scale
        Case {
            scale: Tensor::from(0.1),
            expected_op: "ConvIntegerToFloat",
        },
        // Unsupported scale shapes
        Case {
            scale: Tensor::from([0.1, 0.2, 0.3]),
            expected_op: "Mul",
        },
        Case {
            scale: Tensor::from_data(&[1, 1, 1, 1, 1], vec![0.1]),
            expected_op: "Mul",
        },
    ];

    cases.test_each(|case| {
        let graph = {
            let x = Expr::value("x");
            let weights = Expr::constant(Tensor::<i8>::zeros(&[3, 2, 3, 3]));
            let weights_zero = Expr::constant(Tensor::<i8>::zeros(&[3]));

            let quant = x.apply(
                DynamicQuantizeLinear {},
                &[],
                &[
                    OutputMeta::NoMeta,
                    OutputMeta::Meta((DataType::Float, vec![])),
                    OutputMeta::NoMeta,
                ],
            );
            let quant_x = quant.output(0);
            let quant_scale = quant.output(1);
            let quant_zero = quant.output(2);
            let weight_scale = Expr::constant(case.scale.clone());

            let expr = quant_x
                .apply(
                    ConvInteger {
                        groups: 1,
                        dilations: vec![1, 1],
                        padding: Padding::zero::<2>(),
                        strides: vec![1, 1],
                    },
                    &[weights, quant_zero, weights_zero],
                    &[OutputMeta::NoMeta],
                )
                .unary(Cast {
                    to: DataType::Float,
                })
                * (quant_scale * weight_scale);
            expr.build_graph(["x"])
        };

        let graph = optimize_graph_infer_shapes(graph).unwrap();
        let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();

        assert_eq!(op.operator().name(), case.expected_op);
    })
}

#[test]
fn test_slice_shape_to_constant() {
    // `Slice(Shape(X), starts, ends)` which can be simplified to a constant.
    let graph = {
        let x = Expr::value_with_info(
            "x",
            ValueType::Tensor(DataType::Float),
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

#[test]
fn test_fuse_repeat_interleave() {
    let batch_dim = Dimension::Symbolic("batch".to_string());
    let kv_heads = 8;
    let embed_dim = Dimension::Fixed(16);
    let n_repeats = 3;
    let repeat_axis = 1;

    let input_shape = [
        batch_dim.clone(),
        Dimension::Fixed(kv_heads),
        embed_dim.clone(),
    ];
    let x = Expr::value_with_info("x", ValueType::Tensor(DataType::Float), &input_shape);
    let unsqueeze_axes = Expr::constant(Value::from(NdTensor::from([repeat_axis as i32 + 1])));
    let expand_shape = Expr::value_with_info(
        "expand_shape",
        ValueType::Tensor(DataType::Float),
        &[
            batch_dim.clone(),
            Dimension::Fixed(kv_heads),
            Dimension::Fixed(n_repeats),
            embed_dim.clone(),
        ],
    );
    let out_shape = [
        batch_dim.clone(),
        Dimension::Fixed(kv_heads * n_repeats),
        embed_dim.clone(),
    ];
    let reshape_shape = Expr::value_with_info(
        "reshape_shape",
        ValueType::Tensor(DataType::Float),
        &out_shape,
    );

    let t1 = x.binary(Unsqueeze {}, unsqueeze_axes);
    let t2 = t1.binary(Expand {}, expand_shape);
    let output_meta = OutputMeta::Meta((DataType::Float, out_shape.to_vec()));
    let y = t2.apply(
        Reshape { allow_zero: false },
        &[reshape_shape],
        &[output_meta],
    );

    let graph = y.build_graph([
        "x",
        // In a real model, the shape inputs to the Expand and Reshape ops
        // would be computed from `Shape(X)` somehow. To simplify the graph here
        // we provide them as separate inputs.
        "expand_shape",
        "reshape_shape",
    ]);
    let optimized = optimize_graph(graph).unwrap();
    let op = optimized
        .get_node_id("Reshape")
        .and_then(|id| optimized.get_node(id))
        .and_then(|n| n.as_operator())
        .unwrap();
    let repeat_interleave_op = op.operator().downcast_ref::<RepeatInterleave>().unwrap();
    assert_eq!(repeat_interleave_op.axis, repeat_axis);
    assert_eq!(repeat_interleave_op.repeats, n_repeats);
}

#[test]
fn test_fuse_compute_shape() {
    let in_shape = vec![
        Dimension::Symbolic("batch".to_string()),
        Dimension::Fixed(3),
        Dimension::Fixed(224),
        Dimension::Fixed(224),
    ];

    let x = Expr::value_with_info("pixels", ValueType::Tensor(DataType::Float), &in_shape);
    let eps = Expr::constant(Value::from(NdTensor::from(0.)));
    let y = x.apply(
        Add {},
        &[eps],
        &[OutputMeta::Meta((DataType::Float, in_shape))],
    );

    // Create Shape op whose input has shape inference metadata with a mix
    // of static and symbolic dims.
    let shape = y.unary(Shape {
        start: None,
        end: None,
    });

    let graph = Expr::make_graph([x], [y, shape]);
    let optimized = optimize_graph(graph).unwrap();

    let op = optimized
        .get_node_id("Shape")
        .and_then(|id| optimized.get_node(id))
        .and_then(|n| n.as_operator())
        .unwrap();
    let op = op.operator().downcast_ref::<ComputeShape>().unwrap();

    assert_eq!(
        op.shape,
        [
            SymExpr::Var(
                Symbol {
                    name: "batch".into(),
                    positive: true,
                    synthetic: false,
                }
                .into()
            ),
            SymExpr::Value(3),
            SymExpr::Value(224),
            SymExpr::Value(224),
        ]
    );
}

#[test]
fn test_fuse_grouped_query_attention_matmul() {
    // Test fusion of MatMul(Q, RepeatInterleave(K_or_V)) for Grouped-query Attention.
    let batch_dim = Dimension::Symbolic("batch".to_string());
    let seq_dim = Dimension::Symbolic("seq".to_string());
    let kv_heads = 8;
    let n_repeats = 3;
    let query_heads = kv_heads * n_repeats;
    let d_model = 64;

    // Query tensor: (batch, query_heads, seq, d_model)
    let q_shape = [
        batch_dim.clone(),
        Dimension::Fixed(query_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let q = Expr::value_with_info("q", ValueType::Tensor(DataType::Float), &q_shape);

    // K/V tensor: (batch, kv_heads, seq, d_model)
    let kv_shape = [
        batch_dim.clone(),
        Dimension::Fixed(kv_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let kv = Expr::value_with_info("kv", ValueType::Tensor(DataType::Float), &kv_shape);

    // Create RepeatInterleave pattern: Unsqueeze + Expand + Reshape
    let repeat_axis = 1;
    let unsqueeze_axes = Expr::constant(Value::from(NdTensor::from([repeat_axis + 1])));
    let expand_shape = Expr::value_with_info(
        "expand_shape",
        ValueType::Tensor(DataType::Float),
        &[
            batch_dim.clone(),
            Dimension::Fixed(kv_heads),
            Dimension::Fixed(n_repeats),
            seq_dim.clone(),
            Dimension::Fixed(d_model),
        ],
    );
    let out_shape = [
        batch_dim.clone(),
        Dimension::Fixed(query_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let reshape_shape = Expr::value_with_info(
        "reshape_shape",
        ValueType::Tensor(DataType::Float),
        &out_shape,
    );

    let t1 = kv.binary(Unsqueeze {}, unsqueeze_axes);
    let t2 = t1.binary(Expand {}, expand_shape);
    let output_meta = OutputMeta::Meta((DataType::Float, out_shape.to_vec()));
    let kv_repeated = t2.apply(
        Reshape { allow_zero: false },
        &[reshape_shape],
        &[output_meta],
    );

    // MatMul(Q, RepeatInterleave(K_or_V))
    let expr = q.matmul(kv_repeated);
    let graph = expr.build_graph(["q", "kv", "expand_shape", "reshape_shape"]);

    let optimized = optimize_graph(graph).unwrap();

    // Verify the fusion occurred
    let (_, op) = optimized
        .get_source_node(optimized.output_ids()[0])
        .unwrap();
    let qkv_matmul = op
        .operator()
        .downcast_ref::<GroupedQueryAttentionMatMul>()
        .unwrap();
    assert_eq!(qkv_matmul.repeats, n_repeats);
    assert_eq!(qkv_matmul.alpha, None);
    assert!(!qkv_matmul.transpose_rhs);
}

#[test]
fn test_fuse_grouped_query_attention_matmul_with_transpose_and_scale() {
    // Test fusion of MatMul(Q, Transpose(RepeatInterleave(K))) / scale
    // This is the typical pattern for Q @ K^T in Grouped-query Attention.
    let batch_dim = Dimension::Symbolic("batch".to_string());
    let seq_dim = Dimension::Symbolic("seq".to_string());
    let kv_heads = 8;
    let n_repeats = 3;
    let query_heads = kv_heads * n_repeats;
    let d_model = 64;
    let scale = (d_model as f32).sqrt();

    // Query tensor: (batch, query_heads, seq, d_model)
    let q_shape = [
        batch_dim.clone(),
        Dimension::Fixed(query_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let q = Expr::value_with_info("q", ValueType::Tensor(DataType::Float), &q_shape);

    // K tensor: (batch, kv_heads, seq, d_model)
    let k_shape = [
        batch_dim.clone(),
        Dimension::Fixed(kv_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let k = Expr::value_with_info("k", ValueType::Tensor(DataType::Float), &k_shape);

    // Create RepeatInterleave pattern: Unsqueeze + Expand + Reshape
    let repeat_axis = 1;
    let unsqueeze_axes = Expr::constant(Value::from(NdTensor::from([repeat_axis + 1])));
    let expand_shape = Expr::value_with_info(
        "expand_shape",
        ValueType::Tensor(DataType::Float),
        &[
            batch_dim.clone(),
            Dimension::Fixed(kv_heads),
            Dimension::Fixed(n_repeats),
            seq_dim.clone(),
            Dimension::Fixed(d_model),
        ],
    );
    let repeated_shape = [
        batch_dim.clone(),
        Dimension::Fixed(query_heads),
        seq_dim.clone(),
        Dimension::Fixed(d_model),
    ];
    let reshape_shape = Expr::value_with_info(
        "reshape_shape",
        ValueType::Tensor(DataType::Float),
        &repeated_shape,
    );

    let t1 = k.binary(Unsqueeze {}, unsqueeze_axes);
    let t2 = t1.binary(Expand {}, expand_shape);
    let output_meta = OutputMeta::Meta((DataType::Float, repeated_shape.to_vec()));
    let k_repeated = t2.apply(
        Reshape { allow_zero: false },
        &[reshape_shape],
        &[output_meta],
    );

    // Transpose(RepeatInterleave(K))
    // After transpose, shape becomes (batch, query_heads, d_model, seq)
    let k_repeated_transposed = k_repeated.permute(&[0, 1, 3, 2]);

    // MatMul(Q, Transpose(RepeatInterleave(K))) / scale
    let expr = q.matmul(k_repeated_transposed) / scale;
    let graph = expr.build_graph(["q", "k", "expand_shape", "reshape_shape"]);

    let optimized = optimize_graph(graph).unwrap();

    // Verify the fusion occurred
    let (_, op) = optimized
        .get_source_node(optimized.output_ids()[0])
        .unwrap();
    let qkv_matmul = op
        .operator()
        .downcast_ref::<GroupedQueryAttentionMatMul>()
        .unwrap();
    assert_eq!(qkv_matmul.repeats, n_repeats);
    assert_eq!(qkv_matmul.alpha, Some(1.0 / scale));
    assert!(qkv_matmul.transpose_rhs);
}

#[test]
fn test_infer_shapes() {
    // Build a graph that has input shape and type metadata, but no output
    // metadata.
    let graph = {
        let x = Expr::value_with_info(
            "data",
            ValueType::Tensor(DataType::Float),
            &dims!("batch", 64),
        );
        let w = Expr::constant(NdTensor::<f32, _>::zeros([64, 12]));
        let out = x.apply(MatMul {}, &[w], &[OutputMeta::NoMeta]);
        out.build_graph(["data"])
    };

    // Run optimization with shape inference enabled.
    let optimizer = GraphOptimizer::new();
    let graph = optimizer
        .optimize(
            graph,
            None,
            OptimizeOptions {
                infer_shapes: Some(InferShapeOptions::default()),
                ..Default::default()
            },
        )
        .unwrap();

    // Verify that values were updated with inferred shapes and types.
    let output = graph.get_node(graph.output_ids()[0]).unwrap();
    assert_eq!(
        output.shape().as_deref(),
        Some(dims!("batch", 12).as_slice())
    );
    assert_eq!(output.dtype(), Some(ValueType::Tensor(DataType::Float)));
}

#[test]
fn test_shape_inference_replaces_values_with_constants() {
    let graph = {
        let x = Expr::value_with_info(
            "data",
            ValueType::Tensor(DataType::Float),
            &dims!("batch", 64),
        );

        // Extract second dimension of input via `Gather<axis=0>(Shape(X), indices=[1])`.
        let indices = Expr::constant(1);
        let out = x
            .shape()
            .apply(Gather { axis: 0 }, &[indices], &[OutputMeta::NoMeta]);
        out.build_graph(["data"])
    };

    let optimizer = GraphOptimizer::new();
    let graph = optimizer
        .optimize(
            graph,
            None,
            OptimizeOptions {
                infer_shapes: Some(InferShapeOptions::default()),
                ..Default::default()
            },
        )
        .unwrap();

    // The output should be replaced with a constant as it doesn't depend on
    // model inputs.
    let output = graph.get_node(graph.output_ids()[0]).unwrap();
    assert_eq!(output.as_constant().and_then(|c| c.as_scalar()), Some(64));
}

#[test]
fn test_shape_inference_constants_preserve_value_type() {
    let graph = {
        let x = Expr::value_with_info(
            "data",
            ValueType::Tensor(DataType::Float),
            &dims!("batch", 4),
        );
        let bias_src = Expr::constant(NdTensor::<f32, _>::full([4], 1.));
        let bias = bias_src.unary(Identity {});
        let out = x.binary(Add {}, bias);
        out.build_graph(["data"])
    };

    let optimizer = GraphOptimizer::new();
    let graph = optimizer
        .optimize(
            graph,
            None,
            OptimizeOptions {
                infer_shapes: Some(InferShapeOptions::default()),
                ..Default::default()
            },
        )
        .unwrap();

    let (_, op) = graph.get_source_node(graph.output_ids()[0]).unwrap();
    let bias = op
        .input_ids()
        .iter()
        .flatten()
        .find_map(|id| graph.get_node(*id).and_then(|node| node.as_constant()))
        .expect("expected a constant operand");

    // f32 element type should be preserved
    assert_eq!(bias.as_vector(), Some([1., 1., 1., 1.].as_slice()));
}

#[test]
fn test_shape_inference_constants_shared_by_values_with_different_types() {
    // Create a graph with a `Shape` node whose output has the same value
    // `[2, 3]` as a float tensor. Shape inference will produce a single
    // constant for these, but since the value nodes have different types,
    // one constant node per type must be created in the graph.
    let graph = {
        let x = Expr::value_with_info("data", ValueType::Tensor(DataType::Float), &dims!(2, 3));
        let shape = x.unary(Shape {
            start: None,
            end: None,
        });

        let y = Expr::value_with_info(
            "scaled",
            ValueType::Tensor(DataType::Float),
            &dims!("batch", 2),
        );
        let scale = Expr::constant(NdTensor::from([2., 3.])).unary(Identity {});
        let scaled = y.binary(Add {}, scale);

        Expr::make_graph([x, y], [shape, scaled])
    };

    let optimizer = GraphOptimizer::new();
    let graph = optimizer
        .optimize(
            graph,
            None,
            OptimizeOptions {
                infer_shapes: Some(InferShapeOptions::default()),
                ..Default::default()
            },
        )
        .unwrap();

    let shape_out = graph.get_node(graph.output_ids()[0]).unwrap();
    assert_eq!(
        shape_out.as_constant().and_then(|c| c.as_vector()),
        Some([2i32, 3].as_slice())
    );

    let (_, op) = graph.get_source_node(graph.output_ids()[1]).unwrap();
    let scale = op
        .input_ids()
        .iter()
        .flatten()
        .find_map(|id| graph.get_node(*id).and_then(|node| node.as_constant()))
        .expect("expected a constant operand");
    assert_eq!(scale.as_vector(), Some([2f32, 3.].as_slice()));
}

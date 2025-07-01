use std::error::Error;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use rten_tensor::prelude::*;
use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
use rten_tensor::{Tensor, TensorView};

use smallvec::{smallvec, SmallVec};

use super::{CachedPlan, CaptureEnv, PlanOptions};
use crate::graph::{Dimension, Graph, Node, NodeId, RunError, RunOptions, TypedConstant};
use crate::ops::{
    Add, Concat, Conv, Identity, If, IntoOpResult, MatMul, Mul, OpError, OpRunContext, Operator,
    OutputList, PrepackedInput, Relu, Shape,
};
use crate::timing::Profiler;
use crate::value::{DataType, Value, ValueView};
use crate::weight_cache::WeightCache;

#[derive(Clone, Debug, Default)]
struct Metrics {
    run_count: u32,
    run_in_place_count: u32,
}

/// Operator adapter that wraps an underlying operator in order to track
/// uses of it.
#[derive(Debug)]
struct TrackUsage<Op: Operator> {
    inner: Op,
    metrics: Arc<Mutex<Metrics>>,
}

impl<Op: Operator> TrackUsage<Op> {
    /// Construct a new adapter that wraps `inner`.
    fn new(inner: Op) -> Self {
        TrackUsage {
            inner,
            metrics: Default::default(),
        }
    }

    /// Return a shared reference to the operator's usage counters.
    fn metrics(&self) -> Arc<Mutex<Metrics>> {
        self.metrics.clone()
    }
}

impl<Op: Operator> Operator for TrackUsage<Op> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn can_run_in_place(&self) -> bool {
        self.inner.can_run_in_place()
    }

    fn is_commutative(&self) -> bool {
        self.inner.is_commutative()
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        {
            let mut m = self.metrics.lock().unwrap();
            m.run_count += 1;
        }
        self.inner.run(ctx)
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        {
            let mut m = self.metrics.lock().unwrap();
            m.run_in_place_count += 1;
        }
        self.inner.run_in_place(input, ctx)
    }
}

/// Operator that wraps a function.
///
/// Useful for tests that want to inspect operator inputs.
struct RunFn<F: Fn(&OpRunContext) -> Result<OutputList, OpError> + 'static> {
    run: F,
}

impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> RunFn<F> {
    fn new(run: F) -> Self {
        Self { run }
    }
}

impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> std::fmt::Debug for RunFn<F> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "RunFn")
    }
}

impl<F: Fn(&OpRunContext) -> Result<OutputList, OpError>> Operator for RunFn<F> {
    fn name(&self) -> &str {
        "RunFn"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        (self.run)(ctx)
    }
}

// Test of a very simple graph with a typical structure (one input, one
// output, Conv + Relu operation).
#[test]
fn test_graph_run() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let weights = Tensor::from_data(
        &[1, 1, 3, 3],
        vec![
            0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
        ],
    );
    let weights_id = g.add_constant(Some("weight"), weights);
    let input_id = g.add_value(Some("input"), None, None);

    let (_, conv_out) = g.add_simple_op(
        "conv",
        Conv {
            dilations: vec![1, 1],
            groups: 1,
            padding: [1, 1, 1, 1].into(),
            strides: vec![1, 1],
        },
        &[input_id, weights_id],
    );
    let (_, relu_out) = g.add_simple_op("relu", Relu {}, &[conv_out]);

    let input = Tensor::from_data(
        &[1, 1, 3, 3],
        vec![
            0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
        ],
    );

    let results = g
        .run(vec![(input_id, input.into())], &[relu_out], None, None)
        .unwrap();

    let expected = Tensor::from_data(
        &[1, 1, 3, 3],
        vec![
            1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
        ],
    );
    assert_eq!(results.len(), 1);
    expect_equal_with_tolerance(
        &results[0].as_tensor_view().unwrap(),
        &expected.view(),
        1e-4,
        0.,
    )?;

    Ok(())
}

#[test]
fn test_graph_node_debug_names() {
    let mut g = Graph::new();

    let weights = Tensor::from([0.3230]);
    let weights_id = g.add_constant(Some("weights"), weights.clone());
    let input_id = g.add_value(Some("input"), None, None);
    let relu_out_id = g.add_value(Some("relu_out"), None, None);
    let relu_op_id = g.add_op(
        Some("relu"),
        Box::new(Relu {}),
        &[Some(input_id)],
        &[Some(relu_out_id)],
    );

    assert_eq!(g.node_name(weights_id), "weights");
    assert_eq!(g.node_name(input_id), "input");
    assert_eq!(g.node_name(relu_op_id), "relu");

    let anon_weights_id = g.add_constant(None, weights);
    let anon_input_id = g.add_value(None, None, None);
    let anon_out_id = g.add_value(None, None, None);
    let anon_op_id = g.add_op(
        None,
        Box::new(Relu {}),
        &[Some(input_id)],
        &[Some(anon_out_id)],
    );

    assert_eq!(
        g.node_name(anon_weights_id),
        format!("[ID: {}]", anon_weights_id)
    );
    assert_eq!(
        g.node_name(anon_input_id),
        format!("[ID: {}]", anon_input_id)
    );
    assert_eq!(g.node_name(anon_op_id), format!("[ID: {}]", anon_op_id));
}

#[test]
fn test_graph_node_shapes() {
    let mut g = Graph::new();

    let weights = Tensor::from_data(&[1, 1, 2], vec![0.3230, 0.5]);
    let weights_id = g.add_constant(Some("weights"), weights.clone());
    let input_id = g.add_value(
        Some("input"),
        Some(
            [
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(3),
                Dimension::Fixed(5),
                Dimension::Fixed(5),
            ]
            .to_vec(),
        ),
        None,
    );
    let (relu_op_id, _) = g.add_simple_op("relu", Relu {}, &[input_id]);

    assert_eq!(
        g.get_node(weights_id).and_then(|n| n.shape()),
        Some([1, 1, 2].map(Dimension::Fixed).to_vec())
    );
    assert_eq!(
        g.get_node(input_id).and_then(|n| n.shape()),
        Some(
            [
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(3),
                Dimension::Fixed(5),
                Dimension::Fixed(5),
            ]
            .to_vec()
        )
    );
    assert_eq!(g.get_node(relu_op_id).and_then(|n| n.shape()), None);
}

#[test]
fn test_graph_value_dtype() {
    let mut g = Graph::new();
    for dtype in [
        DataType::Float,
        DataType::Int32,
        DataType::UInt8,
        DataType::Int8,
    ] {
        let input_id = g.add_value(None, None, Some(dtype));
        let input_dtype = g.get_node(input_id).and_then(|n| n.dtype());
        assert_eq!(input_dtype, Some(dtype));
    }
}

#[derive(Debug)]
struct AddOne {}
impl Operator for AddOne {
    fn name(&self) -> &str {
        "AddOne"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        let output_data: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
        Tensor::<f32>::from_data(input.shape().into(), output_data).into_op_result()
    }
}

#[test]
fn test_graph_planning_order() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let input_id = g.add_value(Some("input"), None, None);

    let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
    let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

    // op_c has both op_a and op_b as inputs. Since op_b depends on op_a,
    // execution must run op_a, then op_b, then op_c.
    let (_, op_c_out) = g.add_simple_op("op_c", Concat { axis: 0 }, &[op_a_out, op_b_out]);

    // op_d is the same as op_c, but input order is reversed
    let (_, op_d_out) = g.add_simple_op("op_d", Concat { axis: 0 }, &[op_b_out, op_a_out]);

    let input = Tensor::from([1.]);

    let results = g
        .run(
            vec![(input_id, input.view().into())],
            &[op_c_out],
            None,
            None,
        )
        .unwrap();
    let expected = Tensor::from([2., 3.]);
    expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

    let results = g
        .run(vec![(input_id, input.into())], &[op_d_out], None, None)
        .unwrap();
    let expected = Tensor::from([3., 2.]);
    expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

    Ok(())
}

#[test]
fn test_runs_non_in_place_ops_first() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let input_a_id = g.add_value(Some("input_a"), None, None);
    let input_b_id = g.add_value(Some("input_b"), None, None);

    let (add_op, add_out) = g.add_simple_op("add", Add {}, &[input_a_id, input_b_id]);
    let (shape_op, shape_out) = g.add_simple_op("shape", Shape::default(), &[input_a_id]);

    // The execution plan could run operators in either order and produce
    // the correct output. Since the `Add` op has the _potential_ to run in
    // place (if the input is passed as an owned value) and the `Shape` op
    // does not, the Shape op should be run first.
    let plan = g.execution_plan(
        &[input_a_id, input_b_id],
        &[add_out, shape_out],
        PlanOptions::default(),
    )?;
    assert_eq!(plan, &[shape_op, add_op]);

    // Make sure the results are the same if the order of outputs is
    // swapped.
    let plan = g.execution_plan(
        &[input_a_id, input_b_id],
        &[shape_out, add_out],
        PlanOptions::default(),
    )?;
    assert_eq!(plan, &[shape_op, add_op]);

    Ok(())
}

// Perform a graph run where one of the outputs is also an input for other
// steps of the run.
#[test]
fn test_graph_intermediate_output() {
    let mut g = Graph::new();

    let input_id = g.add_value(Some("input"), None, None);
    let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);
    let (_, op_b_out) = g.add_simple_op("op_b", AddOne {}, &[op_a_out]);

    let input = Tensor::from(0.);
    let results = g
        .run(
            vec![(input_id, input.into())],
            &[op_a_out, op_b_out],
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        &results[0].as_tensor_view().unwrap(),
        &Tensor::from(1.).view()
    );
    assert_eq!(
        &results[1].as_tensor_view().unwrap(),
        &Tensor::from(2.).view()
    );
}

#[test]
fn test_graph_many_steps() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let input = Tensor::from([1., 2., 3., 4., 5.]);
    let input_id = g.add_value(Some("input"), None, None);

    let mut prev_output = input_id;
    for _ in 0..100 {
        let next_output = g.add_value(None, None, None);
        g.add_op(
            None,
            Box::new(AddOne {}),
            &[Some(prev_output)],
            &[Some(next_output)],
        );
        prev_output = next_output;
    }

    let results = g
        .run(vec![(input_id, input.into())], &[prev_output], None, None)
        .unwrap();

    let expected = Tensor::from([101., 102., 103., 104., 105.]);
    expect_equal(&results[0].as_tensor_view().unwrap(), &expected.view())?;

    Ok(())
}

#[test]
fn test_noop_graph() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let input = Tensor::from([1., 2., 3., 4., 5.]);
    let input_id = g.add_value(Some("input"), None, None);

    let results = g
        .run(
            vec![(input_id, input.view().into())],
            &[input_id],
            None,
            None,
        )
        .unwrap();

    expect_equal(&results[0].as_tensor_view().unwrap(), &input.view())?;

    Ok(())
}

#[test]
fn test_constant_graph() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();

    let value = Tensor::from([1., 2., 3., 4., 5.]);
    let const_id = g.add_constant(Some("weight"), value.clone());

    let results = g.run(vec![], &[const_id], None, None).unwrap();

    expect_equal(&results[0].as_tensor_view().unwrap(), &value.view())?;

    Ok(())
}

#[test]
fn test_typed_constant() {
    let mut g = Graph::new();
    let scalar_id = g.add_constant(None, Tensor::from(42.));
    let vec_id = g.add_constant(None, Tensor::from([1, 2, 3]));

    let scalar_node = match g.get_node(scalar_id) {
        Some(Node::Constant(c)) => Some(c),
        _ => None,
    }
    .unwrap();
    let vec_node = match g.get_node(vec_id) {
        Some(Node::Constant(c)) => Some(c),
        _ => None,
    }
    .unwrap();

    assert_eq!(scalar_node.as_scalar(), Some(42.0));
    assert_ne!(scalar_node.as_scalar(), Some(42));
    assert_eq!(vec_node.as_scalar(), None::<i32>);

    assert_eq!(vec_node.as_vector(), Some([1, 2, 3].as_slice()));
    assert_eq!(vec_node.as_scalar(), None::<f32>);
}

#[test]
fn test_total_params() {
    let mut g = Graph::new();
    g.add_constant(Some("floats"), Tensor::<f32>::zeros(&[10, 10]));
    g.add_constant(Some("ints"), Tensor::<i32>::zeros(&[10, 10]));

    let mut subgraph = Graph::new();
    subgraph.add_constant(Some("sg_floats"), Tensor::<f32>::zeros(&[10, 10]));
    g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    assert_eq!(g.total_params(), 300);
}

#[test]
fn test_no_outputs() {
    let g = Graph::new();
    let results = g.run(vec![], &[], None, None).unwrap();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_duplicate_inputs() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);
    let input = Tensor::from([1.]);
    let result = g.run(
        vec![
            (input_id, input.view().into()),
            (input_id, input.view().into()),
        ],
        &[input_id],
        None,
        None,
    );
    assert_eq!(
        result,
        Err(RunError::PlanningError(
            "Inputs are not unique. Input \"input\" is duplicated.".into()
        ))
    );
}

#[test]
fn test_duplicate_outputs() {
    let mut g = Graph::new();

    let input_id = g.add_value(Some("input"), None, None);
    let (_, op_a_out) = g.add_simple_op("op_a", AddOne {}, &[input_id]);

    let input = Tensor::from([1.]);

    let result = g.run(
        vec![(input_id, input.into())],
        &[op_a_out, op_a_out],
        None,
        None,
    );

    assert_eq!(
        result,
        Err(RunError::PlanningError(
            "Outputs are not unique. Output \"op_a_out\" is duplicated.".into()
        ))
    );
}

#[test]
fn test_no_source_for_output() {
    let mut g = Graph::new();
    let output_id = g.add_value(Some("output"), None, None);
    let err = g.run(vec![], &[output_id], None, None);
    assert_eq!(
        err,
        Err(RunError::PlanningError(
            "Source node not found for output \"output\"".into()
        ))
    );
}

#[test]
fn test_invalid_input_id() {
    let mut g = Graph::new();

    let (op_id, op_out) = g.add_simple_op("op", AddOne {}, &[]);
    let input = Tensor::from([1.]);
    let invalid_id = NodeId::from_u32(1234);

    for wrong_input_id in [op_id, invalid_id] {
        let result = g.run(
            [(wrong_input_id, input.view().into())].into(),
            &[op_out],
            None,
            None,
        );
        let name = g.node_name(wrong_input_id);
        assert_eq!(
            result,
            Err(RunError::PlanningError(format!(
                "Input 0 (\"{}\") is not a value node in the graph.",
                name
            ),))
        );
    }
}

#[test]
fn test_invalid_output_id() {
    let mut g = Graph::new();

    let input_id = g.add_value(None, None, None);
    let (op_id, _op_out) = g.add_simple_op("op", AddOne {}, &[input_id]);
    let input = Tensor::from([1.]);
    let invalid_id = NodeId::from_u32(1234);

    for wrong_output_id in [op_id, invalid_id] {
        let result = g.run(
            [(input_id, input.view().into())].into(),
            &[wrong_output_id],
            None,
            None,
        );
        let name = g.node_name(wrong_output_id);
        assert_eq!(
            result,
            Err(RunError::PlanningError(format!(
                "Output 0 (\"{}\") is not a value node in the graph.",
                name
            )))
        );
    }
}

#[test]
fn test_call_op_with_missing_input() {
    let mut g = Graph::new();

    // Call an operator with an input omitted by setting it to `None`,
    // as opposed to passing a shorter input list. This enables omitting
    // an input but still providing subsequent ones.
    let output = g.add_value(None, None, None);
    g.add_op(
        Some("shape"),
        Box::new(Shape::default()),
        &[None],
        &[Some(output)],
    );

    let results = g.run(vec![], &[output], None, None);

    assert_eq!(
        results.err(),
        Some(RunError::OperatorError {
            name: "shape".to_string(),
            error: OpError::MissingInputs,
            inputs: [None].into(),
        })
    );
}

#[test]
fn test_err_if_missing_operator_input() {
    let mut g = Graph::new();
    let (_, output) = g.add_simple_op("op", Relu {}, &[NodeId::from_u32(42)]);
    let result = g.run(vec![], &[output], None, None);
    assert_eq!(
        result.err(),
        Some(RunError::PlanningError(
            "Missing input \"[ID: 42]\" for op \"op\"".to_string()
        ))
    );
}

#[derive(Debug)]
struct AddOneInPlace {}
impl Operator for AddOneInPlace {
    fn name(&self) -> &str {
        "AddOneInPlace"
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        // An operator should normally have the same behavior in `run`
        // and `run_in_place`. Here we use different behavior to make it
        // possible to distinguish which path was used.
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        input.to_tensor().into_op_result()
    }

    fn run_in_place(&self, input: Value, _ctx: &OpRunContext) -> Result<Value, OpError> {
        let mut output = input.into_tensor::<f32>().unwrap();
        for x in output.iter_mut() {
            *x = *x + 1.0;
        }
        Ok(output.into())
    }
}

#[test]
fn test_runs_op_in_place() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    let (_, op1_out) = g.add_simple_op("op1", AddOneInPlace {}, &[input_id]);
    let (_, op2_out) = g.add_simple_op("op2", AddOneInPlace {}, &[op1_out]);
    let (_, op3_out) = g.add_simple_op("op3", AddOneInPlace {}, &[op2_out]);
    let (_, op4_out) = g.add_simple_op("op4", AddOneInPlace {}, &[op2_out]);
    let input = Tensor::<f32>::zeros(&[1, 1]);

    // First operator should not be run in-place, since it has an
    // immutable input. The result should be the same as the input.
    let results = g
        .run(
            vec![(input_id, input.view().into())],
            &[op1_out],
            None,
            None,
        )
        .unwrap();
    assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 0.0);

    // Second operator should be run in-place, as it meets all the
    // requirements for this optimization.
    let results = g
        .run(
            vec![(input_id, input.view().into())],
            &[op2_out],
            None,
            None,
        )
        .unwrap();
    assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 1.0);

    // Third op should not be run in place, because its input is re-used
    // for fourth op. Fourth op can run in place as by then, it is the
    // only consumer of its input.
    let results = g
        .run(
            vec![(input_id, input.view().into())],
            &[op3_out, op4_out],
            None,
            None,
        )
        .unwrap();
    assert_eq!(results[0].as_tensor_view::<f32>().unwrap()[[0, 0]], 1.0);
    assert_eq!(results[1].as_tensor_view::<f32>().unwrap()[[0, 0]], 2.0);
}

// Test that the graph executor will swap inputs to commutative ops if
// necessary to enable running in-place.
#[test]
fn test_runs_commutative_op_in_place() {
    use crate::ops::Add; // A commutative operator

    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);
    let bias_id = g.add_value(Some("bias"), None, None);

    let op1 = TrackUsage::new(Add {});
    let op1_metrics = op1.metrics();

    let op2 = TrackUsage::new(Add {});
    let op2_metrics = op2.metrics();

    let (_, op1_out) = g.add_simple_op("op1", op1, &[input_id, bias_id]);
    let (_, op2_out) = g.add_simple_op(
        "op2",
        op2,
        // Note here the input ordering. The bias value is smaller, but
        // is the first argument. This operator can run in place, but only
        // if the inputs are swapped.
        &[bias_id, op1_out],
    );
    let input = Tensor::<f32>::zeros(&[2, 2]);
    let bias = Tensor::from(1.5);

    let results = g
        .run(
            vec![(input_id, input.view().into()), (bias_id, bias.into())],
            &[op2_out],
            None,
            None,
        )
        .unwrap();

    // Bias value should be added twice to every input.
    assert_eq!(
        results[0]
            .as_tensor_view::<f32>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &[3., 3., 3., 3.]
    );

    // The first operator must copy its input because it is a view.
    let op1_metrics = op1_metrics.lock().unwrap();
    assert_eq!(op1_metrics.run_count, 1);
    assert_eq!(op1_metrics.run_in_place_count, 0);

    // The second operator should run in-place.
    let op2_metrics = op2_metrics.lock().unwrap();
    assert_eq!(op2_metrics.run_count, 0);
    assert_eq!(op2_metrics.run_in_place_count, 1);
}

/// Test operator that produces multiple outputs
#[derive(Debug)]
struct Split {
    run_count: Arc<Mutex<u32>>,
}

impl Split {
    fn new() -> Split {
        Split {
            run_count: Arc::new(Mutex::new(0)),
        }
    }
}

impl Operator for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        {
            let mut rc = self.run_count.lock().unwrap();
            *rc += 1;
        }

        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        let left_split_len = input.len() / 2;
        let left_split = Tensor::from_vec(input.iter().take(left_split_len).copied().collect());
        let right_split = Tensor::from_vec(input.iter().skip(left_split_len).copied().collect());
        Ok(smallvec![left_split.into(), right_split.into()])
    }
}

#[test]
fn test_multiple_outputs() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);
    let left_split_out = g.add_value(Some("left_split"), None, None);
    let right_split_out = g.add_value(Some("right_split"), None, None);

    let split_op = Box::new(Split::new());
    let run_count = split_op.run_count.clone();

    g.add_op(
        Some("split"),
        split_op,
        &[Some(input_id)],
        &[left_split_out, right_split_out].map(Some),
    );

    let input = Tensor::from([1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut results = g
        .run(
            vec![(input_id, input.into())],
            &[left_split_out, right_split_out],
            None,
            None,
        )
        .unwrap();

    assert_eq!(*run_count.lock().unwrap(), 1);

    assert_eq!(results.len(), 2);
    let left_split = results.remove(0).into_tensor::<f32>().unwrap();
    let right_split = results.remove(0).into_tensor::<f32>().unwrap();
    assert_eq!(left_split.to_vec(), &[1.0, 2.0]);
    assert_eq!(right_split.to_vec(), &[3.0, 4.0, 5.0]);
}

#[test]
fn test_partial_run() -> Result<(), Box<dyn Error>> {
    // Set up graph like:
    //
    // C0, V0 --> Op0 --> Op2 --> [Output]
    // C1, V1 --> Op1 --^
    //
    // Where `Cn` are constants, `Vn` are input values and `OpN` are
    // operators.
    let mut g = Graph::new();
    let const_0 = g.add_constant(Some("c0"), Tensor::from(3.));
    let val_0 = g.add_value(Some("i0"), None, None);
    let const_1 = g.add_constant(Some("c1"), Tensor::from(4.));
    let val_1 = g.add_value(Some("i1"), None, None);

    let (_, op_0_out) = g.add_simple_op("Add_0", Add {}, &[const_0, val_0]);
    let (_, op_1_out) = g.add_simple_op("Add_1", Add {}, &[const_1, val_1]);
    let (_, op_2_out) = g.add_simple_op("Add_2", Add {}, &[op_0_out, op_1_out]);

    // Run graph with no inputs. This is equivalent to constant evaluation.
    // In this case no operators can be evaluated with graph constants
    // alone, so the output is empty.
    let partial_outs = g.partial_run(vec![], &[op_2_out], None)?;
    assert_eq!(partial_outs.len(), 0);

    // Run graph with just the `V0` input. This will compute the result of
    // `Op0` but not other nodes which depend on `V1`.
    let input = Tensor::from(2.);
    let partial_outs = g.partial_run(vec![(val_0, input.view().into())], &[op_2_out], None)?;
    assert_eq!(partial_outs.len(), 1);
    assert_eq!(partial_outs[0].0, op_0_out);
    assert_eq!(partial_outs[0].1, Value::FloatTensor(Tensor::from(5.)));

    // Run graph with just the `V1` input. This will compute the result of
    // `Op1` but not other nodes which depend on `V0`.
    let input = Tensor::from(2.);
    let partial_outs = g.partial_run(vec![(val_1, input.view().into())], &[op_2_out], None)?;
    assert_eq!(partial_outs.len(), 1);
    assert_eq!(partial_outs[0].0, op_1_out);
    assert_eq!(partial_outs[0].1, Value::FloatTensor(Tensor::from(6.)));

    // Run graph with all inputs. This should behave like `Graph::run`.
    let partial_outs = g.partial_run(
        vec![(val_1, input.view().into()), (val_0, input.view().into())],
        &[op_2_out],
        None,
    )?;
    assert_eq!(partial_outs.len(), 1);
    assert_eq!(partial_outs[0].0, op_2_out);
    assert_eq!(partial_outs[0].1, Value::FloatTensor(Tensor::from(11.)));

    Ok(())
}

#[derive(Debug)]
struct Counter {
    count: AtomicI32,
}

impl Operator for Counter {
    fn name(&self) -> &str {
        "Counter"
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let count = self.count.fetch_add(1, Ordering::SeqCst);
        Ok([Tensor::from(count).into()].into())
    }
}

#[test]
fn test_partial_run_non_deterministic_ops() -> Result<(), Box<dyn Error>> {
    let mut g = Graph::new();
    let const_val = g.add_constant(Some("c0"), Tensor::from(3));

    // Add deterministic op with constant inputs.
    let (_, add_op_0_out) = g.add_simple_op("Add_0", Add {}, &[const_val, const_val]);

    // Add non-deterministic op.
    let (_, count_op_out) = g.add_simple_op(
        "Count",
        Counter {
            count: AtomicI32::new(0),
        },
        &[],
    );

    // Add final op that combines outputs from other ops.
    let (_, add_op_1_out) = g.add_simple_op("Add_1", Add {}, &[add_op_0_out, count_op_out]);

    // Do a partial run with no inputs. This should propagate constants
    // though all the deterministic operators, but skip any
    // non-deterministic ops.
    let partial_outs = g.partial_run(vec![], &[add_op_1_out], None)?;
    assert_eq!(partial_outs.len(), 1);
    assert_eq!(partial_outs[0].0, add_op_0_out);

    Ok(())
}

#[test]
fn test_cached_plan_matches() {
    let input_ids = &[3, 1, 2].map(NodeId::from_u32);
    let output_ids = &[6, 4, 5].map(NodeId::from_u32);
    let op_ids = &[10, 11, 12].map(NodeId::from_u32);

    let plan = CachedPlan::new(input_ids, output_ids, op_ids.to_vec());

    assert!(plan.matches(input_ids, output_ids));

    // Same input and output IDs, different orders.
    assert!(plan.matches(
        &[1, 2, 3].map(NodeId::from_u32),
        &[4, 5, 6].map(NodeId::from_u32)
    ));
    assert!(plan.matches(
        &[3, 2, 1].map(NodeId::from_u32),
        &[6, 5, 4].map(NodeId::from_u32)
    ));

    // Different input and output IDs
    assert!(!plan.matches(&[20, 21, 22].map(NodeId::from_u32), output_ids));
    assert!(!plan.matches(input_ids, &[20, 21, 22].map(NodeId::from_u32)));
}

/// A trivial control flow operator which just forwards inputs to a subgraph
/// and returns its outputs.
struct Subgraph {
    graph: Graph,
}

impl std::fmt::Debug for Subgraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "Subgraph {{ ... }}")
    }
}

impl Operator for Subgraph {
    fn name(&self) -> &str {
        "Subgraph"
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::InvalidValue(
            "operator must be run with `run_subgraph`",
        ))
    }

    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        SmallVec::from_slice(&[&self.graph])
    }

    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        captures: CaptureEnv,
        weight_caches: Option<&[WeightCache]>,
        profiler: Option<&mut Profiler<'a>>,
        options: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        let inputs = self
            .graph
            .input_ids()
            .iter()
            .copied()
            .zip(ctx.inputs().iter().flatten().map(|i| i.into()))
            .collect();
        self.graph
            .run_subgraph(
                inputs,
                self.graph.output_ids(),
                captures,
                Some(ctx.pool()),
                weight_caches.map(|wcs| &wcs[0]),
                profiler,
                options,
            )
            .map(|xs| xs.into_iter().collect())
    }
}

#[test]
fn test_subgraph() {
    let mut g = Graph::new();
    let input = g.add_value(Some("input"), None, None);

    // Add subgraphs for `If` operation. These capture `input`.
    let mut then_branch = Graph::new();
    let tb_input = then_branch.add_value(Some("input"), None, None);
    let two = then_branch.add_constant(None, Tensor::from(2.));
    let (_, tb_output) = then_branch.add_simple_op("Mul", Mul {}, &[tb_input, two]);
    then_branch.set_captures(&[tb_input]);
    then_branch.set_output_ids(&[tb_output]);

    let mut else_branch = Graph::new();
    let eb_input = else_branch.add_value(Some("input"), None, None);
    let three = else_branch.add_constant(None, Tensor::from(3.));
    let (_, eb_output) = else_branch.add_simple_op("Mul", Mul {}, &[eb_input, three]);
    else_branch.set_captures(&[eb_input]);
    else_branch.set_output_ids(&[eb_output]);

    // Add `If` operator that runs one of two subgraphs.
    let cond = g.add_value(Some("cond"), None, None);
    let branch = If {
        then_branch,
        else_branch,
    };
    let (_, if_out) = g.add_simple_op("If", branch, &[cond]);

    // Evaluate `then` branch
    let mut result = g
        .run(
            vec![
                (input, Tensor::from(2.).into()),
                (cond, Tensor::from(1).into()),
            ],
            &[if_out],
            None,
            None,
        )
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result, Tensor::from(4.));

    // Evaluate `else` branch
    let mut result = g
        .run(
            vec![
                (input, Tensor::from(2.).into()),
                (cond, Tensor::from(0).into()),
            ],
            &[if_out],
            None,
            None,
        )
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result, Tensor::from(6.));
}

#[test]
fn test_nested_subgraph() {
    let mut g = Graph::new();
    let input = g.add_value(Some("input"), None, None);

    let mut subgraph = Graph::new();

    let mut nested_subgraph = Graph::new();
    let ns_input = nested_subgraph.add_value(Some("input"), None, None);
    nested_subgraph.set_captures(&[ns_input]);
    nested_subgraph.set_output_ids(&[ns_input]);

    let (_, ns_out) = subgraph.add_simple_op(
        "Subgraph",
        Subgraph {
            graph: nested_subgraph,
        },
        &[],
    );
    subgraph.set_output_ids(&[ns_out]);

    let (_, sg_out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    let mut result = g
        .run(
            vec![(input, Tensor::from(2.).into())],
            &[sg_out],
            None,
            None,
        )
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result, Tensor::from(2.));
}

#[test]
fn test_captures_not_available_when_subgraph_is_run_directly() {
    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("input"), None, None);
    subgraph.set_captures(&[sg_input]);
    let (_, sg_add) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
    subgraph.set_output_ids(&[sg_add]);

    // When a subgraph is run via `run_subgraph` the planner will assume
    // that captured values are available. If the graph is run directly
    // however, this is not the case.
    //
    // Cases where subgraphs are run directly include the constant
    // propagation pass of graph optimization.

    let result = subgraph.partial_run(Vec::new(), &[sg_add], None).unwrap();
    assert_eq!(result.len(), 0);

    let result = subgraph.run(Vec::new(), &[sg_add], None, None);
    assert_eq!(
        result,
        Err(RunError::PlanningError(
            "Missing input \"input\" for op \"Id\"".to_string()
        ))
    );
}

#[test]
fn test_partial_run_considers_subgraph_captures() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("input"), None, None);
    subgraph.set_captures(&[sg_input]);
    let (_, sg_add) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
    subgraph.set_output_ids(&[sg_add]);

    let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    // `partial_run` should skip operators that can't be evaluated due to
    // missing captures.
    let result = g.partial_run(Vec::new(), &[out], None).unwrap();
    assert_eq!(result.len(), 0);

    // When the captures are available, `partial_run` should evaluate the
    // operator as normal.
    let result = g
        .partial_run([(input_id, Tensor::from(4.).into())].into(), &[out], None)
        .unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_plan_considers_capture_dependencies() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    let (_, _) = g.add_simple_op("Add", Add {}, &[input_id, input_id]);

    // Add a subgraph with a captured value that is the output of an
    // operation in the parent graph.
    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("Add_out"), None, None);
    subgraph.set_captures(&[sg_input]);
    let (_, sg_out) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
    subgraph.set_output_ids(&[sg_out]);

    let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    // Run the graph. The planner must account for captured dependencies
    // in the `Subgraph` op.
    let input = Tensor::from(3.);
    let mut result = g
        .run(vec![(input_id, input.into())], &[out], None, None)
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result.item(), Some(&6.));
}

#[test]
fn test_plan_considers_transitive_capture_dependencies() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    let (_, _) = g.add_simple_op("Add", Add {}, &[input_id, input_id]);

    // Add nested subgraphs where an operation in the innermost graph has
    // a dependency on an operator output in the top-level graph.
    let mut subgraph = Graph::new();
    let mut nested_subgraph = Graph::new();
    let ns_input = nested_subgraph.add_value(Some("Add_out"), None, None);
    nested_subgraph.set_captures(&[ns_input]);
    let (_, ns_out) = nested_subgraph.add_simple_op("Id", Identity {}, &[ns_input]);
    nested_subgraph.set_output_ids(&[ns_out]);

    let (_, sg_out) = subgraph.add_simple_op(
        "Subgraph",
        Subgraph {
            graph: nested_subgraph,
        },
        &[],
    );
    subgraph.set_output_ids(&[sg_out]);

    let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    // Run the graph. The planner must account for captured dependencies
    // from the innermost graph in the `Subgraph` op.
    let input = Tensor::from(3.);
    let mut result = g
        .run(vec![(input_id, input.into())], &[out], None, None)
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result.item(), Some(&6.));
}

#[test]
fn test_keeps_temp_value_needed_as_subgraph_capture() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    // Compute a temporary `id_out` value and use it in the main graph.
    let (_, id_out) = g.add_simple_op("Id", Identity {}, &[input_id]);
    let (_, mul_out) = g.add_simple_op("Mul", Mul {}, &[id_out, id_out]);

    // Add a subgraph which depends on the temporary `id_out` value via a
    // capture. Graph execution must keep the `id_out` value around until
    // this has run, even though no ops in the main graph need it as inputs.
    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("Id_out"), None, None);
    subgraph.set_captures(&[sg_input]);
    let (_, sg_out) = subgraph.add_simple_op("Id", Identity {}, &[sg_input]);
    subgraph.set_output_ids(&[sg_out]);

    // Add op to main graph which runs the subgraph. This has a dummy
    // dependency on `mul_out`.
    let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[mul_out]);

    let input = Tensor::from(3.);
    let mut result = g
        .run(vec![(input_id, input.into())], &[out], None, None)
        .unwrap();
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result.item(), Some(&3.));
}

#[test]
fn test_captures_by_value_if_possible() {
    // Set up a graph that runs a subgraph and passes captures by value,
    // if the value is passed to the graph as an owned value.
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);

    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("input"), None, None);
    subgraph.set_captures(&[sg_input]);

    let id_op = TrackUsage::new(Identity {});
    let id_op_metrics = id_op.metrics();
    let (_, id_out) = subgraph.add_simple_op("Id", id_op, &[sg_input]);
    subgraph.set_output_ids(&[id_out]);
    let (_, out) = g.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[]);

    // Run graph with an owned value as input.
    let input = Tensor::from(42.);
    let mut result = g
        .run(vec![(input_id, input.into())], &[out], None, None)
        .unwrap();

    // Check result and that Identity operation was run in-place.
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result.item(), Some(&42.));

    {
        let id_op_metrics = id_op_metrics.lock().unwrap();
        assert_eq!(id_op_metrics.run_count, 0);
        assert_eq!(id_op_metrics.run_in_place_count, 1);
    }

    // Run graph with view as input.
    let input = Tensor::from(42.);
    let mut result = g
        .run(vec![(input_id, input.view().into())], &[out], None, None)
        .unwrap();

    // Check result and that Identity operation was not run in-place.
    let result: Tensor<f32> = result.remove(0).try_into().unwrap();
    assert_eq!(result.item(), Some(&42.));

    {
        let id_op_metrics = id_op_metrics.lock().unwrap();
        assert_eq!(id_op_metrics.run_count, 1);
        assert_eq!(id_op_metrics.run_in_place_count, 1);
    }
}

// MatMul wrapper that verifies its B input (ie. the weights) are prepacked.
#[derive(Debug)]
struct MatMulExpectPacked {
    inner: MatMul,
}

impl MatMulExpectPacked {
    fn new() -> Self {
        MatMulExpectPacked { inner: MatMul {} }
    }
}

impl Operator for MatMulExpectPacked {
    fn name(&self) -> &str {
        "MatMulExpectPacked"
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(&self, index: usize, input: ValueView) -> Option<PrepackedInput> {
        self.inner.prepack(index, input)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let prepacked = ctx.inputs().get_prepacked(1);
        assert!(prepacked.is_some());
        self.inner.run(ctx)
    }
}

#[test]
fn test_prepack_weights() {
    // Create a graph and a subgraph, both with operators that can
    // use prepacked weights.
    let mut graph = Graph::new();
    let mut cache = WeightCache::new();

    let input = graph.add_value(Some("input"), None, None);
    let weights = graph.add_constant(None, Tensor::<f32>::zeros(&[10, 7]));
    let (_, matmul_out) =
        graph.add_simple_op("MatMul", MatMulExpectPacked::new(), &[input, weights]);

    let mut subgraph = Graph::new();
    let sg_input = subgraph.add_value(Some("sg-input"), None, None);
    let sg_weights = subgraph.add_constant(None, Tensor::<f32>::zeros(&[7, 5]));
    let (_, sg_matmul_out) = subgraph.add_simple_op(
        "sg-MatMul",
        MatMulExpectPacked::new(),
        &[sg_input, sg_weights],
    );
    subgraph.set_input_ids(&[sg_input]);
    subgraph.set_output_ids(&[sg_matmul_out]);

    let (subgraph_op, subgraph_out) =
        graph.add_simple_op("Subgraph", Subgraph { graph: subgraph }, &[matmul_out]);
    graph.set_input_ids(&[input]);
    graph.set_output_ids(&[subgraph_out]);

    // Prepack weights and verify that the cache was populated.
    graph.prepack_weights(&mut cache);
    assert_eq!(cache.len(), 2);
    assert!(cache.get(weights).is_some());

    let sg_cache = cache
        .get_subgraph_caches(subgraph_op)
        .map(|caches| &caches[0])
        .unwrap();
    assert!(sg_cache.get(sg_weights).is_some());

    // Run the graph, passing the cache. The MatMul wrapper will verify
    // that the B / RHS inputs were passed from the cache.
    let input_value = Tensor::<f32>::zeros(&[3, 10]);
    graph
        .run(
            [(input, input_value.into())].into(),
            &[subgraph_out],
            Some(&cache),
            None,
        )
        .unwrap();
}

#[test]
fn test_run_context_num_outputs() {
    let mut g = Graph::new();
    let input_id = g.add_value(Some("input"), None, None);
    let (_, op_out) = g.add_simple_op(
        "test_op",
        RunFn::new(|ctx| {
            assert_eq!(ctx.num_outputs(), Some(1));
            let output: Value = Tensor::from_scalar(0.).into();
            Ok([output].into())
        }),
        &[input_id],
    );
    let input = Tensor::from([1, 2, 3]);
    g.run(vec![(input_id, input.into())], &[op_out], None, None)
        .unwrap();
}

#[test]
fn test_remove_nodes() {
    let mut g = Graph::new();
    let val_id = g.add_value(Some("value"), None, None);
    g.set_input_ids(&[val_id]);
    g.set_output_ids(&[val_id]);

    assert!(g.get_node(val_id).is_some());
    assert!(g.get_node_id("value").is_some());

    g.remove_nodes(&[val_id]);

    assert!(g.get_node(val_id).is_none());
    assert!(g.get_node_id("value").is_none());
    assert!(g.input_ids().is_empty());
    assert!(g.output_ids().is_empty());

    // Removing an operator should remove it as the source node for its outputs.
    let val_id = g.add_value(Some("value2"), None, None);
    let (op_id, out_id) = g.add_simple_op("Mul", Mul {}, &[val_id, val_id]);
    assert!(g.get_source_node(out_id).is_some());

    g.remove_nodes(&[op_id]);

    assert!(g.get_source_node(out_id).is_none());
}

use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use smallvec::SmallVec;

use crate::buffer_pool::{BufferPool, ExtractBuffer};
use crate::graph::{CaptureEnv, Graph, NodeId, RunError, RunOptions};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    OpError, OpRunContext, Operator, OutputList, OutputTypeList, OutputTypesContext,
    SubgraphOperator,
};
use crate::ops::{map_value, map_value_view, resolve_axis};
use crate::timing::Profiler;
use crate::value::Value;
use crate::value::{ValueOrView, ValueView};
use crate::weight_cache::WeightCache;

fn output_list_from_vec(xs: Vec<Value>) -> OutputList {
    xs.into_iter().collect()
}

pub struct If {
    pub then_branch: Graph,
    pub else_branch: Graph,
}

impl std::fmt::Debug for If {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "If {{ ... }}")
    }
}

impl Operator for If {
    fn name(&self) -> &str {
        "If"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn max_outputs(&self) -> Option<usize> {
        None
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::invalid_value(
            "operator must be run with `run_subgraph`",
        ))
    }

    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        Some(self as &dyn SubgraphOperator)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        // Type inference is not implemented for ops with subgraphs yet.
        None
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        // Shape inference does not support ops with subgraphs yet.
        None
    }
}

impl SubgraphOperator for If {
    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        [&self.then_branch, &self.else_branch].into()
    }

    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        captures: CaptureEnv,
        weight_caches: Option<&[WeightCache]>,
        profiler: Option<&mut Profiler<'a>>,
        run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        let node_name = ctx.name().unwrap_or_default();
        let cond: TensorView<i32> = ctx
            .inputs()
            .require_as(0)
            .map_err(|e| RunError::op_error(node_name, e, ctx))?;
        let Some(cond_bool) = cond.item().copied() else {
            return Err(RunError::op_error(
                node_name,
                OpError::invalid_value("cond must be a single value"),
                ctx,
            ));
        };

        if cond_bool != 0 {
            self.then_branch
                .run_subgraph(
                    Vec::new(),
                    self.then_branch.output_ids(),
                    captures,
                    ctx.pool(),
                    weight_caches.map(|wcs| &wcs[0]),
                    profiler,
                    run_opts,
                )
                .map(output_list_from_vec)
        } else {
            self.else_branch
                .run_subgraph(
                    Vec::new(),
                    self.else_branch.output_ids(),
                    captures,
                    ctx.pool(),
                    weight_caches.map(|wcs| &wcs[1]),
                    profiler,
                    run_opts,
                )
                .map(output_list_from_vec)
        }
    }
}

pub struct Loop {
    pub body: Graph,
}

impl std::fmt::Debug for Loop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "Loop {{ ... }}")
    }
}

impl Operator for Loop {
    fn name(&self) -> &str {
        "Loop"
    }

    fn max_inputs(&self) -> Option<usize> {
        None
    }

    fn max_outputs(&self) -> Option<usize> {
        None
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::invalid_value(
            "operator must be run with `run_subgraph`",
        ))
    }

    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        Some(self as &dyn SubgraphOperator)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        // Type inference is not implemented for ops with subgraphs yet.
        None
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        // Shape inference does not support ops with subgraphs yet.
        None
    }
}

impl SubgraphOperator for Loop {
    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        SmallVec::from_slice(&[&self.body])
    }

    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        captures: CaptureEnv,
        weight_caches: Option<&[WeightCache]>,
        mut profiler: Option<&mut Profiler<'a>>,
        run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        let node_name = ctx.name().unwrap_or_default();

        // Create a `RunError` from an `OpError`
        let make_run_error = |err| RunError::op_error(node_name, err, ctx);

        let trip_count: Option<i32> = ctx.inputs().get_as(0).map_err(make_run_error)?;
        let trip_count = trip_count.unwrap_or(i32::MAX);

        let cond: Option<i32> = ctx
            .inputs()
            .get_as(1)
            .map_err(|err| RunError::op_error(node_name, err, ctx))?;
        let mut cond = cond.unwrap_or(1);

        let mut loop_carried_deps: Vec<ValueOrView> = ctx
            .inputs()
            .iter()
            .skip(2)
            .flatten()
            .map(|val| val.into())
            .collect();
        let carried_deps_len = loop_carried_deps.len();

        let input_ids = self.body.input_ids();
        if input_ids.len() != 2 + loop_carried_deps.len() {
            return Err(make_run_error(OpError::invalid_value(
                "loop body has too few inputs",
            )));
        }

        let output_ids = self.body.output_ids();
        if output_ids.len() < 1 + loop_carried_deps.len() {
            return Err(make_run_error(OpError::invalid_value(
                "loop body has too few outputs",
            )));
        }

        let scan_output_len = output_ids.len() - 1 - loop_carried_deps.len();
        let mut scan_outputs: Vec<Vec<Value>> = (0..scan_output_len).map(|_| Vec::new()).collect();

        let mut step_index = 0usize;
        while (step_index as i32) < trip_count && cond != 0 {
            let mut step_inputs: Vec<(NodeId, ValueOrView)> = Vec::with_capacity(input_ids.len());
            step_inputs.push((input_ids[0], Tensor::from(step_index as i32).into()));
            step_inputs.push((input_ids[1], Tensor::from(cond).into()));
            for (node_id, dep) in input_ids.iter().skip(2).zip(loop_carried_deps.drain(..)) {
                step_inputs.push((*node_id, dep));
            }

            let mut step_outputs = self.body.run_subgraph(
                step_inputs,
                output_ids,
                captures.clone(),
                ctx.pool(),
                weight_caches.map(|wcs| &wcs[0]),
                profiler.as_deref_mut(),
                run_opts.clone(),
            )?;

            // `step_outputs` format is `[condition, loop_carried_dependencies...,
            // scan_outputs...]`.

            // Extract condition.
            let next_cond: Tensor<i32> = step_outputs.remove(0).try_into().map_err(|_| {
                make_run_error(OpError::invalid_value(
                    "condition output has incorrect type",
                ))
            })?;
            let Some(&next_cond) = next_cond.item() else {
                return Err(make_run_error(OpError::invalid_value(
                    "condition output should be a scalar",
                )));
            };
            cond = next_cond;

            // Extract loop-carried dependencies.
            loop_carried_deps.extend(
                step_outputs
                    .drain(..carried_deps_len)
                    .map(|value| value.into()),
            );

            // Extract scan outputs.
            for (i, scan_output) in step_outputs.into_iter().enumerate() {
                scan_outputs[i].push(scan_output);
            }

            step_index += 1;
        }

        // Construct output sequence consisting of final loop carried
        // dependencies followed by concatenated scan outputs.
        let mut outputs: Vec<Value> = Vec::with_capacity(loop_carried_deps.len() + scan_output_len);
        outputs.extend(loop_carried_deps.into_iter().map(|dep| dep.into_owned()));

        for mut output_seq in scan_outputs.into_iter() {
            if output_seq.is_empty() {
                continue;
            }
            let first = output_seq.remove(0);

            // Stack outputs. This can fail if the outputs have different
            // shapes or the value is not a tensor.
            map_value!(first, first, {
                stack_scan_outputs(ctx.pool(), first, output_seq, 0)
                    .map(|out| outputs.push(out.into()))
            })
            .map_err(make_run_error)?;
        }

        Ok(outputs.into())
    }
}

/// Direction in which a [`Scan`] operator reads a scan input or builds a scan
/// output.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum ScanDirection {
    /// Read elements from the first to the last, or append each output element
    /// after the previous one.
    #[default]
    Forward,

    /// Read elements from the last to the first, or prepend each output element
    /// before the previous one.
    Reverse,
}

/// Iterate over one or more input tensors, running a subgraph for each element.
///
/// The inputs are the initial values of the loop-carried state variables,
/// followed by [`num_scan_inputs`](Scan::num_scan_inputs) tensors which are
/// scanned over. The outputs are the final values of the state variables,
/// followed by the per-iteration body outputs stacked into a tensor.
///
/// See <https://onnx.ai/onnx/operators/onnx__Scan.html>. The opset 8 revision of
/// this operator, which has an extra `sequence_lens` input and a batch
/// dimension, is not supported.
pub struct Scan {
    /// Subgraph run once per iteration.
    ///
    /// Its inputs are the current state variables followed by one element from
    /// each scan input. Its outputs are the updated state variables followed by
    /// one element of each scan output.
    pub body: Graph,

    /// Number of trailing inputs which are scanned over.
    pub num_scan_inputs: usize,

    /// Axis of each scan input which is iterated over.
    ///
    /// If empty, every scan input is iterated over axis zero.
    pub scan_input_axes: Vec<i32>,

    /// Direction in which each scan input is read.
    ///
    /// If empty, every scan input is read forwards.
    pub scan_input_directions: Vec<ScanDirection>,

    /// Axis of each scan output along which the per-iteration elements are
    /// stacked.
    ///
    /// If empty, the elements of every scan output are stacked along axis zero.
    pub scan_output_axes: Vec<i32>,

    /// Direction in which each scan output is built.
    ///
    /// If empty, every scan output is built forwards.
    pub scan_output_directions: Vec<ScanDirection>,
}

impl std::fmt::Debug for Scan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "Scan {{ ... }}")
    }
}

impl Operator for Scan {
    fn name(&self) -> &str {
        "Scan"
    }

    fn max_inputs(&self) -> Option<usize> {
        None
    }

    fn max_outputs(&self) -> Option<usize> {
        None
    }

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::invalid_value(
            "operator must be run with `run_subgraph`",
        ))
    }

    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        Some(self as &dyn SubgraphOperator)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        // Type inference is not implemented for ops with subgraphs yet.
        None
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        // Shape inference does not support ops with subgraphs yet.
        None
    }
}

impl SubgraphOperator for Scan {
    fn subgraphs(&self) -> SmallVec<[&Graph; 2]> {
        SmallVec::from_slice(&[&self.body])
    }

    fn run_subgraph<'a>(
        &'a self,
        ctx: &OpRunContext,
        captures: CaptureEnv,
        weight_caches: Option<&[WeightCache]>,
        mut profiler: Option<&mut Profiler<'a>>,
        run_opts: Option<RunOptions>,
    ) -> Result<OutputList, RunError> {
        let node_name = ctx.name().unwrap_or_default();

        // Create a `RunError` from an `OpError`
        let make_run_error = |err| RunError::op_error(node_name, err, ctx);

        let inputs = ctx.inputs();

        // The scan inputs determine the iteration count, so there must be at
        // least one of them.
        if self.num_scan_inputs == 0 || inputs.len() < self.num_scan_inputs {
            return Err(make_run_error(OpError::MissingInputs));
        }
        let n_state_vars = inputs.len() - self.num_scan_inputs;

        let body_input_ids = self.body.input_ids();
        if body_input_ids.len() != inputs.len() {
            return Err(make_run_error(OpError::invalid_value(
                "scan body has incorrect number of inputs",
            )));
        }

        let body_output_ids = self.body.output_ids();
        if body_output_ids.len() < n_state_vars {
            return Err(make_run_error(OpError::invalid_value(
                "scan body has too few outputs",
            )));
        }
        let n_scan_outputs = body_output_ids.len() - n_state_vars;

        check_attr_len(
            &self.scan_input_axes,
            self.num_scan_inputs,
            "scan_input_axes has incorrect length",
        )
        .map_err(make_run_error)?;
        check_attr_len(
            &self.scan_input_directions,
            self.num_scan_inputs,
            "scan_input_directions has incorrect length",
        )
        .map_err(make_run_error)?;
        check_attr_len(
            &self.scan_output_axes,
            n_scan_outputs,
            "scan_output_axes has incorrect length",
        )
        .map_err(make_run_error)?;
        check_attr_len(
            &self.scan_output_directions,
            n_scan_outputs,
            "scan_output_directions has incorrect length",
        )
        .map_err(make_run_error)?;

        // Resolve the axis and direction of each scan input, and determine the
        // number of iterations.
        let mut scan_inputs = Vec::with_capacity(self.num_scan_inputs);
        let mut sequence_len = None;
        for i in 0..self.num_scan_inputs {
            let index = n_state_vars + i;
            let input = inputs.require(index).map_err(make_run_error)?;
            let axis = attr_for_index(&self.scan_input_axes, i);
            let axis = resolve_axis(input.ndim(), axis as isize)
                .map_err(|err| make_run_error(err.with_input_index(index)))?;
            let direction = attr_for_index(&self.scan_input_directions, i);

            let len = input.size(axis);
            match sequence_len {
                Some(expected) if expected != len => {
                    return Err(make_run_error(OpError::incompatible_input_shapes(
                        "scan inputs have different sizes along their scan axis",
                    )));
                }
                _ => sequence_len = Some(len),
            }

            scan_inputs.push((input, axis, direction));
        }
        let sequence_len = sequence_len.expect("scan should have at least one input");

        // The shape and type of a scan output is only known once the body has
        // run, so an empty scan can only be evaluated if there are none.
        if sequence_len == 0 && n_scan_outputs > 0 {
            return Err(make_run_error(OpError::unsupported_value(
                "scan inputs are empty and scan has outputs",
            )));
        }

        let mut state: Vec<ValueOrView> = Vec::with_capacity(n_state_vars);
        for i in 0..n_state_vars {
            state.push(inputs.require(i).map_err(make_run_error)?.into());
        }

        let mut scan_output_elements: Vec<Vec<Value>> = (0..n_scan_outputs)
            .map(|_| Vec::with_capacity(sequence_len))
            .collect();

        for step in 0..sequence_len {
            let mut step_inputs: Vec<(NodeId, ValueOrView)> =
                Vec::with_capacity(body_input_ids.len());
            for (node_id, value) in body_input_ids.iter().zip(state.drain(..)) {
                step_inputs.push((*node_id, value));
            }
            for (i, (input, axis, direction)) in scan_inputs.iter().enumerate() {
                let index = match direction {
                    ScanDirection::Forward => step,
                    ScanDirection::Reverse => sequence_len - 1 - step,
                };
                let element = index_axis(input.clone(), *axis, index).map_err(make_run_error)?;
                step_inputs.push((body_input_ids[n_state_vars + i], element.into()));
            }

            let mut step_outputs = self.body.run_subgraph(
                step_inputs,
                body_output_ids,
                captures.clone(),
                ctx.pool(),
                weight_caches.map(|wcs| &wcs[0]),
                profiler.as_deref_mut(),
                run_opts.clone(),
            )?;

            // `step_outputs` format is `[state_variables..., scan_output_elements...]`.
            for (i, element) in step_outputs.drain(n_state_vars..).enumerate() {
                scan_output_elements[i].push(element);
            }
            state.extend(step_outputs.into_iter().map(|value| value.into()));
        }

        // Construct output sequence consisting of the final state variables
        // followed by the stacked scan outputs.
        let mut outputs: Vec<Value> = Vec::with_capacity(n_state_vars + n_scan_outputs);
        outputs.extend(state.into_iter().map(|value| value.into_owned()));

        for (i, mut elements) in scan_output_elements.into_iter().enumerate() {
            let axis = attr_for_index(&self.scan_output_axes, i);

            // Put the elements in the order they appear in the output.
            if attr_for_index(&self.scan_output_directions, i) == ScanDirection::Reverse {
                elements.reverse();
            }

            // The sequence is non-empty because `sequence_len` is non-zero
            // whenever there are scan outputs.
            let first = elements.remove(0);

            // The scan output has one more dimension than the elements produced
            // by each iteration.
            let axis = resolve_axis(first.ndim() + 1, axis as isize).map_err(make_run_error)?;

            map_value!(first, first, {
                stack_scan_outputs(ctx.pool(), first, elements, axis)
                    .map(|out| outputs.push(out.into()))
            })
            .map_err(make_run_error)?;
        }

        Ok(outputs.into())
    }
}

/// Check that a per-input or per-output [`Scan`] attribute is either empty,
/// meaning the default applies to every entry, or has one entry per scan input
/// or output.
fn check_attr_len<T>(values: &[T], expected: usize, err: &'static str) -> Result<(), OpError> {
    if values.is_empty() || values.len() == expected {
        Ok(())
    } else {
        Err(OpError::invalid_value(err))
    }
}

/// Get the entry of a per-input or per-output [`Scan`] attribute whose length
/// has been checked with [`check_attr_len`].
fn attr_for_index<T: Copy + Default>(values: &[T], index: usize) -> T {
    values.get(index).copied().unwrap_or_default()
}

/// Index a value along `axis`, returning a view with that axis removed.
fn index_axis(value: ValueView<'_>, axis: usize, index: usize) -> Result<ValueView<'_>, OpError> {
    map_value_view!(value, value, { Ok(value.index_axis(axis, index).into()) })
}

/// Stack the per-iteration elements of a scan output into a single tensor.
///
/// The elements are stacked, in order, along `axis`, which is a dimension of
/// the result and hence in `[0, first.ndim()]`.
///
/// Elements from every iteration must have the same type and shape.
fn stack_scan_outputs<T: Copy>(
    pool: &BufferPool,
    first: Tensor<T>,
    rest: Vec<Value>,
    axis: usize,
) -> Result<Tensor<T>, OpError>
where
    Tensor<T>: TryFrom<Value>,
{
    fn append_element<T: Copy>(
        pool: &BufferPool,
        output: &mut Tensor<T>,
        axis: usize,
        mut element: Tensor<T>,
    ) -> Result<(), OpError> {
        const SHAPE_MISMATCH: &str = "scan output has different shape across iterations";

        if element.ndim() + 1 != output.ndim() {
            return Err(OpError::invalid_value(SHAPE_MISMATCH));
        }
        element.insert_axis(axis);
        output
            .append(axis, &element)
            .map_err(|_| OpError::invalid_value(SHAPE_MISMATCH))?;
        if let Some(buf) = element.extract_buffer() {
            pool.add(buf);
        }
        Ok(())
    }

    let sequence_len = 1 + rest.len();
    let mut shape = first.shape().to_vec();
    shape.insert(axis, sequence_len);
    let mut output = Tensor::with_capacity_in(pool, &shape, axis);

    append_element(pool, &mut output, axis, first)?;
    for element in rest {
        let element = element.try_into().map_err(|_| {
            OpError::invalid_value("scan output has different type across iterations")
        })?;
        append_element(pool, &mut output, axis, element)?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;

    use crate::buffer_pool::BufferPool;
    use crate::graph::builder::{Expr, OutputMeta};
    use crate::graph::{CaptureEnv, Graph, RunError, RunErrorKind};
    use crate::operator::{InputList, OpRunContext, OutputMask, SubgraphOperator};
    use crate::ops::{Identity, Range, Squeeze};
    use crate::value::{Scalar, Value, ValueView};

    use super::{Loop, Scan, ScanDirection};

    /// Wraps a `Loop` operator to simplify running it.
    struct LoopRunner {
        op: Loop,
    }

    impl LoopRunner {
        fn new(op: Loop) -> Self {
            Self { op }
        }

        fn run(
            &self,
            max_iterations: Option<i32>,
            cond: Option<bool>,
            inputs: &[ValueView],
        ) -> Result<Vec<Value>, RunError> {
            let max_iter_tensor = max_iterations.map(Tensor::from);
            let cond_tensor = cond.map(|c| if c { 1i32 } else { 0i32 }).map(Tensor::from);

            let input_list = InputList::from_iter(
                [
                    max_iter_tensor.as_ref().map(ValueView::from),
                    cond_tensor.as_ref().map(ValueView::from),
                ]
                .into_iter()
                .chain(inputs.into_iter().cloned().map(Some)),
            );

            let pool = BufferPool::new();
            // A `Loop` produces one output per body output, except the first
            // body output which is the loop condition.
            let num_outputs = self.op.body.output_ids().len().saturating_sub(1);
            let ctx = OpRunContext::new(&pool, &input_list, OutputMask::all_used(num_outputs));
            let captures = CaptureEnv::empty();
            let weight_caches = None;
            let profiler = None;
            let run_opts = None;
            self.op
                .run_subgraph(&ctx, captures, weight_caches, profiler, run_opts)
                .map(|v| v.to_vec())
        }
    }

    #[test]
    fn test_loop_scan_outputs() {
        let body = {
            let x = Expr::value("i");
            let cond = Expr::value("cond");
            let x_2 = x.clone() * x.clone();
            Expr::make_graph([x, cond.clone()], [cond, x_2])
        };
        let runner = LoopRunner::new(Loop { body });
        let mut outputs = runner.run(Some(5), None, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        let squares: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        // Output should be concatenated values of `x_2` from each iteration.
        assert_eq!(squares, Tensor::from([0, 1, 4, 9, 16]));
    }

    // Test where loop condition is initially true and becomes false after a
    // certain iteration.
    #[test]
    fn test_loop_condition() {
        let body = {
            let x = Expr::value("i");
            let cond = Expr::value("cond");

            // Add termination condition which stops after third loop iteration.
            let next_cond = x.binary(crate::ops::Less {}, Expr::constant(2));

            let x_2 = x.clone() * x.clone();
            Expr::make_graph([x, cond], [next_cond, x_2])
        };
        let runner = LoopRunner::new(Loop { body });
        let mut outputs = runner.run(Some(5), None, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        let squares: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        // Output should be concatenated values of `x_2` from each iteration.
        assert_eq!(squares, Tensor::from([0, 1, 4]));
    }

    #[test]
    fn test_loop_condition_initially_false() {
        let body = {
            let x = Expr::value("i");
            let cond = Expr::value("cond");
            let x_2 = x.clone() * x.clone();
            Expr::make_graph([x, cond.clone()], [cond, x_2])
        };
        let runner = LoopRunner::new(Loop { body });
        let outputs = runner.run(Some(5), Some(false), &[]).unwrap();

        // Since the condition is initially false, the loop never runs and
        // there are no scan outputs. If initial values have been provided for
        // loop-carried dependencies, those will still be returned as outputs.
        assert_eq!(outputs.len(), 0);
    }

    #[test]
    fn test_loop_carried_deps() {
        let fibonacci_body = {
            let iter = Expr::value("i");
            let cond = Expr::value("cond");

            let prev_x_0 = Expr::value("x0");
            let prev_x_1 = Expr::value("x1");

            let x_0 = prev_x_0.clone() + prev_x_1.clone();
            let x_1 = prev_x_1.clone() + x_0.clone();

            Expr::make_graph([iter, cond.clone(), prev_x_0, prev_x_1], [cond, x_0, x_1])
        };
        let runner = LoopRunner::new(Loop {
            body: fibonacci_body,
        });

        let fib_seq = [Tensor::from(0), Tensor::from(1)];
        let fib_seq = [ValueView::from(&fib_seq[0]), ValueView::from(&fib_seq[1])];

        let mut outputs = runner.run(Some(3), None, &fib_seq).unwrap();
        assert_eq!(outputs.len(), 2);

        let final_x_0: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        let final_x_1: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        // Output should be values of x_0 and x_1 from last iteration.
        //
        // Fibonacci sequence is 0 1 1 2 3 5 8 13... Each iteration produces two
        // new values and the first two values are provided as inputs, so after
        // the third iteration the loop will yield (8, 13).
        assert_eq!(final_x_0, Tensor::from(8));
        assert_eq!(final_x_1, Tensor::from(13));
    }

    #[test]
    fn test_loop_invalid() {
        struct Case {
            body: Graph,
            expected: &'static str,
        }

        impl std::fmt::Debug for Case {
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                fmt.debug_struct("Case")
                    .field("expected", &self.expected)
                    .finish()
            }
        }

        let cases = [
            Case {
                body: Expr::make_graph([], []),
                expected: "loop body has too few inputs",
            },
            Case {
                body: {
                    let x = Expr::value("x");
                    let cond = Expr::value("cond");
                    Expr::make_graph([x, cond], [])
                },
                expected: "loop body has too few outputs",
            },
            Case {
                body: {
                    let x = Expr::value("x");
                    let cond = Expr::value("cond");
                    let next_cond = Expr::constant(5.);
                    Expr::make_graph([x, cond], [next_cond])
                },
                expected: "condition output has incorrect type",
            },
            Case {
                body: {
                    let x = Expr::value("x");
                    let cond = Expr::value("cond");
                    let next_cond = Expr::constant(Tensor::from([1, 2, 3]));
                    Expr::make_graph([x, cond], [next_cond])
                },
                expected: "condition output should be a scalar",
            },
            Case {
                body: {
                    let iter = Expr::value("x");
                    let cond = Expr::value("cond");

                    // Create scan output which is a vector of length equal to
                    // the iteration index.
                    let iter_vec = iter.clone() + Expr::constant(Tensor::from([0]));
                    let output = iter_vec.unary(crate::ops::ConstantOfShape {
                        value: Scalar::Int32(1),
                    });

                    Expr::make_graph([iter, cond.clone()], [cond, output])
                },
                expected: "scan output has different shape across iterations",
            },
            // TODO: Scan outputs with different types across iterations
        ];

        for Case { body, expected } in cases {
            let runner = LoopRunner::new(Loop { body });
            let err = runner.run(Some(3), None, &[]).err().unwrap();
            assert_eq!(err.kind(), RunErrorKind::OperatorError);
            assert!(
                err.to_string().contains(expected),
                "expected {} to contain {}",
                err.to_string(),
                expected
            );
        }
    }

    /// Wraps a `Scan` operator to simplify running it.
    struct ScanRunner {
        op: Scan,
    }

    impl ScanRunner {
        fn new(op: Scan) -> Self {
            Self { op }
        }

        fn run(&self, inputs: &[ValueView]) -> Result<Vec<Value>, RunError> {
            let input_list = InputList::from_iter(inputs.iter().cloned().map(Some));
            let pool = BufferPool::new();

            // A `Scan` produces one output per body output.
            let num_outputs = self.op.body.output_ids().len();
            let ctx = OpRunContext::new(&pool, &input_list, OutputMask::all_used(num_outputs));

            self.op
                .run_subgraph(&ctx, CaptureEnv::empty(), None, None, None)
                .map(|v| v.to_vec())
        }
    }

    /// Create a `Scan` operator which uses the default value for every optional
    /// attribute.
    fn scan(body: Graph, num_scan_inputs: usize) -> Scan {
        Scan {
            body,
            num_scan_inputs,
            scan_input_axes: Vec::new(),
            scan_input_directions: Vec::new(),
            scan_output_axes: Vec::new(),
            scan_output_directions: Vec::new(),
        }
    }

    /// Create a body which adds each scan element to a running total, emitting
    /// the total as both the new state and a scan output.
    ///
    /// The scan output is aliased via `Identity` because the graph builder
    /// deduplicates outputs.
    fn cumulative_sum_body() -> Graph {
        let total = Expr::value("total");
        let x = Expr::value("x");
        let new_total = total.clone() + x.clone();
        let scan_output = new_total.unary(Identity {});
        Expr::make_graph([total, x], [new_total, scan_output])
    }

    #[test]
    fn test_scan_state_and_output() {
        let runner = ScanRunner::new(scan(cumulative_sum_body(), 1));

        let total = Tensor::from(0);
        let x = Tensor::from([1, 2, 3, 4]);
        let mut outputs = runner
            .run(&[ValueView::from(&total), ValueView::from(&x)])
            .unwrap();
        assert_eq!(outputs.len(), 2);

        let final_total: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        let sums: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        assert_eq!(final_total, Tensor::from(10));
        assert_eq!(sums, Tensor::from([1, 3, 6, 10]));
    }

    #[test]
    fn test_scan_reverse_input_direction() {
        let mut op = scan(cumulative_sum_body(), 1);
        op.scan_input_directions = vec![ScanDirection::Reverse];
        let runner = ScanRunner::new(op);

        let total = Tensor::from(0);
        let x = Tensor::from([1, 2, 3, 4]);
        let mut outputs = runner
            .run(&[ValueView::from(&total), ValueView::from(&x)])
            .unwrap();

        let final_total: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        let sums: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        // The input is read as 4, 3, 2, 1.
        assert_eq!(final_total, Tensor::from(10));
        assert_eq!(sums, Tensor::from([4, 7, 9, 10]));
    }

    #[test]
    fn test_scan_reverse_output_direction() {
        let mut op = scan(cumulative_sum_body(), 1);
        op.scan_output_directions = vec![ScanDirection::Reverse];
        let runner = ScanRunner::new(op);

        let total = Tensor::from(0);
        let x = Tensor::from([1, 2, 3, 4]);
        let mut outputs = runner
            .run(&[ValueView::from(&total), ValueView::from(&x)])
            .unwrap();

        let final_total: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        let sums: Tensor<i32> = outputs.remove(0).try_into().unwrap();

        // The same values as `test_scan_state_and_output`, but each element is
        // prepended rather than appended.
        assert_eq!(final_total, Tensor::from(10));
        assert_eq!(sums, Tensor::from([10, 6, 3, 1]));
    }

    /// Create a body with no state variables which doubles each scan element.
    fn double_body() -> Graph {
        let x = Expr::value("x");
        let doubled = x.clone() + x.clone();
        Expr::make_graph([x], [doubled])
    }

    /// Create a body whose scan output rank depends on the scan element.
    ///
    /// The output is `squeeze(range(0, x, 1))`, which has rank 1 if `x > 1` and
    /// rank 0 if `x == 1`.
    fn variable_rank_body() -> Graph {
        let x = Expr::value("x");
        let zero = Expr::constant(Tensor::from(0));
        let one = Expr::constant(Tensor::from(1));
        let range = zero.apply(Range {}, &[x.clone(), one], &[OutputMeta::NoMeta]);
        let squeezed = range.unary(Squeeze {});
        Expr::make_graph([x], [squeezed])
    }

    #[test]
    fn test_scan_input_and_output_axes() {
        struct Case {
            input_axis: i32,
            output_axis: i32,
            expected: Tensor<i32>,
        }

        let cases = [
            // Scan over rows, stack the results as rows.
            Case {
                input_axis: 0,
                output_axis: 0,
                expected: Tensor::from([[2, 4, 6], [8, 10, 12]]),
            },
            // Scan over columns, stack the results as rows.
            Case {
                input_axis: 1,
                output_axis: 0,
                expected: Tensor::from([[2, 8], [4, 10], [6, 12]]),
            },
            // Scan over columns, stack the results as columns.
            Case {
                input_axis: 1,
                output_axis: 1,
                expected: Tensor::from([[2, 4, 6], [8, 10, 12]]),
            },
            // Negative axes count back from the last dimension.
            Case {
                input_axis: -1,
                output_axis: -1,
                expected: Tensor::from([[2, 4, 6], [8, 10, 12]]),
            },
        ];

        for Case {
            input_axis,
            output_axis,
            expected,
        } in cases
        {
            let mut op = scan(double_body(), 1);
            op.scan_input_axes = vec![input_axis];
            op.scan_output_axes = vec![output_axis];
            let runner = ScanRunner::new(op);

            let x = Tensor::from([[1, 2, 3], [4, 5, 6]]);
            let mut outputs = runner.run(&[ValueView::from(&x)]).unwrap();
            assert_eq!(outputs.len(), 1);

            let doubled: Tensor<i32> = outputs.remove(0).try_into().unwrap();
            assert_eq!(doubled, expected);
        }
    }

    #[test]
    fn test_scan_multiple_inputs() {
        let body = {
            let x = Expr::value("x");
            let y = Expr::value("y");
            let sum = x.clone() + y.clone();
            Expr::make_graph([x, y], [sum])
        };
        let runner = ScanRunner::new(scan(body, 2));

        let x = Tensor::from([1, 2, 3]);
        let y = Tensor::from([10, 20, 30]);
        let mut outputs = runner
            .run(&[ValueView::from(&x), ValueView::from(&y)])
            .unwrap();
        assert_eq!(outputs.len(), 1);

        let sums: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        assert_eq!(sums, Tensor::from([11, 22, 33]));
    }

    #[test]
    fn test_scan_invalid() {
        struct Case {
            op: Scan,
            inputs: Vec<Tensor<i32>>,
            expected: &'static str,
        }

        let cases = [
            Case {
                op: scan(cumulative_sum_body(), 1),
                inputs: [Tensor::from([1, 2, 3])].into(),
                expected: "scan body has incorrect number of inputs",
            },
            Case {
                // The body must emit a value for the state variable.
                op: scan(
                    Expr::make_graph([Expr::value("total"), Expr::value("x")], []),
                    1,
                ),
                inputs: [Tensor::from(0), Tensor::from([1, 2, 3])].into(),
                expected: "scan body has too few outputs",
            },
            Case {
                op: {
                    let mut op = scan(double_body(), 1);
                    op.scan_input_axes = vec![2];
                    op
                },
                inputs: [Tensor::from([1, 2, 3])].into(),
                expected: "Axis 2 is out of range. Must be in [-1, 1)",
            },
            Case {
                op: {
                    let mut op = scan(double_body(), 1);
                    op.scan_input_directions = vec![ScanDirection::Forward; 2];
                    op
                },
                inputs: [Tensor::from([1, 2, 3])].into(),
                expected: "scan_input_directions has incorrect length",
            },
            Case {
                op: {
                    let body = {
                        let x = Expr::value("x");
                        let y = Expr::value("y");
                        let sum = x.clone() + y.clone();
                        Expr::make_graph([x, y], [sum])
                    };
                    scan(body, 2)
                },
                inputs: [Tensor::from([1, 2, 3]), Tensor::from([1, 2])].into(),
                expected: "scan inputs have different sizes along their scan axis",
            },
            Case {
                op: scan(double_body(), 1),
                inputs: [Tensor::from([0i32; 0])].into(),
                expected: "scan inputs are empty and scan has outputs",
            },
            // Scan output elements whose rank varies across iterations. The
            // output axis is only valid for the first element's rank.
            Case {
                op: {
                    let mut op = scan(variable_rank_body(), 1);
                    op.scan_output_axes = vec![1];
                    op
                },
                inputs: [Tensor::from([3, 1])].into(),
                expected: "scan output has different shape across iterations",
            },
        ];

        for Case {
            op,
            inputs,
            expected,
        } in cases
        {
            let runner = ScanRunner::new(op);
            let inputs: Vec<ValueView> = inputs.iter().map(ValueView::from).collect();
            let err = runner
                .run(&inputs)
                .err()
                .unwrap_or_else(|| panic!("expected error containing \"{}\"", expected));
            assert_eq!(err.kind(), RunErrorKind::OperatorError);

            let err = err.to_string();
            assert!(
                err.contains(expected),
                "expected {} to contain {}",
                err,
                expected
            );
        }
    }
}

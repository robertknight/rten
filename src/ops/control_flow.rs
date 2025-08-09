use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use smallvec::SmallVec;

use crate::graph::{CaptureEnv, Graph, NodeId, RunError, RunOptions};
use crate::ops::{map_value, OpError, OpRunContext, Operator, OutputList, Value};
use crate::timing::Profiler;
use crate::value::ValueOrView;
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

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::InvalidValue(
            "operator must be run with `run_subgraph`",
        ))
    }

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
                OpError::InvalidValue("cond must be a single value"),
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

    fn run(&self, _ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Err(OpError::InvalidValue(
            "operator must be run with `run_subgraph`",
        ))
    }

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
            return Err(make_run_error(OpError::InvalidValue(
                "body input count should be 2 + number of loop-carried dependencies",
            )));
        }

        let output_ids = self.body.output_ids();
        if output_ids.len() < 1 + loop_carried_deps.len() {
            return Err(make_run_error(OpError::InvalidValue(
                "body output count should be at least 1 + number of loop-carried dependencies",
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
                make_run_error(OpError::InvalidValue("condition output has incorrect type"))
            })?;
            let Some(&next_cond) = next_cond.item() else {
                return Err(make_run_error(OpError::InvalidValue(
                    "cond output should be a scalar",
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

        // Construct outputs sequence consisting of final loop carried
        // dependencies followed by concatenated scan outputs.
        let mut outputs: Vec<Value> = Vec::with_capacity(loop_carried_deps.len() + scan_output_len);
        outputs.extend(loop_carried_deps.into_iter().map(|dep| dep.into_owned()));

        for mut output_seq in scan_outputs.into_iter() {
            // TODO - Decide how to handle zero iterations, as we can't know the
            // type or shape of scan outputs in that case.
            if output_seq.is_empty() {
                continue;
            }
            let first = output_seq.remove(0);

            map_value!(first, first, {
                let concatenated_output =
                    concat_scan_outputs(first, output_seq).map_err(make_run_error)?;
                outputs.push(concatenated_output.into());
            })
        }

        Ok(outputs.into())
    }
}

/// Concatenate the scan outputs from across loop iterations.
///
/// Outputs from each iteration must have the same type and shape.
fn concat_scan_outputs<T: Copy>(
    mut first: Tensor<T>,
    rest: Vec<Value>,
) -> Result<Tensor<T>, OpError>
where
    Tensor<T>: TryFrom<Value>,
{
    let mut shape = Vec::new();
    shape.push(1 + rest.len());
    shape.extend(first.shape().iter().copied());
    let mut tensor = Tensor::with_capacity(&shape, 0);

    // First `append` should always succeed because we used the
    // appended tensor's shape to determine the output shape.
    first.insert_axis(0);
    tensor.append(0, &first).unwrap();

    for output in rest {
        let mut typed_output: Tensor<T> = output.try_into().map_err(|_| {
            OpError::InvalidValue("scan outputs have different types across iterations")
        })?;
        typed_output.insert_axis(0);
        tensor
            .append(0, &typed_output)
            .map_err(|_| OpError::InvalidValue("shape mismatch for loop scan output"))?;
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;

    use crate::graph::builder::Expr;
    use crate::graph::CaptureEnv;
    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, OpRunContext, Operator, RunError};
    use crate::value::{Value, ValueView};

    use super::Loop;

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

            let mut input_list = InputList::new();
            input_list.push_optional(max_iter_tensor.as_ref().map(ValueView::from));
            input_list.push_optional(cond_tensor.as_ref().map(ValueView::from));
            for input in inputs {
                input_list.push(input.clone());
            }

            let pool = new_pool();
            let ctx = OpRunContext::new(&pool, &input_list);
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
    fn test_loop() {
        let body = {
            let x = Expr::value("i");
            let cond = Expr::value("cond");
            let x_2 = x.clone() * x.clone();
            Expr::make_graph(&["i", "cond"], &[cond, x_2])
        };
        let runner = LoopRunner::new(Loop { body });
        let mut outputs = runner.run(Some(5), None, &[]).unwrap();
        assert_eq!(outputs.len(), 1);
        let squares: Tensor<i32> = outputs.remove(0).try_into().unwrap();
        assert_eq!(squares, Tensor::from([0, 1, 4, 9, 16]));
    }
}

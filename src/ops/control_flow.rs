use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use smallvec::SmallVec;

use crate::graph::{CaptureEnv, Graph, NodeId, RunError, RunOptions};
use crate::ops::{OpError, OpRunContext, Operator, OutputList, SubgraphOperator, Value, map_value};
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

    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        Some(self as &dyn SubgraphOperator)
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

    fn as_subgraph_op(&self) -> Option<&dyn SubgraphOperator> {
        Some(self as &dyn SubgraphOperator)
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
            return Err(make_run_error(OpError::InvalidValue(
                "loop body has too few inputs",
            )));
        }

        let output_ids = self.body.output_ids();
        if output_ids.len() < 1 + loop_carried_deps.len() {
            return Err(make_run_error(OpError::InvalidValue(
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
                make_run_error(OpError::InvalidValue("condition output has incorrect type"))
            })?;
            let Some(&next_cond) = next_cond.item() else {
                return Err(make_run_error(OpError::InvalidValue(
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

            // Concatenate outputs. This can fail if the outputs have different
            // shapes or the value is not a tensor.
            map_value!(first, first, {
                concat_scan_outputs(first, output_seq).map(|out| outputs.push(out.into()))
            })
            .map_err(make_run_error)?;
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
            OpError::InvalidValue("scan output has different type across iterations")
        })?;
        typed_output.insert_axis(0);
        tensor.append(0, &typed_output).map_err(|_| {
            OpError::InvalidValue("scan output has different shape across iterations")
        })?;
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;

    use crate::graph::builder::Expr;
    use crate::graph::{CaptureEnv, Graph};
    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, OpError, OpRunContext, RunError, SubgraphOperator};
    use crate::value::{Scalar, Value, ValueView};

    use super::Loop;

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
                        value: Scalar::Int(1),
                    });

                    Expr::make_graph([iter, cond.clone()], [cond, output])
                },
                expected: "scan output has different shape across iterations",
            },
            // TODO: Scan outputs with different types across iterations
        ];

        for Case { body, expected } in cases {
            let runner = LoopRunner::new(Loop { body });
            let error = runner.run(Some(3), None, &[]).err().unwrap();

            match error {
                RunError::OperatorError { error, .. } => {
                    assert_eq!(error, OpError::InvalidValue(expected));
                }
                _ => panic!("expected OperatorError"),
            }
        }
    }
}

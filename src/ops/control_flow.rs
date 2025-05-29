use rten_tensor::TensorView;
use smallvec::SmallVec;

use crate::graph::{CaptureEnv, Graph, RunError, RunOptions};
use crate::ops::{OpError, OpRunContext, Operator, Output, OutputList};
use crate::timing::Profiler;
use crate::weight_cache::WeightCache;

fn output_list_from_vec(xs: Vec<Output>) -> OutputList {
    xs.into_iter().collect()
}

fn run_error_from_op_error(error: OpError) -> RunError {
    RunError::OperatorError {
        name: "If".to_string(),
        error,
    }
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
        let cond: TensorView<i32> = ctx
            .inputs()
            .require_as(0)
            .map_err(run_error_from_op_error)?;
        let Some(cond_bool) = cond.item().copied() else {
            return Err(run_error_from_op_error(OpError::InvalidValue(
                "cond must be a single value",
            )));
        };

        if cond_bool != 0 {
            self.then_branch
                .run_subgraph(
                    Vec::new(),
                    self.then_branch.output_ids(),
                    captures,
                    Some(ctx.pool()),
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
                    Some(ctx.pool()),
                    weight_caches.map(|wcs| &wcs[1]),
                    profiler,
                    run_opts,
                )
                .map(output_list_from_vec)
        }
    }
}

use rten_base::num::IsNaN;
use rten_shape_inference::UnaryOp;
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::reduce::{cmp_nan_greater, cmp_nan_less};
use crate::ops::{map_value_view, resolve_axis, try_resolve_index};
use crate::value::ValueView;

// Specifies how to combine an existing element value with an update in a
// scatter operation.
#[derive(Copy, Clone, Debug)]
pub enum ScatterReduction {
    /// Add the existing value and update.
    Add,

    /// Multiply the existing value with the update.
    Mul,

    /// Take the minimum of the existing value and the update, propagating NaNs.
    Min,

    /// Take the maximum of the existing value and the update, propagating NaNs.
    Max,
}

fn scatter_reduce<
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    current: T,
    update: T,
    reduction: Option<ScatterReduction>,
) -> T {
    match reduction {
        Some(ScatterReduction::Add) => current + update,
        Some(ScatterReduction::Mul) => current * update,

        // nb. In the operations below, we prefer to keep the current value
        // unless the update is definitely less or NaN.
        Some(ScatterReduction::Min) => match cmp_nan_less(update, current) {
            std::cmp::Ordering::Less => update,
            _ => current,
        },
        Some(ScatterReduction::Max) => match cmp_nan_greater(update, current) {
            std::cmp::Ordering::Greater => update,
            _ => current,
        },
        None => update,
    }
}

pub fn scatter_elements<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    pool: &BufferPool,
    data: TensorView<T>,
    indices: TensorView<i32>,
    updates: TensorView<T>,
    axis: isize,
    reduction: Option<ScatterReduction>,
) -> Result<Tensor<T>, OpError> {
    if indices.ndim() != data.ndim() {
        return Err(OpError::invalid_value(
            "`data` and `indices` must have same rank",
        ));
    }
    if indices.shape() != updates.shape() {
        return Err(OpError::invalid_value(
            "`indices` and `updates` must have same shape",
        ));
    }
    let axis = resolve_axis(data.ndim(), axis)?;

    let axis_size = data.size(axis);
    let mut output = data.to_tensor_in(pool);

    for (output_lane, (update_lane, index_lane)) in output
        .lanes_mut(axis)
        .zip(updates.lanes(axis).zip(indices.lanes(axis)))
    {
        let mut output_lane = output_lane.into_view();

        for (idx, update) in index_lane.zip(update_lane) {
            let idx = try_resolve_index(axis_size, *idx)?;
            let out_el = &mut output_lane[[idx]];
            *out_el = scatter_reduce(*out_el, *update, reduction);
        }
    }

    Ok(output)
}

/// Deprecated alias for [`ScatterElements`].
///
/// See https://github.com/onnx/onnx/pull/2143.
#[derive(Debug)]
pub struct Scatter {
    pub axis: isize,
}

impl Operator for Scatter {
    fn name(&self) -> &str {
        "Scatter"
    }

    fn max_inputs(&self) -> Option<usize> {
        ScatterElements {
            axis: self.axis,
            reduction: None,
        }
        .max_inputs()
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        ScatterElements {
            axis: self.axis,
            reduction: None,
        }
        .run(ctx)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

#[derive(Debug)]
pub struct ScatterElements {
    pub axis: isize,
    pub reduction: Option<ScatterReduction>,
}

impl Operator for ScatterElements {
    fn name(&self) -> &str {
        "ScatterElements"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let data = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(data, x, {
            let updates = inputs.require_as(2)?;
            scatter_elements(ctx.pool(), x, indices, updates, self.axis, self.reduction)
                .into_op_result()
        })
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

pub fn scatter_nd<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    pool: &BufferPool,
    data: TensorView<T>,
    indices: TensorView<i32>,
    updates: TensorView<T>,
    reduction: Option<ScatterReduction>,
) -> Result<Tensor<T>, OpError> {
    if data.ndim() == 0 || indices.ndim() == 0 {
        return Err(OpError::invalid_value(
            "`data` and `indices` must have rank >= 1",
        ));
    }

    // Per spec, the `indices` tensor is treated as a set of K-tuples where
    // `k <= data.ndim()`, specifying the indices of slices to update.
    let k = indices.size(indices.ndim() - 1);

    let expected_update_dim = data.ndim() + indices.ndim() - k - 1;
    if updates.ndim() != expected_update_dim {
        return Err(OpError::invalid_value(
            "`updates` does not have expected rank",
        ));
    }

    let mut expected_update_shape: SmallVec<[usize; 5]> = SmallVec::new();
    expected_update_shape.extend_from_slice(&indices.shape()[..indices.ndim() - 1]);
    expected_update_shape.extend_from_slice(&data.shape()[k..data.ndim()]);
    if updates.shape() != expected_update_shape.as_slice() {
        return Err(OpError::invalid_value(
            "`updates` does not have expected shape",
        ));
    }

    // Assuming the updates and indices are likely already contiguous, we can
    // optimize iterating over slices of the innermost dimensions using slice
    // chunks.
    let updates = updates.to_contiguous_in(pool).auto_return(pool);
    let update_slice_len: usize = updates.shape()[indices.ndim() - 1..].iter().product();
    let update_slices = updates.data().chunks(update_slice_len);

    let indices = indices.to_contiguous_in(pool).auto_return(pool);
    let index_slices = indices.data().chunks(indices.size(indices.ndim() - 1));

    let mut output = data.to_tensor_in(pool);
    for (index, update_slice) in index_slices.zip(update_slices) {
        let mut output_slice_offset = 0;
        for (i, (size, stride)) in index
            .iter()
            .zip(output.shape().iter().zip(output.strides().iter()))
        {
            let idx = try_resolve_index(*size, *i)?;
            output_slice_offset += idx * stride;
        }
        let out_data = output.data_mut().unwrap();
        let out_slice = &mut out_data[output_slice_offset..][..update_slice_len];

        for (out_el, update) in out_slice.iter_mut().zip(update_slice.iter()) {
            *out_el = scatter_reduce(*out_el, *update, reduction);
        }
    }
    Ok(output)
}

#[derive(Debug)]
pub struct ScatterND {
    pub reduction: Option<ScatterReduction>,
}

impl Operator for ScatterND {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let data = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(data, x, {
            let updates = inputs.require_as(2)?;
            scatter_nd(ctx.pool(), x, indices, updates, self.reduction).into_op_result()
        })
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use crate::buffer_pool::BufferPool;
    use crate::operator::OpError;
    use crate::ops::{ScatterReduction, scatter_elements, scatter_nd};

    #[test]
    fn test_scatter_elements() {
        #[derive(Debug)]
        struct Case {
            data: Tensor,
            indices: Tensor<i32>,
            updates: Tensor,
            axis: isize,
            expected: Result<Tensor, OpError>,
        }

        let cases = [
            // Example #1 from ONNX spec
            Case {
                data: Tensor::zeros(&[3, 3]),
                indices: Tensor::from([[1, 0, 2], [0, 2, 1]]),
                updates: Tensor::from([[1., 1.1, 1.2], [2., 2.1, 2.2]]),
                axis: 0,
                expected: Ok(Tensor::from([[2., 1.1, 0.], [1., 0., 2.2], [0., 2.1, 1.2]])),
            },
            // Example #2 from ONNX spec
            Case {
                data: Tensor::from([[1., 2., 3., 4., 5.]]),
                indices: Tensor::from([[1, 3]]),
                updates: Tensor::from([[1.1, 2.1]]),
                axis: 1,
                expected: Ok(Tensor::from([[1., 1.1, 3., 2.1, 5.]])),
            },
            // Invalid index
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([4]),
                updates: Tensor::from([1.]),
                axis: 0,
                expected: Err(OpError::invalid_value(
                    "Index 4 is out of range. Must be in [-3, 3)",
                )),
            },
            // Rank mismatch
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([[4]]),
                updates: Tensor::from([[1.]]),
                axis: 0,
                expected: Err(OpError::invalid_value(
                    "`data` and `indices` must have same rank",
                )),
            },
            // `indices` and `updates` shape mismatch
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([4]),
                updates: Tensor::from([1., 2.]),
                axis: 0,
                expected: Err(OpError::invalid_value(
                    "`indices` and `updates` must have same shape",
                )),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_elements(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                case.axis,
                None,
            );
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_scatter_elements_reduction() {
        let pool = BufferPool::new();

        let data = Tensor::from([1, 2, 3, 4]);
        let indices = Tensor::from([1, 3]);
        let updates = Tensor::from([2, 2]);

        let scatter = |reduction: Option<ScatterReduction>| {
            scatter_elements(
                &pool,
                data.view(),
                indices.view(),
                updates.view(),
                0, /* axis */
                reduction,
            )
            .unwrap()
        };

        let result = scatter(Some(ScatterReduction::Add));
        assert_eq!(result, Tensor::from([1, 4, 3, 6]));

        let result = scatter(Some(ScatterReduction::Mul));
        assert_eq!(result, Tensor::from([1, 4, 3, 8]));

        let result = scatter(Some(ScatterReduction::Min));
        assert_eq!(result, Tensor::from([1, 2, 3, 2]));

        let result = scatter(Some(ScatterReduction::Max));
        assert_eq!(result, Tensor::from([1, 2, 3, 4]));
    }

    #[test]
    fn test_scatter_nd() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<i32>,
            indices: Tensor<i32>,
            updates: Tensor<i32>,
            expected: Tensor<i32>,
        }

        let cases = [
            // Example 1 from ONNX spec.
            Case {
                data: [1, 2, 3, 4, 5, 6, 7, 8].into(),
                indices: Tensor::from_data(&[4, 1], vec![4, 3, 1, 7]),
                updates: [9, 10, 11, 12].into(),
                expected: [1, 11, 3, 10, 9, 6, 7, 12].into(),
            },
            // Example 2 from ONNX spec.
            Case {
                data: [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
                .into(),
                indices: [[0], [2]].into(),
                updates: [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                ]
                .into(),
                expected: [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
                .into(),
            },
            // Test for issue when `updates` has a lower rank than `indices`.
            Case {
                data: [[1, 2], [3, 4]].into(),
                indices: [[0, 0], [0, 1]].into(),
                updates: [5, 6].into(),
                expected: [[5, 6], [3, 4]].into(),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                None,
            )
            .unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_scatter_nd_reduce() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<f32>,
            indices: Tensor<i32>,
            updates: Tensor<f32>,
            expected: Tensor<f32>,
            reduction: ScatterReduction,
        }

        let cases = [
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., 2., 3., 4.].into(),
                expected: [2., 4., 6., 8.].into(),
                reduction: ScatterReduction::Add,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., 2., 3., 4.].into(),
                expected: [1., 4., 9., 16.].into(),
                reduction: ScatterReduction::Mul,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., -2., 3., -4.].into(),
                expected: [1., -2., 3., -4.].into(),
                reduction: ScatterReduction::Min,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., -2., 3., -4.].into(),
                expected: [1., 2., 3., 4.].into(),
                reduction: ScatterReduction::Max,
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                Some(case.reduction),
            )
            .unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_scatter_nd_invalid() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<f32>,
            indices: Tensor<i32>,
            updates: Tensor<f32>,
            expected: OpError,
        }

        let cases = [
            Case {
                data: (5.).into(),
                indices: [0].into(),
                updates: [0.].into(),
                expected: OpError::invalid_value("`data` and `indices` must have rank >= 1"),
            },
            Case {
                data: Tensor::from([0.]),
                indices: Tensor::from(0),
                updates: [0.].into(),
                expected: OpError::invalid_value("`data` and `indices` must have rank >= 1"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [3]].into(),
                updates: [[1., 2., 3., 4.]].into(),
                expected: OpError::invalid_value("`updates` does not have expected rank"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [3]].into(),
                updates: [1., 2., 3., 4., 5.].into(),
                expected: OpError::invalid_value("`updates` does not have expected shape"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [4]].into(),
                updates: [1., 2., 3., 4.].into(),
                expected: OpError::invalid_value("Index 4 is out of range. Must be in [-4, 4)"),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                None,
            );
            assert_eq!(result.as_ref(), Err(&case.expected));
        })
    }
}

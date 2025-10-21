//! Operators which query or change the shape of a tensor, or copy/move/reorder
//! elements.

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView, is_valid_permutation};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{IntoOpResult, OpError, OpRunContext, Operator, OutputList, static_dims};
use crate::ops::binary_elementwise::{broadcast_shapes, fast_broadcast_cycles_repeats};
use crate::ops::{map_value, map_value_view, resolve_axes, resolve_axis};
use crate::value::{Value, ValueView};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DepthToSpaceMode {
    DepthColumnRow,
    ColumnRowDepth,
}

pub fn depth_to_space<T: Clone>(
    pool: &BufferPool,
    input: TensorView<T>,
    block_size: u32,
    mode: DepthToSpaceMode,
) -> Result<Tensor<T>, OpError> {
    if block_size == 0 {
        return Err(OpError::InvalidValue("`block_size` must be > 0"));
    }

    let input = static_dims!(input, 4, "NCHW")?;
    let [n, c, h, w] = input.shape();
    let block_size = block_size as usize;

    if c % (block_size * block_size) != 0 {
        return Err(OpError::InvalidValue(
            "input channels must be a multiple of `block_size` squared",
        ));
    }

    let new_c = c / (block_size * block_size);
    let new_shape = [n, new_c, h * block_size, w * block_size];

    // Reshape following steps in `DepthToSpace` ONNX spec.
    // See https://onnx.ai/onnx/operators/onnx__DepthToSpace.html#summary
    let tmp = input.to_contiguous_in(pool);
    let tmp = match mode {
        DepthToSpaceMode::DepthColumnRow => tmp.reshaped([n, block_size, block_size, new_c, h, w]),
        DepthToSpaceMode::ColumnRowDepth => tmp.reshaped([n, new_c, block_size, block_size, h, w]),
    };
    let tmp = match mode {
        DepthToSpaceMode::DepthColumnRow => tmp.permuted([0, 3, 4, 1, 5, 2]),
        DepthToSpaceMode::ColumnRowDepth => tmp.permuted([0, 1, 4, 2, 5, 3]),
    };
    let mut tmp = tmp.to_tensor_in(pool).into_dyn();
    tmp.reshape(&new_shape);

    Ok(tmp)
}

#[derive(Debug)]
pub struct DepthToSpace {
    pub block_size: u32,
    pub mode: DepthToSpaceMode,
}

impl Operator for DepthToSpace {
    fn name(&self) -> &str {
        "DepthToSpace"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        depth_to_space::<f32>(ctx.pool(), input, self.block_size, self.mode).into_op_result()
    }
}

/// Return the tensor shape resulting from broadcasting `input_shape` with `shape`.
fn expand_output_shape(
    input_shape: &[usize],
    shape: &NdTensorView<i32, 1>,
) -> Result<SmallVec<[usize; 4]>, OpError> {
    let shape_vec: SmallVec<[usize; 4]> = shape.iter().map(|el| *el as usize).collect();
    broadcast_shapes(input_shape, &shape_vec).ok_or(OpError::IncompatibleInputShapes(
        "Cannot broadcast input with target shape",
    ))
}

/// Broadcast `input` to `out_shape`. This assumes that `out_shape` has already
/// been verified to be a valid broadcast target.
pub(crate) fn expand_to<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    out_shape: &[usize],
) -> Tensor<T> {
    let out_len = out_shape.iter().product();

    match (
        input.data(),
        fast_broadcast_cycles_repeats(input.shape(), out_shape),
    ) {
        // Fast path for common case of contiguous input and broadcast that can
        // be performed using cycle + repeat.
        (Some(in_data), Some((cycles, repeats))) => {
            assert!(out_len == input.len() * cycles * repeats);

            let mut out_data: Vec<T> = pool.alloc(out_len);
            let mut out_ptr = out_data.as_mut_ptr();
            for _ in 0..cycles {
                if repeats == 1 {
                    // Super-fast path for cycling only.
                    unsafe {
                        std::ptr::copy_nonoverlapping(in_data.as_ptr(), out_ptr, in_data.len());
                        out_ptr = out_ptr.add(in_data.len());
                    }
                } else {
                    for el in in_data.iter() {
                        for _ in 0..repeats {
                            unsafe {
                                *out_ptr = *el;
                                out_ptr = out_ptr.add(1);
                            }
                        }
                    }
                }
            }
            // Safety: We have initialized all output elements.
            unsafe { out_data.set_len(out_len) };

            Tensor::from_data(out_shape, out_data)
        }
        _ => input.broadcast(out_shape).to_tensor_in(pool),
    }
}

pub fn expand<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    shape: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    let out_shape = expand_output_shape(input.shape(), shape)?;
    Ok(expand_to(pool, input, &out_shape))
}

#[derive(Debug)]
pub struct Expand {}

impl Operator for Expand {
    fn name(&self) -> &str {
        "Expand"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;

        map_value_view!(input, x, { expand(ctx.pool(), x, &shape).into_op_result() })
    }

    fn can_run_in_place(&self) -> bool {
        // Expand can run in place if it is a noop, ie. if the broadcasted
        // shape is the same as the input shape.
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let shape = ctx.inputs().require_as(0)?;

        let out_shape = expand_output_shape(&input.shape(), &shape)?;
        if input.shape() == out_shape {
            return Ok(input);
        }

        map_value!(input, input, {
            let input = input.auto_return(ctx.pool());
            let output = expand_to(ctx.pool(), input.view(), &out_shape);
            Ok(output.into())
        })
    }
}

fn flattened_shape(shape: &[usize], axis: isize) -> Result<[usize; 2], OpError> {
    let outer_dims = if axis == shape.len() as isize {
        shape.len()
    } else {
        resolve_axis(shape.len(), axis)?
    };
    let outer_size = shape.iter().take(outer_dims).product();
    let inner_size = shape.iter().skip(outer_dims).product();
    Ok([outer_size, inner_size])
}

pub fn flatten<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    let mut output = input.to_tensor_in(pool);
    output.reshape(&shape);
    Ok(output)
}

pub fn flatten_in_place<T: Copy>(
    pool: &BufferPool,
    input: &mut Tensor<T>,
    axis: isize,
) -> Result<(), OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    input.reshape_in(pool, &shape);
    Ok(())
}

#[derive(Debug)]
pub struct Flatten {
    pub axis: isize,
}

impl Operator for Flatten {
    fn name(&self) -> &str {
        "Flatten"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        map_value_view!(input, x, {
            flatten(ctx.pool(), x, self.axis).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        map_value!(input, x, {
            flatten_in_place(ctx.pool(), &mut x, self.axis)?;
            Ok(x.into())
        })
    }
}

/// Compute the target shape for a reshape operation, given the shape of the
/// input tensor and a target `shape` which may contain a "-1" entry to indicate
/// a dimension whose size should be inferred.
///
/// Handling of zeros in the target shape depends on `allow_zero`. If false,
/// the corresponding input dimension is copied, otherwise the zero is
/// preserved in the output shape.
fn resolve_shape(
    input_shape: &[usize],
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<SmallVec<[usize; 4]>, OpError> {
    // If exactly one of the new shape's dimensions is -1, infer the size
    // from the input length and the sizes of the other dimensions.
    let mut unspecified_dim = None;
    let mut specified_dims_size = 1;
    for (dim, &size) in shape.iter().enumerate() {
        if size < -1 {
            return Err(OpError::InvalidValue("Invalid dimension size in shape"));
        } else if size == 0 && !allow_zero {
            if dim >= input_shape.len() {
                return Err(OpError::InvalidValue(
                    "Zero dim has no corresponding input dim",
                ));
            }
            specified_dims_size *= input_shape[dim];
        } else if size != -1 {
            specified_dims_size *= size as usize;
        } else if unspecified_dim.is_some() {
            return Err(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1",
            ));
        } else {
            unspecified_dim = Some(dim);
        }
    }

    let input_len = input_shape.iter().product();
    let (unspecified_dim_size, remainder) = match input_len {
        0 => (0, 0),
        _ => {
            if specified_dims_size == 0 {
                // If `specified_dims_size` is zero but `input_len` is non-zero,
                // this means that the target shape doesn't match the input length.
                //
                // Return a non-zero remainder here to cause the appropriate
                // error to be returned.
                (0, 1)
            } else {
                (
                    input_len / specified_dims_size,
                    input_len % specified_dims_size,
                )
            }
        }
    };

    if remainder != 0 {
        return Err(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));
    }

    Ok(shape
        .iter()
        .enumerate()
        .map(|(dim, &size)| match size {
            -1 => unspecified_dim_size,
            0 if !allow_zero => input_shape[dim],
            valid => valid as usize,
        })
        .collect())
}

pub fn reshape<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<Tensor<T>, OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    let output = input.to_tensor_in(pool).into_shape(out_shape.as_slice());
    Ok(output)
}

pub fn reshape_in_place<T: Copy>(
    pool: &BufferPool,
    input: &mut Tensor<T>,
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<(), OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    input.reshape_in(pool, &out_shape);
    Ok(())
}

#[derive(Debug)]
pub struct Reshape {
    pub allow_zero: bool,
}

impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;

        map_value_view!(input, x, {
            reshape(ctx.pool(), x, &shape, self.allow_zero).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let shape = ctx.inputs().require_as(0)?;

        map_value!(input, output, {
            reshape_in_place(ctx.pool(), &mut output, &shape, self.allow_zero)?;
            Ok(output.into())
        })
    }
}

#[derive(Debug, Default)]
pub struct Shape {
    pub start: Option<i32>,
    pub end: Option<i32>,
}

impl Operator for Shape {
    fn name(&self) -> &str {
        "Shape"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let ndim = input.ndim() as i32;

        // Convert `start` and `end` to positive values in `[0, ndim]`, clamping
        // if out of range.
        //
        // The spec says to clamp to `[0, r-1]` but this is incorrect as the end
        // bound is exclusive and so needs to be `r` to include the entire range.
        // See https://github.com/onnx/onnx/issues/6862.
        let start = self
            .start
            .map(|start| {
                let start = if start < 0 { start + ndim } else { start };
                start.clamp(0, ndim) as usize
            })
            .unwrap_or(0);

        let end = self
            .end
            .map(|end| {
                let end = if end < 0 { end + ndim } else { end };
                end.clamp(0, ndim) as usize
            })
            .unwrap_or(input.ndim())
            // Spec doesn't say how to handle the case where `start > end`,
            // we clamp `end` to prevent this.
            .max(start);

        let shape_slice = &input.shape()[start..end];

        // Allocate output from pool for consistency with other operators,
        // even though the buffer is tiny, so there is no performance benefit.
        let mut data = ctx.pool().alloc(input.ndim());
        data.extend(shape_slice.iter().map(|&el| el as i32));

        Tensor::from_data(&[shape_slice.len()], data).into_op_result()
    }
}

#[derive(Debug)]
pub struct Size {}

impl Operator for Size {
    fn name(&self) -> &str {
        "Size"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let len = input.len() as i32;

        // Allocate output from pool for consistency with other operators,
        // even though the buffer is tiny, so there is no performance benefit.
        let mut output = Tensor::zeros_in(ctx.pool(), &[]);
        output[[]] = len;

        output.into_op_result()
    }
}

pub fn squeeze_in_place<T: Clone>(
    input: &mut Tensor<T>,
    axes: Option<NdTensorView<i32, 1>>,
) -> Result<(), OpError> {
    let axes = axes
        .map(|axes| resolve_axes(input.ndim(), axes.iter()))
        .transpose()?;
    let sorted_axes = if let Some(mut axes) = axes {
        for &axis in axes.iter() {
            if axis >= input.ndim() {
                return Err(OpError::InvalidValue("Axis is invalid"));
            }
            if input.size(axis) != 1 {
                return Err(OpError::InvalidValue(
                    "Can only remove dimensions of size 1",
                ));
            }
        }
        axes.sort();
        axes
    } else {
        input
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, size)| if *size == 1 { Some(i) } else { None })
            .collect()
    };

    for (n_removed, axis) in sorted_axes.into_iter().enumerate() {
        input.remove_axis(axis - n_removed);
    }

    Ok(())
}

pub fn squeeze<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    axes: Option<NdTensorView<i32, 1>>,
) -> Result<Tensor<T>, OpError> {
    let mut output = input.to_tensor_in(pool);
    squeeze_in_place(&mut output, axes)?;
    Ok(output)
}

#[derive(Debug)]
pub struct Squeeze {}

impl Operator for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let axes = inputs.get_as(1)?;

        map_value_view!(input, x, { squeeze(ctx.pool(), x, axes).into_op_result() })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let axes = ctx.inputs().get_as(0)?;

        map_value!(input, output, {
            squeeze_in_place(&mut output, axes)?;
            Ok(output.into())
        })
    }
}

pub fn transpose<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    permutation: Option<&[usize]>,
) -> Result<Tensor<T>, OpError> {
    let mut transposed = input.view();
    match permutation {
        Some(order) => {
            if !is_valid_permutation(input.ndim(), order) {
                return Err(OpError::InvalidValue("Permutation is invalid"));
            }
            transposed.permute(order)
        }
        None => {
            transposed.transpose();
        }
    };
    let output = Tensor::uninit_in(pool, transposed.shape());
    Ok(output.init_from(&transposed))
}

#[derive(Debug)]
pub struct Transpose {
    /// The order of the transposed dimensions. If ommitted, the dimensions
    /// are reversed.
    pub perm: Option<Vec<usize>>,
}

impl Operator for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let perm_slice = self.perm.as_deref();

        map_value_view!(input, x, {
            transpose(ctx.pool(), x, perm_slice).into_op_result()
        })
    }
}

pub fn unsqueeze_in_place<T: Clone>(
    mut input: Tensor<T>,
    axes: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    let sorted_axes = if axes.len() == 1 {
        let axis = resolve_axis(input.ndim() + 1, axes[0] as isize)?;
        SmallVec::from_slice(&[axis])
    } else {
        let mut sorted_axes = resolve_axes(input.ndim() + axes.len(), axes.iter())?;
        sorted_axes.sort_unstable();

        let axes_unique = sorted_axes
            .iter()
            .skip(1)
            .zip(sorted_axes.iter())
            .all(|(prev, current)| prev != current);
        if !axes_unique {
            return Err(OpError::InvalidValue("Axes must be unique"));
        }
        sorted_axes
    };

    for axis in sorted_axes {
        input.insert_axis(axis);
    }

    Ok(input)
}

pub fn unsqueeze<T: Copy>(
    pool: &BufferPool,
    input: TensorView<T>,
    axes: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    unsqueeze_in_place(input.to_tensor_in(pool), axes)
}

#[derive(Debug)]
pub struct Unsqueeze {}

impl Operator for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let axes = inputs.require_as(1)?;

        map_value_view!(input, x, {
            unsqueeze(ctx.pool(), x, &axes).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let axes = ctx.inputs().require_as(0)?;

        map_value!(input, output, {
            Ok(unsqueeze_in_place(output, &axes)?.into())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use super::{DepthToSpaceMode, depth_to_space};
    use crate::buffer_pool::BufferPool;
    use crate::operator::{OpError, OperatorExt};
    use crate::ops::layout::{
        Reshape, Shape, Size, expand, flatten, reshape, reshape_in_place, squeeze,
        squeeze_in_place, transpose, unsqueeze,
    };
    use crate::value::Value;

    #[test]
    fn test_depth_to_space() {
        #[derive(Debug)]
        struct Case {
            input: NdTensor<f32, 4>,
            block_size: u32,
            mode: DepthToSpaceMode,
            expected: Result<Tensor, OpError>,
        }

        let input = NdTensor::from([
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[4.0]],
            [[5.0]],
            [[6.0]],
            [[7.0]],
            [[8.0]],
        ])
        .into_shape([1, 8, 1, 1]);

        let cases = [
            // DepthColumnRow (DCR) mode
            Case {
                input: input.clone(),
                block_size: 2,
                mode: DepthToSpaceMode::DepthColumnRow,
                expected: Ok(
                    NdTensor::from([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])
                        .into_shape([1, 2, 2, 2].as_slice()),
                ),
            },
            // ColumnRowDepth (CRD) mode
            Case {
                input: input.clone(),
                block_size: 2,
                mode: DepthToSpaceMode::ColumnRowDepth,
                expected: Ok(
                    NdTensor::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
                        .into_shape([1, 2, 2, 2].as_slice()),
                ),
            },
            // C % block_size^2 != 0
            Case {
                input: NdTensor::full([1, 16, 2, 2], 1.0),
                block_size: 3,
                mode: DepthToSpaceMode::ColumnRowDepth,
                expected: Err(OpError::InvalidValue(
                    "input channels must be a multiple of `block_size` squared",
                )),
            },
            // block_size == 0
            Case {
                input: NdTensor::full([1, 16, 2, 2], 1.0),
                block_size: 0,
                mode: DepthToSpaceMode::ColumnRowDepth,
                expected: Err(OpError::InvalidValue("`block_size` must be > 0")),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = depth_to_space(&pool, case.input.as_dyn(), case.block_size, case.mode);
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_expand() {
        let pool = BufferPool::new();

        // Broadcast scalar
        let input = Tensor::from(5.);
        let shape = NdTensor::from([2, 2]);
        let expected = Tensor::from_data(&[2, 2], vec![5., 5., 5., 5.]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(&result, &expected);

        // Broadcast that changes dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = NdTensor::from([2, 3, 1]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 3, 1]);

        // Broadcast that uses dimensions from both the input shape and target
        // shape in the output shape.
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = NdTensor::from([2, 1, 6]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 3, 6]);

        // Broadcast that does not change dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = NdTensor::from([3, 4]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[3, 4]);

        // Broadcast of leading and trailing dims
        let input = Tensor::from([1, 2]).into_shape([1, 2, 1].as_slice());
        let shape = NdTensor::from([2, 2, 2]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[1, 1, 2, 2, 1, 1, 2, 2]);

        // Broadcast of inner dim
        let input = Tensor::from([1, 2, 3, 4]).into_shape([2, 1, 2].as_slice());
        let shape = NdTensor::from([2, 2, 2]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[1, 2, 1, 2, 3, 4, 3, 4]);
    }

    #[test]
    fn test_expand_invalid_inputs() {
        let pool = BufferPool::new();

        // Invalid broadcast shape
        let input = Tensor::from([1, 2, 3]);
        let shape = NdTensor::from([2, 2]);
        let result = expand(&pool, input.view(), &shape.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast input with target shape"
            ))
        );
    }

    #[test]
    fn test_flatten() {
        #[derive(Debug)]
        struct Case {
            shape: Vec<usize>,
            axis: isize,
            expected: Result<Vec<usize>, OpError>,
        }

        let cases = [
            Case {
                shape: [1, 5, 1, 1].into(),
                axis: 1,
                expected: Ok([1, 5].into()),
            },
            Case {
                shape: [2, 3, 1, 4].into(),
                axis: 2,
                expected: Ok([6, 4].into()),
            },
            // Axis = 0
            Case {
                shape: [2, 3, 1, 4].into(),
                axis: 0,
                expected: Ok([1, 24].into()),
            },
            // Axis equal to rank of input
            Case {
                shape: [2, 2].into(),
                axis: 2,
                expected: Ok([4, 1].into()),
            },
            // Negative values count from the back
            Case {
                shape: [2, 3, 4].into(),
                axis: -1,
                expected: Ok([6, 4].into()),
            },
            Case {
                shape: [2, 3, 4].into(),
                axis: -2,
                expected: Ok([2, 12].into()),
            },
            Case {
                shape: [2, 3, 4].into(),
                axis: -3,
                expected: Ok([1, 24].into()),
            },
            // Values outside `[-r, r]` are invalid
            Case {
                shape: [2, 3, 4].into(),
                axis: 4,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
            },
            Case {
                shape: [2, 3, 4].into(),
                axis: -4,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let input = Tensor::<f32>::zeros(case.shape.as_slice());
            let result =
                flatten(&pool, input.view(), case.axis).map(|tensor| tensor.shape().to_vec());
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_reshape_with_unspecified_dim() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        // Reshape with an unspecified (-1) dim and nonzero-length input
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = NdTensor::from([1, -1, 2]);
        let expected = input.to_shape([1, 2, 2].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Reshape with an unspecified (-1) dim and zero-length input
        let zero_sized_input = Tensor::<f32>::from_data(&[4, 0, 1], vec![]);
        let shape = NdTensor::from([100, -1]);
        let result = reshape(
            &pool,
            zero_sized_input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        let expected = zero_sized_input.to_shape([100, 0].as_slice());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reshape_with_zero_dim() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        // When the target shape has a zero dim, the corresponding input dim
        // size should be copied.
        let input = Tensor::from_data(&[1, 1, 4], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = NdTensor::from([-1, 0]);
        let expected = input.to_shape([4, 1].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Case where copied input dim is also zero.
        let input = Tensor::from([0.; 0]);
        let shape = NdTensor::from([0]);
        let expected = input.to_shape([0].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Case where there is no corresponding input dim.
        let input = Tensor::from([5.]);
        let shape = NdTensor::from([1, 0]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        );
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Zero dim has no corresponding input dim"
            ))
        );

        // Case when allow_zero is true
        let input = Tensor::<f32>::from_data(&[0, 0, 10], vec![]);
        let shape = NdTensor::from([10, 0, 0]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            true, /* allow_zero */
        )
        .unwrap();
        let expected = input.to_shape([10, 0, 0].as_slice());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reshape_with_multiple_unspecified_dims() {
        let pool = BufferPool::new();
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = NdTensor::from([1, -1, -1]);
        assert_eq!(
            reshape(
                &pool,
                input.view(),
                &shape.view(),
                false /* allow_zero */
            )
            .err(),
            Some(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1"
            ))
        );
    }

    #[test]
    fn test_reshape_with_unsolvable_unspecified_dim() {
        let pool = BufferPool::new();
        let expected_err = Some(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));

        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = NdTensor::from([5, -1]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        );
        assert_eq!(result.err(), expected_err);

        // Case when allow_zero is true
        let input = Tensor::from([1]);
        let shape = NdTensor::from([0, -1]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            true, /* allow_zero */
        );
        assert_eq!(result.err(), expected_err);
    }

    #[test]
    fn test_reshape_in_place() {
        let pool = BufferPool::new();
        let mut input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = NdTensor::from([4]);
        let expected = input.to_shape([4].as_slice());
        reshape_in_place(
            &pool,
            &mut input,
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        assert_eq!(&input, &expected);
    }

    #[test]
    fn test_reshape_op() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = Tensor::from([4]);
        let expected = input.to_shape([4].as_slice());

        let op = Reshape { allow_zero: false };
        let result: Tensor<f32> = op.run_simple((&input, &shape))?;

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_shape() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            op: Shape,
            expected: Vec<i32>,
        }

        let cases = [
            // No `start` or `end` offsets.
            Case {
                input: Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).into(),
                op: Shape::default(),
                expected: [1, 1, 2, 2].into(),
            },
            Case {
                input: Tensor::<i32>::zeros(&[1, 2, 3, 4]).into(),
                op: Shape::default(),
                expected: [1, 2, 3, 4].into(),
            },
            // Positive offsets.
            Case {
                input: Tensor::<i32>::zeros(&[1, 2, 3, 4]).into(),
                op: Shape {
                    start: Some(1),
                    end: Some(3),
                },
                expected: [2, 3].into(),
            },
            // Negative offsets.
            Case {
                input: Tensor::<i32>::zeros(&[1, 2, 3, 4]).into(),
                op: Shape {
                    start: Some(-3),
                    end: Some(-1),
                },
                expected: [2, 3].into(),
            },
            // Out of bound offsets.
            Case {
                input: Tensor::<i32>::zeros(&[1, 2, 3, 4]).into(),
                op: Shape {
                    start: Some(-6),
                    end: Some(7),
                },
                expected: [1, 2, 3, 4].into(),
            },
            // Start > end
            Case {
                input: Tensor::<i32>::zeros(&[1, 2, 3, 4]).into(),
                op: Shape {
                    start: Some(2),
                    end: Some(1),
                },
                expected: [].into(),
            },
            // Scalar tensor
            Case {
                input: Tensor::from(1i32).into(),
                op: Shape::default(),
                expected: [].into(),
            },
        ];

        cases.test_each(|case| {
            let result: Tensor<i32> = case.op.run_simple(&case.input).unwrap();
            assert_eq!(result.shape(), &[case.expected.len()]);
            assert_eq!(result.to_vec(), case.expected);
        });
    }

    #[test]
    fn test_size() {
        let op = Size {};
        let input = Tensor::from([[1, 2], [3, 4]]);
        let result: Tensor<i32> = op.run_simple(&input).unwrap();
        assert_eq!(result.ndim(), 0);
        assert_eq!(result.item(), Some(&4));
    }

    #[test]
    fn test_squeeze() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[1, 5, 5, 1], &mut rng);
        let mut expected = input.clone();

        // Remove all 1-size axes.
        expected.reshape(&[5, 5]);
        let result = squeeze(&pool, input.view(), None).unwrap();
        expect_equal(&result, &expected)?;

        // Remove final 1-size axis.
        expected.reshape(&[1, 5, 5]);
        let result = squeeze(&pool, input.view(), Some(NdTensor::from([3]).view())).unwrap();
        expect_equal(&result, &expected)?;

        // Remove first 1-size axis.
        expected.reshape(&[5, 5, 1]);
        let result = squeeze(&pool, input.view(), Some(NdTensor::from([0]).view())).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_squeeze_in_place() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(5678);

        // Contiguous tensor
        let mut input = Tensor::<f32>::rand(&[1, 1, 5, 5], &mut rng);
        let expected = input.clone().into_shape([5, 5].as_slice());

        squeeze_in_place(&mut input, None).unwrap();
        expect_equal(&input, &expected)?;

        // Non-contiguous tensor
        let mut input = Tensor::<f32>::rand(&[1, 5, 2, 5], &mut rng);
        input.permute(&[3, 2, 1, 0]);
        let expected = input.clone().into_shape([5, 2, 5].as_slice());

        squeeze_in_place(&mut input, None).unwrap();
        expect_equal(&input, &expected)?;

        Ok(())
    }

    #[test]
    fn test_squeeze_invalid_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[1, 5, 5, 1], &mut rng);

        let pool = BufferPool::new();
        let result = squeeze(&pool, input.view(), Some(NdTensor::from([1]).view()));

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Can only remove dimensions of size 1"
            ))
        );
    }

    #[test]
    fn test_transpose() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[10, 20], &mut rng);

        let mut reversed = input.clone();
        reversed.permute(&[1, 0]);

        // With no explicit permutation given, the axes should be reversed.
        let result = transpose(&pool, input.view(), None).unwrap();
        expect_equal(&result, &reversed)?;

        // With a no-op permutation given, the output should be unchanged.
        let result = transpose(&pool, input.view(), Some(&[0, 1])).unwrap();
        expect_equal(&result, &input)?;

        // With a transposed permutation given, the axes should be reversed.
        let result = transpose(&pool, input.view(), Some(&[1, 0])).unwrap();
        expect_equal(&result, &reversed)?;

        Ok(())
    }

    #[test]
    fn test_transpose_invalid_inputs() {
        let pool = BufferPool::new();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[10, 20], &mut rng);

        // Too many dims
        let result = transpose(&pool, input.view(), Some(&[0, 1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Too few dims
        let result = transpose(&pool, input.view(), Some(&[]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Invalid dimension index
        let result = transpose(&pool, input.view(), Some(&[2, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Repeated dimension index
        let result = transpose(&pool, input.view(), Some(&[1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );
    }

    #[test]
    fn test_unsqueeze() {
        let pool = BufferPool::new();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[3, 4, 5], &mut rng);

        // Unsqueeze with axes in increasing order
        let output = unsqueeze(&pool, input.view(), &NdTensor::from([0, 4]).view()).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze with axes in decreasing order
        let output = unsqueeze(&pool, input.view(), &NdTensor::from([4, 0]).view()).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze a scalar into a 1-item vec
        let scalar = Tensor::from(2.0);
        let output = unsqueeze(&pool, scalar.view(), &NdTensor::from([0]).view()).unwrap();
        assert_eq!(output.shape(), &[1]);
        assert_eq!(output.to_vec(), &[2.0]);
    }

    #[test]
    fn test_unsqueeze_invalid_inputs() {
        let pool = BufferPool::new();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::<f32>::rand(&[10, 20], &mut rng);

        // Invalid dimension index
        let result = unsqueeze(&pool, input.view(), &NdTensor::from([3]).view());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Repeated dimension index
        let result = unsqueeze(&pool, input.view(), &NdTensor::from([1, 1]).view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Axes must be unique"))
        );
    }

    use rten_tensor::{NdTensorView, TensorView, TensorViewMut};

    fn reference_transpose_into<'a, T: Clone>(src: TensorView<T>, mut dest: TensorViewMut<T>) {
        // Merge axes to maximize iteration count of the innermost loops.
        let mut src = src.clone();
        src.merge_axes();

        while src.ndim() < 4 {
            src.insert_axis(0);
        }

        let dest_data = dest.data_mut().unwrap();

        let src: NdTensorView<T, 4> = src.nd_view();
        let mut dest_offset = 0;
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        unsafe {
                            let elt = src.get_unchecked([i0, i1, i2, i3]).clone();
                            *dest_data.get_unchecked_mut(dest_offset) = elt;
                            dest_offset += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_transpose() {
        let mut rng = XorShiftRng::new(1234);

        struct Case<'a> {
            /// Input shape
            shape: &'a [usize],

            /// Permutation order (eg. `[1, 0]` for a matrix transpose)
            perm: &'a [usize],
        }

        let cases = [
            // No-op transpose
            Case {
                shape: &[512, 512],
                perm: &[0, 1],
            },
            // Matrix transpose with power-of-2 sizes.
            //
            // In a naive transpose implementation, these are liable to
            // experience slowdown due to poor cache usage. There can also be
            // issues to a lesser extent with sizes which are a multiple of
            // (cache_line_size / element_size).
            Case {
                shape: &[128, 128],
                perm: &[1, 0],
            },
            Case {
                shape: &[256, 256],
                perm: &[1, 0],
            },
            Case {
                shape: &[512, 512],
                perm: &[1, 0],
            },
            Case {
                shape: &[1024, 1024],
                perm: &[1, 0],
            },
            // Matrix transpose with non power-of-2 sizes.
            Case {
                shape: &[127, 127],
                perm: &[1, 0],
            },
            Case {
                shape: &[255, 255],
                perm: &[1, 0],
            },
            Case {
                shape: &[513, 513],
                perm: &[1, 0],
            },
            Case {
                shape: &[1023, 1023],
                perm: &[1, 0],
            },
            // Transpose ops taken from Whisper encoder (base model) with 4
            // batches of samples.
            //
            // Note the last two dimensions are powers of 2.
            Case {
                shape: &[4, 1500, 8, 64],
                perm: &[0, 2, 1, 3],
            },
            Case {
                shape: &[4, 8, 1500, 64],
                perm: &[0, 2, 1, 3],
            },
            // Transpose ops taken from Whisper decoder (base model)
            Case {
                shape: &[1, 1500, 8, 64],
                perm: &[0, 2, 3, 1],
            },
            Case {
                shape: &[1, 288, 8, 64],
                perm: &[0, 2, 1, 3],
            },
        ];

        for Case { shape, perm } in cases {
            let tensor = Tensor::<f32>::rand(shape, &mut rng);
            let mut dest = Tensor::zeros(shape);

            // Do a simple copy. This provides a lower-bound on how fast
            // transpose can operate.
            let copy_stats = run_bench(100, None, || {
                dest.copy_from(&tensor.view());
            });
            assert_eq!(dest, tensor);

            let reference_transpose_stats = run_bench(100, None, || {
                let transposed = tensor.permuted(perm);
                reference_transpose_into(
                    transposed.view(),
                    dest.reshaped_mut(transposed.shape()).unwrap(),
                );
            });

            let transpose_stats = run_bench(100, None, || {
                let transposed = tensor.permuted(perm);
                dest.reshape(transposed.shape());
                dest.copy_from(&transposed);
            });
            assert_eq!(dest, tensor.permuted(perm));

            let transpose_overhead =
                (transpose_stats.mean - copy_stats.mean).max(0.) / copy_stats.mean;
            println!(
                "transpose shape {:?} perm {:?} copy {:.3}ms ref transpose {:.3}ms opt transpose {:.3}ms overhead {}",
                shape,
                perm,
                copy_stats.median,
                reference_transpose_stats.median,
                transpose_stats.median,
                transpose_overhead
            );
        }
    }
}

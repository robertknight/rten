//! ONNX Runtime contrib gather operators.

use std::mem::MaybeUninit;

use rayon::prelude::*;

use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{InitEmpty, Tensor, TensorView};

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::quantize::Dequantize;
use crate::ops::{resolve_axis, resolve_index};
use crate::value::{DataType, ValueType};

use super::INVALID_INDEX_ERR;

/// Extract the `index`th `bits`-sized element from a byte.
///
/// Elements are packed with the first element in the lowest-order bits.
fn unpack(byte: u8, index: usize, bits: usize) -> i32 {
    let mask = (1 << bits) - 1;
    ((byte as usize >> (index * bits)) & mask) as i32
}

/// Dequantize one block of `BITS`-wide elements packed into `block_bytes`.
///
/// `out` and `block_bytes` must have the same length in elements, ie.
/// `out.len() == block_bytes.len() * (8 / BITS)`.
fn dequantize_block<const BITS: usize>(
    out: &mut [MaybeUninit<f32>],
    block_bytes: &[u8],
    scale: f32,
    zero_point: i32,
) {
    let components = 8 / BITS;
    for (out_chunk, &byte) in out.chunks_exact_mut(components).zip(block_bytes) {
        for (i, out) in out_chunk.iter_mut().enumerate() {
            let quantized = unpack(byte, i, BITS);
            out.write(quantized.dequantize(scale, zero_point));
        }
    }
}

fn gather_block_quantized(
    pool: &BufferPool,
    data: TensorView<u8>,
    indices: TensorView<i32>,
    scales: TensorView<f32>,
    zero_points: Option<TensorView<u8>>,
    bits: usize,
    block_size: usize,
    gather_axis: isize,
    quantize_axis: isize,
) -> Result<Tensor<f32>, OpError> {
    // Only the inner dequantization loop is specialized on the bit width. The
    // outer loops and validation use `bits` as a runtime value.
    let dequantize_block: fn(&mut [MaybeUninit<f32>], &[u8], f32, i32) = match bits {
        2 => dequantize_block::<2>,
        4 => dequantize_block::<4>,
        8 => dequantize_block::<8>,
        _ => return Err(OpError::UnsupportedValue("`bits` must be 2, 4 or 8")),
    };

    // Number of quantized elements packed into each byte of `data` and
    // `zero_points`.
    let components = 8 / bits;

    if block_size < 16 || !block_size.is_power_of_two() {
        return Err(OpError::UnsupportedValue(
            "`block_size` must be a power of 2 and at least 16",
        ));
    }

    let ndim = data.ndim();
    if ndim < 2 {
        return Err(OpError::InvalidValue("`data` must have at least 2 dims"));
    }
    let gather_axis = resolve_axis(ndim, gather_axis)?;
    let quantize_axis = resolve_axis(ndim, quantize_axis)?;

    // ONNX Runtime requires uint8 data to be gathered along the first axis
    // and quantized along the last axis, so only these cases need to be
    // supported.
    if gather_axis != 0 {
        return Err(OpError::UnsupportedValue(
            "Only `gather_axis` of 0 is supported for uint8 data",
        ));
    }
    if quantize_axis != ndim - 1 {
        return Err(OpError::UnsupportedValue(
            "Only `quantize_axis` of -1 is supported for uint8 data",
        ));
    }

    // Packed bytes and dequantized elements in each lane along the
    // quantize axis.
    let packed_lane_len = data.size(ndim - 1);
    let lane_len = packed_lane_len * components;
    let n_blocks = lane_len.div_ceil(block_size);

    if scales.ndim() != ndim
        || scales.shape()[..ndim - 1] != data.shape()[..ndim - 1]
        || scales.size(ndim - 1) != n_blocks
    {
        return Err(OpError::IncompatibleInputShapes(
            "`scales` shape does not match quantization blocks in `data`",
        ));
    }

    // Zero points are packed along the last axis in the same way as `data`.
    let zp_lane_len = n_blocks.div_ceil(components);
    if let Some(zero_points) = &zero_points
        && (zero_points.ndim() != ndim
            || zero_points.shape()[..ndim - 1] != data.shape()[..ndim - 1]
            || zero_points.size(ndim - 1) != zp_lane_len)
    {
        return Err(OpError::IncompatibleInputShapes(
            "`zero_points` shape does not match `scales` shape",
        ));
    }
    let default_zero_point = 1 << (bits - 1);

    let out_shape: Vec<usize> = indices
        .shape()
        .iter()
        .chain(&data.shape()[1..ndim - 1])
        .copied()
        .chain([lane_len])
        .collect();

    let data = data.to_contiguous_in(pool).auto_return(pool);
    let scales = scales.to_contiguous_in(pool).auto_return(pool);
    let zero_points = zero_points.map(|zp| zp.to_contiguous_in(pool).auto_return(pool));
    let indices = indices.to_contiguous_in(pool).auto_return(pool);

    let data_slice = data.data();
    let scales_slice = scales.data();
    let zp_slice = zero_points.as_ref().map(|zp| zp.data());

    // Sizes of the slices of `data`, `scales` and `zero_points` that
    // correspond to one entry along the gather axis.
    let lanes_per_entry: usize = data.shape()[1..ndim - 1].iter().product();
    let data_entry_len = lanes_per_entry * packed_lane_len;
    let scales_entry_len = lanes_per_entry * n_blocks;
    let zp_entry_len = lanes_per_entry * zp_lane_len;
    let out_entry_len = lanes_per_entry * lane_len;

    let axis_size = data.size(0);
    let output = Tensor::uninit_in(pool, &out_shape);

    // An empty output means there is nothing to dequantize, but indices must
    // still be validated. This also guarantees that `out_entry_len` is
    // non-zero below, as required by `par_chunks_mut`.
    let mut output = match output.init_if_empty() {
        InitEmpty::Empty(output) => {
            for &index in indices.iter() {
                resolve_index(axis_size, index as isize).ok_or(INVALID_INDEX_ERR)?;
            }
            return Ok(output);
        }
        InitEmpty::NotEmpty(output) => output,
    };

    output
        .data_mut()
        .unwrap()
        .par_chunks_mut(out_entry_len)
        .zip(indices.data().par_iter())
        .try_for_each(|(out_entry, &index)| {
            let index = resolve_index(axis_size, index as isize).ok_or(INVALID_INDEX_ERR)?;
            let data_entry = &data_slice[index * data_entry_len..][..data_entry_len];
            let scales_entry = &scales_slice[index * scales_entry_len..][..scales_entry_len];
            let zp_entry = zp_slice.map(|zp| &zp[index * zp_entry_len..][..zp_entry_len]);

            for lane in 0..lanes_per_entry {
                let data_lane = &data_entry[lane * packed_lane_len..][..packed_lane_len];
                let scales_lane = &scales_entry[lane * n_blocks..][..n_blocks];
                let out_lane = &mut out_entry[lane * lane_len..][..lane_len];

                for (block, &scale) in scales_lane.iter().enumerate() {
                    let zero_point = zp_entry
                        .map(|zp| {
                            unpack(
                                zp[lane * zp_lane_len + block / components],
                                block % components,
                                bits,
                            )
                        })
                        .unwrap_or(default_zero_point);

                    // Block boundaries are always byte-aligned because
                    // `block_size` and `lane_len` are both multiples of
                    // `components`.
                    let start = block * block_size;
                    let end = (start + block_size).min(lane_len);
                    let block_bytes = &data_lane[start / components..end / components];
                    let out_block = &mut out_lane[start..end];

                    dequantize_block(out_block, block_bytes, scale, zero_point);
                }
            }
            Ok::<_, OpError>(())
        })?;

    // Safety: Each chunk of the output corresponds to one index, and the
    // loop over blocks fully initializes each lane of the chunk.
    Ok(unsafe { output.assume_init() })
}

/// Gather from a tensor quantized block-wise along one axis, dequantizing
/// the gathered data.
///
/// This is typically used for quantized embedding lookup tables.
///
/// See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatherBlockQuantized.
#[derive(Debug)]
pub struct GatherBlockQuantized {
    /// Number of bits used to quantize each element of `data`. Must be 2,
    /// 4 or 8.
    pub bits: usize,

    /// Number of consecutive elements along the quantize axis which share
    /// a scale and zero point.
    pub block_size: usize,

    /// Axis of `data` along which to gather.
    pub gather_axis: isize,

    /// Axis of `data` along which elements are quantized block-wise.
    pub quantize_axis: isize,
}

impl Operator for GatherBlockQuantized {
    fn name(&self) -> &str {
        "GatherBlockQuantized"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let data: TensorView<u8> = inputs.require_as(0)?;
        let indices: TensorView<i32> = inputs.require_as(1)?;
        let scales: TensorView<f32> = inputs.require_as(2)?;
        let zero_points: Option<TensorView<u8>> = inputs.get_as(3)?;

        gather_block_quantized(
            ctx.pool(),
            data,
            indices,
            scales,
            zero_points,
            self.bits,
            self.block_size,
            self.gather_axis,
            self.quantize_axis,
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl_infer_shapes!(
    GatherBlockQuantized,
    op,
    shape_ops::GatherBlockQuantized {
        gather_axis: op.gather_axis as i32,
        bits: op.bits as i32,
    }
);

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use crate::operator::{InputList, OpError, OperatorExt};

    use super::GatherBlockQuantized;

    #[test]
    fn test_gather_block_quantized() {
        #[derive(Debug)]
        struct Case {
            bits: usize,
            block_size: usize,
            gather_axis: isize,
            quantize_axis: isize,
            data: Tensor<u8>,
            indices: Tensor<i32>,
            scales: Tensor<f32>,
            zero_points: Option<Tensor<u8>>,
            expected: Result<Tensor<f32>, OpError>,
        }

        // 4-bit quantized rows used by several cases. Each byte packs two
        // elements, low-order nibble first, so the unpacked rows are
        // [1, 2, 3, 4], [7, 8, 9, 10] and [15, 15, 0, 0].
        let data_4bit = Tensor::from([[0x21u8, 0x43], [0x87, 0xA9], [0xFF, 0x00]]);

        let cases = [
            // 4-bit data with zero points. Also uses a negative quantize_axis.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: -1,
                data: data_4bit.clone(),
                indices: [[0, 2]].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: Some([[0x02u8], [0x08], [0x00]].into()),
                expected: Ok([[[-0.5, 0., 0.5, 1.], [30., 30., 0., 0.]]].into()),
            },
            // 4-bit data without zero points (default is 8), negative index.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [-1].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Ok([[14., 14., -16., -16.]].into()),
            },
            // Multiple blocks per lane, with the zero points for adjacent
            // blocks packed into one byte. Scalar indices.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: Tensor::from_data(&[1, 16], vec![0u8; 16]),
                indices: Tensor::from(0),
                scales: [[1.0, 2.0]].into(),
                zero_points: Some([[0x21u8]].into()),
                expected: Ok(Tensor::from_data(
                    &[32],
                    [vec![-1.0f32; 16], vec![-4.0; 16]].concat(),
                )),
            },
            // 3D data, where each gathered entry contains multiple lanes along
            // the quantize axis.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 2,
                data: Tensor::from([[[0x21u8, 0x43], [0x65, 0x87]], [[0x00, 0x00], [0xFF, 0xFF]]]),
                indices: [1].into(),
                scales: Tensor::from([[[1.0], [2.0]], [[0.5], [1.0]]]),
                zero_points: None,
                expected: Ok(Tensor::from([[[-4., -4., -4., -4.], [7., 7., 7., 7.]]])),
            },
            // 8-bit data. Elements are not packed and the default zero point
            // is 128.
            Case {
                bits: 8,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: Tensor::from_data(&[1, 16], (0..16u8).collect::<Vec<_>>()),
                indices: [0].into(),
                scales: [[2.0]].into(),
                zero_points: None,
                expected: Ok(Tensor::from_data(
                    &[1, 16],
                    (0..16).map(|i| (i as f32 - 128.) * 2.).collect::<Vec<_>>(),
                )),
            },
            // 2-bit data. Each byte packs four elements and the default zero
            // point is 2.
            Case {
                bits: 2,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: Tensor::from([[0xE4u8, 0xE4, 0x00, 0xFF]]),
                indices: [0].into(),
                scales: [[1.0]].into(),
                zero_points: None,
                expected: Ok(Tensor::from([[
                    -2., -1., 0., 1., -2., -1., 0., 1., -2., -2., -2., -2., 1., 1., 1., 1.,
                ]])),
            },
            // Empty entries, so the output has no elements.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: Tensor::zeros(&[3, 0]),
                indices: [0].into(),
                scales: Tensor::zeros(&[3, 0]),
                zero_points: None,
                expected: Ok(Tensor::zeros(&[1, 0])),
            },
            // Out-of-range index.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [3].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Err(OpError::InvalidValue("Entry in `indices` is out of range")),
            },
            // Out-of-range index with an empty output. Indices are validated
            // even though there is nothing to dequantize.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: Tensor::zeros(&[3, 0]),
                indices: [3].into(),
                scales: Tensor::zeros(&[3, 0]),
                zero_points: None,
                expected: Err(OpError::InvalidValue("Entry in `indices` is out of range")),
            },
            // Scales shape does not match quantization blocks.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5, 1.0], [1.0, 1.0], [2.0, 1.0]].into(),
                zero_points: None,
                expected: Err(OpError::IncompatibleInputShapes(
                    "`scales` shape does not match quantization blocks in `data`",
                )),
            },
            // Zero points shape does not match scales shape.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: Some([[0x02u8, 0x00], [0x08, 0x00], [0x00, 0x00]].into()),
                expected: Err(OpError::IncompatibleInputShapes(
                    "`zero_points` shape does not match `scales` shape",
                )),
            },
            // Unsupported gather axis.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 1,
                quantize_axis: 0,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Err(OpError::UnsupportedValue(
                    "Only `gather_axis` of 0 is supported for uint8 data",
                )),
            },
            // Unsupported quantize axis.
            Case {
                bits: 4,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 0,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Err(OpError::UnsupportedValue(
                    "Only `quantize_axis` of -1 is supported for uint8 data",
                )),
            },
            // Unsupported block size.
            Case {
                bits: 4,
                block_size: 8,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Err(OpError::UnsupportedValue(
                    "`block_size` must be a power of 2 and at least 16",
                )),
            },
            // Unsupported bits.
            Case {
                bits: 3,
                block_size: 16,
                gather_axis: 0,
                quantize_axis: 1,
                data: data_4bit.clone(),
                indices: [0].into(),
                scales: [[0.5], [1.0], [2.0]].into(),
                zero_points: None,
                expected: Err(OpError::UnsupportedValue("`bits` must be 2, 4 or 8")),
            },
        ];

        cases.test_each(|case| {
            let op = GatherBlockQuantized {
                bits: case.bits,
                block_size: case.block_size,
                gather_axis: case.gather_axis,
                quantize_axis: case.quantize_axis,
            };
            let mut inputs: InputList =
                (case.data.view(), case.indices.view(), case.scales.view()).into();
            inputs.push_optional(case.zero_points.as_ref().map(|zp| zp.view()));
            let result: Result<Tensor<f32>, OpError> = op.run_simple(inputs);
            assert_eq!(result, case.expected);
        });
    }
}

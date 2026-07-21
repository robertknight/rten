//! Fused GLU (Gated Linear Unit) activations used in transformer models.

use rayon::prelude::*;

use rten_base::bit_set::BitSet;
use rten_simd::{Isa, SimdOp, SimdUnaryOp};
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use rten_vecmath as vecmath;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};

use super::CHUNK_SIZE;

/// The activation function applied to the gate input of a [`Glu`] operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GluActivation {
    /// Silu activation, used in SwiGLU layers.
    Silu,
    /// Gelu activation, used in GeGLU layers.
    Gelu { approximate: bool },
}

impl SimdUnaryOp<f32> for GluActivation {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        match self {
            GluActivation::Silu => vecmath::Silu {}.eval(isa, x),
            GluActivation::Gelu { approximate: false } => vecmath::Gelu {}.eval(isa, x),
            GluActivation::Gelu { approximate: true } => vecmath::ApproxGelu {}.eval(isa, x),
        }
    }
}

const SHAPE_MISMATCH: OpError = OpError::IncompatibleInputShapes("Inputs must have the same shape");

fn glu(
    pool: &BufferPool,
    activation: GluActivation,
    a: TensorView,
    b: TensorView,
) -> Result<Tensor, OpError> {
    if a.shape() != b.shape() {
        return Err(SHAPE_MISMATCH);
    }

    // Fast path for contiguous inputs, processing the data as one flat slice.
    // This also handles zero-dim and empty tensors.
    if a.is_contiguous() && b.is_contiguous() {
        let mut output = Tensor::uninit_in(pool, a.shape());
        let a_chunks = a.data().unwrap().par_chunks(CHUNK_SIZE);
        let b_chunks = b.data().unwrap().par_chunks(CHUNK_SIZE);
        let out_chunks = output.data_mut().unwrap().par_chunks_mut(CHUNK_SIZE);
        a_chunks
            .zip(b_chunks)
            .zip(out_chunks)
            .for_each(|((a_chunk, b_chunk), out_chunk)| {
                vecmath::Glu::new(activation, (a_chunk, out_chunk), b_chunk).dispatch();
            });

        // Safety: The loop above initialized each chunk of the output.
        return Ok(unsafe { output.assume_init() });
    }

    // Copy inputs whose innermost lanes are not contiguous. Only the lanes
    // need to be contiguous, so this is expected to be a no-op in most cases.
    let axis = a.ndim() - 1;
    if a.stride(axis) != 1 {
        let a = a.to_contiguous_in(pool).auto_return(pool);
        return glu(pool, activation, a.view().into(), b);
    }
    if b.stride(axis) != 1 {
        let b = b.to_contiguous_in(pool).auto_return(pool);
        return glu(pool, activation, a, b.view().into());
    }

    let mut output = Tensor::uninit_in(pool, a.shape());
    output
        .lanes_mut(axis)
        .into_par_iter()
        .zip(
            a.lanes(axis)
                .into_par_iter()
                .zip(b.lanes(axis).into_par_iter()),
        )
        .for_each(|(mut out_lane, (a_lane, b_lane))| {
            // OK: The lanes are contiguous, as checked above, and the output
            // is contiguous since it was newly allocated.
            let out_lane = out_lane.as_slice_mut().unwrap();
            let a_lane = a_lane.as_slice().unwrap();
            let b_lane = b_lane.as_slice().unwrap();
            vecmath::Glu::new(activation, (a_lane, out_lane), b_lane).dispatch();
        });

    // Safety: The loop above initialized each lane of the output.
    Ok(unsafe { output.assume_init() })
}

fn glu_in_place(
    pool: &BufferPool,
    activation: GluActivation,
    mut a: Tensor,
    b: TensorView,
) -> Result<Tensor, OpError> {
    if a.shape() != b.shape() {
        return Err(SHAPE_MISMATCH);
    }

    // Fast path for contiguous inputs, processing the data as one flat slice.
    if a.is_contiguous() && b.is_contiguous() {
        a.data_mut()
            .unwrap()
            .par_chunks_mut(CHUNK_SIZE)
            .zip(b.data().unwrap().par_chunks(CHUNK_SIZE))
            .for_each(|(a_chunk, b_chunk)| {
                vecmath::Glu::new(activation, a_chunk, b_chunk).dispatch();
            });
        return Ok(a);
    }

    let axis = a.ndim() - 1;
    if a.stride(axis) != 1 {
        // The innermost lanes of `a` are not contiguous, so the result cannot
        // be computed in place lane-by-lane. Fall back to allocating a new
        // output.
        let a = a.auto_return(pool);
        return glu(pool, activation, a.view(), b);
    }
    if b.stride(axis) != 1 {
        let b = b.to_contiguous_in(pool).auto_return(pool);
        return glu_in_place(pool, activation, a, b.view().into());
    }

    a.lanes_mut(axis)
        .into_par_iter()
        .zip(b.lanes(axis).into_par_iter())
        .for_each(|(mut a_lane, b_lane)| {
            // OK: The lanes are contiguous, as checked above.
            let a_lane = a_lane.as_slice_mut().unwrap();
            let b_lane = b_lane.as_slice().unwrap();
            vecmath::Glu::new(activation, a_lane, b_lane).dispatch();
        });
    Ok(a)
}

/// Gated Linear Unit activation.
///
/// This computes `activation(A) * B` where `A` and `B` must have the same
/// shape. It is a fusion of the SwiGLU (`activation` = Silu) and GeGLU
/// (`activation` = Gelu) patterns found in the feed-forward layers of
/// transformer models.
#[derive(Debug)]
pub struct Glu {
    pub activation: GluActivation,

    /// If true, the operator takes a single input containing `A` and `B`
    /// concatenated along the last axis, instead of two separate inputs.
    pub split_input: bool,
}

impl Operator for Glu {
    fn name(&self) -> &str {
        "Glu"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        if self.split_input {
            let x: TensorView = ctx.inputs().require_as(0)?;
            if x.ndim() == 0 || x.size(x.ndim() - 1) % 2 != 0 {
                return Err(OpError::InvalidValue(
                    "Last dimension of input must be even",
                ));
            }
            let axis = x.ndim() - 1;
            let half = x.size(axis) / 2;
            let a = x.slice_axis(axis, 0..half);
            let b = x.slice_axis(axis, half..half * 2);
            glu(ctx.pool(), self.activation, a, b).into_op_result()
        } else {
            let a: TensorView = ctx.inputs().require_as(0)?;
            let b: TensorView = ctx.inputs().require_as(1)?;
            glu(ctx.pool(), self.activation, a, b).into_op_result()
        }
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        if self.split_input {
            // The output has a different shape than the input.
            BitSet::new()
        } else {
            BitSet::from_indices([0])
        }
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let a: Tensor = in_place.into_single().try_into()?;
        let b: TensorView = ctx.inputs().require_first_present_as()?;
        glu_in_place(ctx.pool(), self.activation, a, b).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        if self.split_input {
            None
        } else {
            // Output shape is the shape of the first input, since both inputs
            // must have the same shape.
            Some(&UnaryOp)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_testing::TestCases;

    use super::super::tests::{reference_approx_gelu, reference_gelu, reference_sigmoid};
    use super::{Glu, GluActivation};
    use crate::operator::{OpError, OperatorExt};

    #[test]
    fn test_glu() -> Result<(), Box<dyn Error>> {
        #[derive(Debug)]
        struct Case {
            activation: GluActivation,
            reference: fn(f32) -> f32,
        }

        let cases = [
            Case {
                activation: GluActivation::Silu,
                reference: |x| x * reference_sigmoid(x),
            },
            Case {
                activation: GluActivation::Gelu { approximate: false },
                reference: reference_gelu,
            },
            Case {
                activation: GluActivation::Gelu { approximate: true },
                reference: reference_approx_gelu,
            },
        ];

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::<f32>::rand(&[3, 4], &mut rng);
            let b = Tensor::<f32>::rand(&[3, 4], &mut rng);

            let expected = Tensor::from_data(
                a.shape(),
                a.iter()
                    .zip(b.iter())
                    .map(|(a, b)| (case.reference)(*a) * b)
                    .collect::<Vec<_>>(),
            );

            let op = Glu {
                activation: case.activation,
                split_input: false,
            };
            let result: Tensor = op.run_simple((a.view(), b.view())).unwrap();
            expect_equal(&result, &expected).unwrap();

            let in_place_result: Tensor = op.run_simple_in_place(a.clone(), b.view()).unwrap();
            expect_equal(&in_place_result, &expected).unwrap();
        });

        Ok(())
    }

    // Test with inputs whose innermost lanes are contiguous, but which are
    // not contiguous overall. These can be used without copying.
    #[test]
    fn test_glu_non_contiguous_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let a_base = Tensor::<f32>::rand(&[3, 6], &mut rng);
        let b_base = Tensor::<f32>::rand(&[3, 6], &mut rng);
        let a = a_base.slice((.., ..4));
        let b = b_base.slice((.., 2..));

        let expected = Tensor::from_data(
            a.shape().to_vec().as_slice(),
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| a * reference_sigmoid(*a) * b)
                .collect::<Vec<_>>(),
        );

        let op = Glu {
            activation: GluActivation::Silu,
            split_input: false,
        };
        let result: Tensor = op.run_simple((a, b)).unwrap();
        expect_equal(&result, &expected).unwrap();
    }

    // Test the in-place path with an input whose innermost lanes are not
    // contiguous. This falls back to allocating a new output.
    #[test]
    fn test_glu_in_place_transposed_input() {
        let mut rng = XorShiftRng::new(5678);
        let mut a = Tensor::<f32>::rand(&[4, 3], &mut rng);
        a.permute(&[1, 0]);
        let b = Tensor::<f32>::rand(&[3, 4], &mut rng);

        let expected = Tensor::from_data(
            a.shape().to_vec().as_slice(),
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| a * reference_sigmoid(*a) * b)
                .collect::<Vec<_>>(),
        );

        let op = Glu {
            activation: GluActivation::Silu,
            split_input: false,
        };
        let result: Tensor = op.run_simple_in_place(a, b.view()).unwrap();
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_glu_shape_mismatch() {
        let a = Tensor::<f32>::zeros(&[3, 4]);
        let b = Tensor::<f32>::zeros(&[3, 5]);
        let op = Glu {
            activation: GluActivation::Silu,
            split_input: false,
        };
        let result = op.run_simple::<_, Tensor>((a.view(), b.view()));
        assert_eq!(
            result,
            Err(OpError::IncompatibleInputShapes(
                "Inputs must have the same shape"
            ))
        );
    }
}

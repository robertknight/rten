use rten_tensor::{AsView, Layout, NdTensorView, Tensor, TensorView};

use crate::value::Value;
use crate::{
    operator::{OpError, OpRunContext, Operator, OutputList, OutputTypeList, OutputTypesContext},
    ops::{
        binary_elementwise::{add, mul, sub},
        gather,
    },
};

#[derive(Debug)]
pub struct RotaryEmbedding {
    pub interleaved: isize,
    pub num_heads: Option<usize>,
    pub rotary_embedding_dim: usize,
}

impl Operator for RotaryEmbedding {
    fn name(&self) -> &str {
        "RotaryEmbedding"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        if inputs.len() < 3 || inputs.len() > 4 {
            return Err(OpError::MissingInputs);
        }

        let input: TensorView<f32> = inputs.require_as(0)?;
        let cos: TensorView<f32> = inputs.require_as(1)?;
        let sin: TensorView<f32> = inputs.require_as(2)?;
        let position_ids: Option<NdTensorView<i32, 1>> = inputs.get_as(3)?;

        let reshaped_input = match input.shape() {
            &[batch, _seq_len, hidden_size] => {
                let num_heads = self.num_heads.unwrap_or(0);
                if num_heads == 0 {
                    return Err(OpError::InvalidValue(
                        "num_heads must not be 0 for 3 dimensioned input",
                    ));
                }

                let head_size = hidden_size / num_heads;
                input.reshaped([batch, hidden_size, num_heads, head_size])
            }
            [_batch, _num_heads, _seq_len, _head_size] => {
                input.nd_view().permuted([0, 2, 1, 3]).as_cow()
            }
            _ => {
                return Err(OpError::IncompatibleInputShapes(
                    "Input processed needs 3-4 dimensions",
                ));
            }
        };

        let rotary_embedding_dim = if self.rotary_embedding_dim == 0 {
            reshaped_input.shape()[3]
        } else {
            self.rotary_embedding_dim
        };

        let x_rotate = reshaped_input.slice((.., .., .., ..rotary_embedding_dim));
        let x_not_rotate = reshaped_input.slice((.., .., .., rotary_embedding_dim..));

        let rotary_embedding_dim_half = rotary_embedding_dim / 2;

        // TODO this is a bit ugly, just doing it to coerce the subsets into the same type as cos
        // and sin if no position indices are given with the data living long enough
        let mut cos_subset = None;
        let mut sin_subset = None;

        let (cos_cache, sin_cache) = if let Some(position_ids) = position_ids {
            cos_subset = Some(gather(ctx.pool(), cos, 0, position_ids.as_dyn())?);
            sin_subset = Some(gather(ctx.pool(), sin, 0, position_ids.as_dyn())?);

            // Ugly unwrap
            (
                cos_subset.as_ref().map(|x| x.view()).unwrap(),
                sin_subset.as_ref().map(|x| x.view()).unwrap(),
            )
        } else {
            (cos, sin)
        };

        if cos_cache.shape()[cos_cache.ndim() - 1] != rotary_embedding_dim_half {
            return Err(OpError::InvalidValue(
                "Last dimension of cos cache does not match rotary_embedding_dim/2",
            ));
        }

        if sin_cache.shape()[cos_cache.ndim() - 1] != rotary_embedding_dim_half {
            return Err(OpError::InvalidValue(
                "Last dimension of sin cache does not match rotary_embedding_dim/2",
            ));
        }

        let cos_cache = cos_cache.with_new_axis(2);
        let sin_cache = sin_cache.with_new_axis(2);

        let (x1, x2) = if self.interleaved != 0 {
            (
                x_rotate.slice((.., .., .., 0..2)),
                x_rotate.slice((.., .., .., 1..2)),
            )
        } else {
            x_rotate.split_at(3, rotary_embedding_dim_half)
        };

        // TODO maybe inplace version is possible and nicer
        let lhs = mul(ctx.pool(), cos_cache.view(), x1.as_dyn())?;
        let rhs = mul(ctx.pool(), sin_cache.view(), x2.as_dyn())?;
        let real = sub(ctx.pool(), lhs.view(), rhs.view())?;

        let lhs = mul(ctx.pool(), sin_cache.view(), x1.as_dyn())?;
        let rhs = mul(ctx.pool(), cos_cache.view(), x2.as_dyn())?;
        let imag = add(ctx.pool(), lhs.view(), rhs.view())?;

        if self.interleaved != 0 {

            // real = np.expand_dims(real, axis=-1)
            // imag = np.expand_dims(imag, axis=-1)
            // x_rotate_concat = np.concatenate((real, imag), axis=-1)
            // x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
        } else {
            // x_rotate = np.concatenate((real, imag), axis=-1)
        }

        /*
        output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
        if len(original_input_shape) == 3:
            output = np.reshape(output, original_input_shape)
        else:
            output = np.transpose(output, (0, 2, 1, 3))
        return output
         */

        todo!()
    }

    fn is_commutative(&self) -> bool {
        false
    }

    fn can_run_in_place(&self) -> bool {
        false
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        todo!()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        todo!()
    }
}

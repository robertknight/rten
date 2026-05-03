use rten_tensor::{AsView, Layout, NdTensorView, Tensor, TensorView};

use crate::{
    buffer_pool::AutoReturn,
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
    ops::{
        binary_elementwise::{add, mul, sub},
        concat, gather,
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

        let (cos_cache, sin_cache) = if let Some(position_ids) = position_ids {
            let cos_subset = gather(ctx.pool(), cos, 0, position_ids.as_dyn())?.into_cow();
            let sin_subset = gather(ctx.pool(), sin, 0, position_ids.as_dyn())?.into_cow();
            (cos_subset, sin_subset)
        } else {
            (cos.as_cow(), sin.as_cow())
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

        let cos_cache = cos_cache.view().with_new_axis(2);
        let sin_cache = sin_cache.view().with_new_axis(2);

        let (x1, x2) = if self.interleaved != 0 {
            (
                x_rotate.slice((.., .., .., 0..2)),
                x_rotate.slice((.., .., .., 1..2)),
            )
        } else {
            x_rotate.split_at(3, rotary_embedding_dim_half)
        };

        let lhs = mul(ctx.pool(), cos_cache.view(), x1.as_dyn())?.auto_return(ctx.pool());
        let rhs = mul(ctx.pool(), sin_cache.view(), x2.as_dyn())?.auto_return(ctx.pool());
        let real = sub(ctx.pool(), lhs.view(), rhs.view())?.auto_return(ctx.pool());

        let lhs = mul(ctx.pool(), sin_cache.view(), x1.as_dyn())?.auto_return(ctx.pool());
        let rhs = mul(ctx.pool(), cos_cache.view(), x2.as_dyn())?.auto_return(ctx.pool());
        let imag = add(ctx.pool(), lhs.view(), rhs.view())?.auto_return(ctx.pool());

        let x_rotate = if self.interleaved != 0 {
            let insert_axis = real.ndim() - 1;
            let real = real.view().with_new_axis(insert_axis);
            let imag = imag.view().with_new_axis(insert_axis);

            let mut x_rotate_concat = concat(ctx.pool(), &[real, imag], -1)?;
            x_rotate_concat.reshape(&x_rotate.shape());
            x_rotate_concat
        } else {
            concat(ctx.pool(), &[real.view(), imag.view()], -1)?
        }
        .auto_return(ctx.pool());

        let output = concat(ctx.pool(), &[x_rotate.view(), x_not_rotate.as_dyn()], -1)?;

        let output = if input.ndim() == 3 {
            output.into_shape(input.shape())
        } else {
            input.permuted(&[0, 2, 1, 3]).to_tensor_in(ctx.pool())
        };

        output.into_op_result()
    }

    fn is_commutative(&self) -> bool {
        false
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

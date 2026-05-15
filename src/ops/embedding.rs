use rten_tensor::{AsView, Layout, NdTensorView, Tensor, TensorView};

use crate::{
    buffer_pool::AutoReturn,
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
    ops::{
        binary_elementwise::{add, mul, sub},
        concat, gather, slice,
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
            let starts_x1 = Tensor::<i32>::from([0]);
            let starts_x2 = Tensor::<i32>::from([1]);
            let ends = Tensor::<i32>::from([i32::MAX]); // i32::MAX means "to the end"
            let axes = Tensor::<i32>::from([3]);
            let steps = Tensor::<i32>::from([2]);

            let x1 = slice(
                ctx.pool(),
                x_rotate.as_dyn(),
                &starts_x1.nd_view(),
                &ends.nd_view(),
                Some(&axes.nd_view()),
                Some(&steps.nd_view()),
            )?;

            let x2 = slice(
                ctx.pool(),
                x_rotate.as_dyn(),
                &starts_x2.nd_view(),
                &ends.nd_view(),
                Some(&axes.nd_view()),
                Some(&steps.nd_view()),
            )?;

            (x1, x2)
        } else {
            let (a, b) = x_rotate.split_at(3, rotary_embedding_dim_half);

            (a.to_tensor().into_dyn(), b.to_tensor().into_dyn())
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
            output.permuted(&[0, 2, 1, 3]).to_tensor_in(ctx.pool())
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

#[cfg(test)]
mod tests {
    use crate::{BufferPool, operator::InputList};

    use super::*;
    use rten_tensor::{Tensor, test_util::expect_equal};
    use rten_testing::TestCases;

    #[derive(Debug)]
    struct Case {
        input_data: Tensor<f32>,
        position_ids: Tensor<i32>,
        cos_cache: Tensor<f32>,
        sin_cache: Tensor<f32>,
        expected: Tensor<f32>,
    }

    // Test rotary embedding using a ported version of the test case from
    // https://github.com/microsoft/onnxruntime/blob/e3c34da40639669f3dbb7ae95db0662afbec8cc9/onnxruntime/test/providers/cpu/llm/rotary_embedding_op_test.cc#L509
    #[test]
    fn rotary_embedding_test() {
        let rotary = RotaryEmbedding {
            interleaved: 1,
            num_heads: Some(2),
            rotary_embedding_dim: 0,
        };
        let mut input_data = Tensor::from_vec(vec![
            // Head 0: sequence 0, 1, 2
            -1.0408, 0.9166, -1.3042, -1.1097, // seq 0
            -1.2188, 1.1676, -1.0574, -0.1188, // seq 1
            -0.8110, 0.6737, -1.1233, -0.0919, // seq 2
            // Head 1: sequence 0, 1, 2
            -0.1320, -0.2751, -0.2350, 0.0937, // seq 0
            -0.7396, -1.2425, -0.1752, 0.6990, // seq 1
            -0.6861, 0.7202, 0.1963, 0.6142,
        ]);

        input_data.reshape(&[1, 2, 3, 4]);

        // TODO batch size is mentioned as a dimension of this tensor - maybe something wrong in my
        // impl
        let position_ids = Tensor::from_vec(vec![0, 1, 2]);

        let mut cos_cache = Tensor::from_vec(vec![
            1.0000, 1.0000, 0.5403, 0.9999, -0.4161, 0.9998, -0.9900, 0.9996, -0.6536, 0.9992,
            0.2837, 0.9988, 0.9602, 0.9982, 0.7539, 0.9976,
        ]);
        cos_cache.reshape(&[8, 2]);
        let mut sin_cache = Tensor::from_vec(vec![
            0.0000, 0.0000, 0.8415, 0.0100, 0.9093, 0.0200, 0.1411, 0.0300, -0.7568, 0.0400,
            -0.9589, 0.0500, -0.2794, 0.0600, 0.6570, 0.0699,
        ]);
        sin_cache.reshape(&[8, 2]);
        let mut expected = Tensor::from_vec(vec![
            // Head 0: sequence 0, 1, 2
            -1.0408, 0.9166, -1.3042, -1.1097, // seq 0 (no change)
            -1.6411, -0.3948, -1.0561, -0.1294, // seq 1 (rotated)
            -0.2751, -1.0178, -1.1212, -0.1143, // seq 2 (rotated)
            // Head 1: sequence 0, 1, 2
            -0.1320, -0.2751, -0.2350, 0.0937, // seq 0 (no change)
            0.6460, -1.2937, -0.1822, 0.6972, // seq 1 (rotated)
            -0.3694, -0.9235, 0.1840, 0.6180,
        ]);
        expected.reshape(&[1, 2, 3, 4]); // ?? Should it be same shape as input?
        let cases = [Case {
            input_data,
            position_ids,
            cos_cache,
            sin_cache,
            expected,
        }];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let Case {
                input_data,
                position_ids,
                cos_cache,
                sin_cache,
                expected,
            } = case;

            let mut input_list = InputList::new();
            input_list.push(input_data.view());
            input_list.push(cos_cache.view());
            input_list.push(sin_cache.view());
            input_list.push(position_ids.view());

            let ctx = OpRunContext::new(&pool, &input_list);

            let result = rotary.run(&ctx).unwrap();
            expect_equal(
                &expected.view(),
                &result[0].as_tensor_view().unwrap().view(),
            )
            .unwrap();
        });
    }
}

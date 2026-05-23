use rten_tensor::{AsView, Layout, NdTensorView, Tensor, TensorView};

use crate::{
    buffer_pool::{AutoReturn, BufferPool},
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
    ops::{
        binary_elementwise::{add, mul, sub},
        concat, gather, slice,
    },
};

fn rotary_embedding(
    pool: &BufferPool,
    input: TensorView<f32>,
    cos: TensorView<f32>,
    sin: TensorView<f32>,
    position_ids: Option<NdTensorView<i32, 2>>,
    interleaved: bool,
    num_heads: Option<usize>,
    rotary_embedding_dim: usize,
) -> Result<Tensor, OpError> {
    let reshaped_input = match input.shape() {
        &[batch, seq_len, hidden_size] => {
            let num_heads = num_heads.unwrap_or(0);
            if num_heads == 0 {
                return Err(OpError::InvalidValue(
                    "num_heads must not be 0 for 3 dimensioned input",
                ));
            }
            if hidden_size % num_heads != 0 {
                // The reference implementation also adds "or rank-3 input", but not sure fully
                // what that means in this context so excluded it = after all input is rank-3
                // here.
                //
                // Note without this check this becomes a panic as the resize will fail - maybe
                // acceptable?
                return Err(OpError::InvalidValue(
                    "hidden_size must be divisible by num_heads",
                ));
            }

            let head_size = hidden_size / num_heads;
            input.reshaped([batch, seq_len, num_heads, head_size])
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

    let rotary_embedding_dim = if rotary_embedding_dim == 0 {
        reshaped_input.shape()[3]
    } else {
        rotary_embedding_dim
    };

    let x_rotate = reshaped_input.slice((.., .., .., ..rotary_embedding_dim));
    let x_not_rotate = reshaped_input.slice((.., .., .., rotary_embedding_dim..));

    let rotary_embedding_dim_half = rotary_embedding_dim / 2;

    let (cos_cache, sin_cache) = if let Some(position_ids) = position_ids {
        let cos_subset = gather(pool, cos, 0, position_ids.as_dyn())?.into_cow();
        let sin_subset = gather(pool, sin, 0, position_ids.as_dyn())?.into_cow();
        (cos_subset, sin_subset)
    } else {
        (cos.as_cow(), sin.as_cow())
    };

    if cos_cache.shape().last() != Some(&rotary_embedding_dim_half) {
        return Err(OpError::InvalidValue(
            "Last dimension of cos cache does not match rotary_embedding_dim/2",
        ));
    }

    if sin_cache.shape().last() != Some(&rotary_embedding_dim_half) {
        return Err(OpError::InvalidValue(
            "Last dimension of sin cache does not match rotary_embedding_dim/2",
        ));
    }

    let cos_cache = cos_cache.view().with_new_axis(2);
    let sin_cache = sin_cache.view().with_new_axis(2);

    let (x1, x2) = if interleaved {
        let starts_x1 = Tensor::<i32>::from([0]);
        let starts_x2 = Tensor::<i32>::from([1]);
        let ends = Tensor::<i32>::from([i32::MAX]); // i32::MAX means "to the end"
        let axes = Tensor::<i32>::from([3]);
        let steps = Tensor::<i32>::from([2]);

        let x1 = slice(
            pool,
            x_rotate.as_dyn(),
            &starts_x1.nd_view(),
            &ends.nd_view(),
            Some(&axes.nd_view()),
            Some(&steps.nd_view()),
        )?;

        let x2 = slice(
            pool,
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

    let cos_x1 = mul(pool, cos_cache.view(), x1.as_dyn())?.auto_return(pool);
    let sin_x2 = mul(pool, sin_cache.view(), x2.as_dyn())?.auto_return(pool);
    let real = sub(pool, cos_x1.view(), sin_x2.view())?.auto_return(pool);

    let sin_x1 = mul(pool, sin_cache.view(), x1.as_dyn())?.auto_return(pool);
    let cos_x2 = mul(pool, cos_cache.view(), x2.as_dyn())?.auto_return(pool);
    let imag = add(pool, sin_x1.view(), cos_x2.view())?.auto_return(pool);

    let x_rotate = if interleaved {
        let insert_axis = real.ndim();
        let real = real.view().with_new_axis(insert_axis);
        let imag = imag.view().with_new_axis(insert_axis);

        let mut x_rotate_concat = concat(pool, &[real, imag], -1)?;
        x_rotate_concat.reshape(&x_rotate.shape());
        x_rotate_concat
    } else {
        concat(pool, &[real.view(), imag.view()], -1)?
    }
    .auto_return(pool);

    let output = concat(pool, &[x_rotate.view(), x_not_rotate.as_dyn()], -1)?;

    let output = if input.ndim() == 3 {
        output.into_shape(input.shape())
    } else {
        output.permuted(&[0, 2, 1, 3]).to_tensor_in(pool)
    };
    Ok(output)
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    pub interleaved: bool,
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
        let input = ctx.inputs().require_as(0)?;
        let cos = ctx.inputs().require_as(1)?;
        let sin = ctx.inputs().require_as(2)?;
        let position_ids = ctx.inputs().get_as(3)?;

        let output = rotary_embedding(
            ctx.pool(),
            input,
            cos,
            sin,
            position_ids,
            self.interleaved,
            self.num_heads,
            self.rotary_embedding_dim,
        )?;

        output.into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[derive(Debug)]
pub struct RotaryEmbeddingMicrosoft {
    pub interleaved: bool,
    pub num_heads: Option<usize>,
    pub rotary_embedding_dim: usize,
}

impl Operator for RotaryEmbeddingMicrosoft {
    fn name(&self) -> &str {
        "com.microsoft.RotaryEmbedding"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();

        let input = inputs.require_as(0)?;
        let position_ids: TensorView<i32> = inputs.require_as(1)?;
        let position_ids = match position_ids.ndim() {
            1 => position_ids.with_new_axis(0).nd_view(),
            2 => position_ids.nd_view(),
            _ => {
                return Err(OpError::InvalidValue("position_ids must have 1 or 2 dims"));
            }
        };
        let cos = inputs.require_as(2)?;
        let sin = inputs.require_as(3)?;

        let output = rotary_embedding(
            ctx.pool(),
            input,
            cos,
            sin,
            Some(position_ids),
            self.interleaved,
            self.num_heads,
            self.rotary_embedding_dim,
        )?;

        output.into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{BufferPool, operator::InputList};

    use super::*;
    use rten_tensor::{Tensor, test_util::expect_equal_with_tolerance};
    use rten_testing::TestCases;

    #[derive(Debug)]
    struct Case {
        input_is_4d: bool,
        input_data: Tensor<f32>,
        position_ids: Option<Tensor<i32>>,
        cos_cache: Tensor<f32>,
        sin_cache: Tensor<f32>,
        expected: Tensor<f32>,
        op: RotaryEmbedding,
        batch_size: usize,
        sequence_length: usize,
        max_sequence_length: usize,
        head_size: usize,
    }

    impl Case {
        // Applying the shape information to all the tensors is a bit onerous, so I'm putting them
        // in as 1D and using this method to apply the correct shaping based on the 3-D and 4-D
        // input cases.
        fn shape_inputs(&mut self) {
            if let Some(pids) = self.position_ids.as_mut() {
                pids.reshape(&[self.batch_size, self.sequence_length]);
            }
            if self.input_is_4d {
                self.input_data.reshape(&[
                    self.batch_size,
                    self.op.num_heads.unwrap(),
                    self.sequence_length,
                    self.head_size,
                ]);
                self.expected.reshape(&[
                    self.batch_size,
                    self.op.num_heads.unwrap(),
                    self.sequence_length,
                    self.head_size,
                ]);
            } else {
                let hidden_size = self.head_size * self.op.num_heads.unwrap();
                self.input_data
                    .reshape(&[self.batch_size, self.sequence_length, hidden_size]);
                self.expected
                    .reshape(&[self.batch_size, self.sequence_length, hidden_size]);
            }
            let cache_dim = if self.op.rotary_embedding_dim > 0 {
                self.op.rotary_embedding_dim / 2
            } else {
                self.head_size / 2
            };
            let cache_shape = if self.position_ids.is_some() {
                vec![self.max_sequence_length, cache_dim]
            } else {
                vec![self.batch_size, self.sequence_length, cache_dim]
            };

            self.cos_cache.reshape(&cache_shape);
            self.sin_cache.reshape(&cache_shape);
        }
    }

    // Test rotary embedding using a ported version of the test case from
    // https://github.com/microsoft/onnxruntime/blob/e3c34da40639669f3dbb7ae95db0662afbec8cc9/onnxruntime/test/providers/cpu/llm/rotary_embedding_op_test.cc#L509
    #[test]
    fn test_rotary_embedding() {
        let mut cases = vec![];
        let op = RotaryEmbedding {
            interleaved: true,
            num_heads: Some(2),
            rotary_embedding_dim: 0,
        };
        let input_data = Tensor::from_vec(vec![
            // Head 0: sequence 0, 1, 2
            -1.0408, 0.9166, -1.3042, -1.1097, // seq 0
            -1.2188, 1.1676, -1.0574, -0.1188, // seq 1
            -0.8110, 0.6737, -1.1233, -0.0919, // seq 2
            // Head 1: sequence 0, 1, 2
            -0.1320, -0.2751, -0.2350, 0.0937, // seq 0
            -0.7396, -1.2425, -0.1752, 0.6990, // seq 1
            -0.6861, 0.7202, 0.1963, 0.6142,
        ]);

        let position_ids = Tensor::from_vec(vec![0, 1, 2]);

        let cos_cache = Tensor::from_vec(vec![
            1.0000, 1.0000, 0.5403, 0.9999, -0.4161, 0.9998, -0.9900, 0.9996, -0.6536, 0.9992,
            0.2837, 0.9988, 0.9602, 0.9982, 0.7539, 0.9976,
        ]);
        let sin_cache = Tensor::from_vec(vec![
            0.0000, 0.0000, 0.8415, 0.0100, 0.9093, 0.0200, 0.1411, 0.0300, -0.7568, 0.0400,
            -0.9589, 0.0500, -0.2794, 0.0600, 0.6570, 0.0699,
        ]);
        let expected = Tensor::from_vec(vec![
            // Head 0: sequence 0, 1, 2
            -1.0408, 0.9166, -1.3042, -1.1097, // seq 0 (no change)
            -1.6411, -0.3948, -1.0561, -0.1294, // seq 1 (rotated)
            -0.2751, -1.0178, -1.1212, -0.1143, // seq 2 (rotated)
            // Head 1: sequence 0, 1, 2
            -0.1320, -0.2751, -0.2350, 0.0937, // seq 0 (no change)
            0.6460, -1.2937, -0.1822, 0.6972, // seq 1 (rotated)
            -0.3694, -0.9235, 0.1840, 0.6180,
        ]);
        // Case name: RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_4D_Input
        cases.push(Case {
            input_is_4d: true,
            input_data,
            position_ids: Some(position_ids),
            cos_cache,
            sin_cache,
            expected,
            op,
            batch_size: 1,
            sequence_length: 3,
            max_sequence_length: 8,
            head_size: 4,
        });

        let op = RotaryEmbedding {
            interleaved: false,
            num_heads: Some(3),
            rotary_embedding_dim: 0,
        };
        let input_data = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529, -0.2250, -0.4327,
            -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911, -0.9320, -0.8579, -1.0574, -0.1188,
            -0.9078, 0.3452, -0.5713, -0.2351, -0.8480, 0.5266, -1.2944, -0.0243, -0.2354, -0.7087,
            -0.9647, -0.0991, -0.2994, -0.0650, -1.5720, -1.3211,
        ]);

        let position_ids = Tensor::from_vec(vec![0, 1]);

        let cos_cache = Tensor::from_vec(vec![
            1.0000, 1.0000, 1.0000, 0.5403, 0.9989, 1.0000, -0.4161, 0.9957, 1.0000, -0.9900,
            0.9903, 1.0000,
        ]);
        let sin_cache = Tensor::from_vec(vec![
            0.0000, 0.0000, 0.0000, 0.8415, 0.0464, 0.0022, 0.9093, 0.0927, 0.0043, 0.1411, 0.1388,
            0.0065,
        ]);
        let expected = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529, -0.2250, -0.4327,
            -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911, -0.9320, -0.8579, -0.8618, -0.0922,
            -0.9073, -0.7032, -0.5762, -0.2371, -0.4377, 0.5370, -1.2929, -0.7267, -0.2107,
            -0.7115, -0.4666, -0.0261, -0.2965, -0.8469, -1.5749, -1.3217,
        ]);
        // Case name RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT
        cases.push(Case {
            input_is_4d: false,
            input_data,
            position_ids: Some(position_ids),
            cos_cache,
            sin_cache,
            expected,
            op,
            batch_size: 1,
            sequence_length: 2,
            max_sequence_length: 4,
            head_size: 6,
        });

        let op = RotaryEmbedding {
            interleaved: false,
            num_heads: Some(1),
            rotary_embedding_dim: 4,
        };
        let input_data = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529, -0.2250, -0.4327,
            -1.5071, -0.4586,
        ]);

        let position_ids = Tensor::from_vec(vec![0, 1]);

        let cos_cache = Tensor::from_vec(vec![1.0000, 1.0000, 1.0000, 0.5403]);
        let sin_cache = Tensor::from_vec(vec![0.0000, 0.0000, 0.0000, 0.8415]);
        let expected = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.0427, -0.2250, -0.8673,
            -1.5071, -0.4586,
        ]);
        // Case name: RotaryEmbedding_CustomRotaryDim_SmallData_Phi
        cases.push(Case {
            input_is_4d: false,
            input_data,
            position_ids: Some(position_ids),
            cos_cache,
            sin_cache,
            expected,
            op,
            batch_size: 1,
            sequence_length: 2,
            max_sequence_length: 2,
            head_size: 6,
        });

        let op = RotaryEmbedding {
            interleaved: false,
            num_heads: Some(3),
            rotary_embedding_dim: 0,
        };
        let input_data = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529, -0.2250, -0.4327,
            -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911, -0.9320, -0.8579, -1.0574, -0.1188,
            -0.9078, 0.3452, -0.5713, -0.2351, -0.8480, 0.5266, -1.2944, -0.0243, -0.2354, -0.7087,
            -0.9647, -0.0991, -0.2994, -0.0650, -1.5720, -1.3211,
        ]);

        let cos_cache = Tensor::from_vec(vec![1.0000, 1.0000, 1.0000, 0.5403, 0.9989, 1.0000]);
        let sin_cache = Tensor::from_vec(vec![0.0000, 0.0000, 0.0000, 0.8415, 0.0464, 0.0022]);
        let expected = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529, -0.2250, -0.4327,
            -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911, -0.9320, -0.8579, -0.8618, -0.0922,
            -0.9073, -0.7032, -0.5762, -0.2371, -0.4377, 0.5370, -1.2929, -0.7267, -0.2107,
            -0.7115, -0.4666, -0.0261, -0.2965, -0.8469, -1.5749, -1.3217,
        ]);
        // Case name: RotaryEmbedding_NotInterleaved_NoPosIds_SmallData_LlamaMSFT
        cases.push(Case {
            input_is_4d: false,
            input_data,
            position_ids: None,
            cos_cache,
            sin_cache,
            expected,
            op,
            batch_size: 1,
            sequence_length: 2,
            max_sequence_length: 4,
            head_size: 6,
        });

        let op = RotaryEmbedding {
            interleaved: true,
            num_heads: Some(2),
            rotary_embedding_dim: 0,
        };
        let input_data = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -0.1320, -0.2751, -0.2350, 0.0937, -1.2188, 1.1676,
            -1.0574, -0.1188, -0.7396, -1.2425, -0.1752, 0.6990, -0.8110, 0.6737, -1.1233, -0.0919,
            -0.6861, 0.7202, 0.1963, 0.6142,
        ]);

        let cos_cache = Tensor::from_vec(vec![1.0000, 1.0000, 0.5403, 0.9999, -0.4161, 0.9998]);
        let sin_cache = Tensor::from_vec(vec![0.0000, 0.0000, 0.8415, 0.0100, 0.9093, 0.0200]);
        let expected = Tensor::from_vec(vec![
            -1.0408, 0.9166, -1.3042, -1.1097, -0.1320, -0.2751, -0.2350, 0.0937, -1.6411, -0.3948,
            -1.0561, -0.1294, 0.6460, -1.2937, -0.1822, 0.6972, -0.2751, -1.0178, -1.1212, -0.1143,
            -0.3694, -0.9235, 0.1840, 0.6180,
        ]);
        // Case name: RotaryEmbedding_Interleaved_NoPosIds_SmallData_LlamaMSFT
        cases.push(Case {
            input_is_4d: false,
            input_data,
            position_ids: None,
            cos_cache,
            sin_cache,
            expected,
            op,
            batch_size: 1,
            sequence_length: 3,
            max_sequence_length: 8,
            head_size: 4,
        });

        cases.iter_mut().for_each(|x| x.shape_inputs());

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let Case {
                input_data,
                position_ids,
                cos_cache,
                sin_cache,
                expected,
                op,
                ..
            } = case;

            let mut input_list = InputList::new();
            input_list.push(input_data.view());
            input_list.push(cos_cache.view());
            input_list.push(sin_cache.view());
            if let Some(position_ids) = position_ids.as_ref() {
                input_list.push(position_ids.view());
            }

            let ctx = OpRunContext::new(&pool, &input_list);

            let result = op.run(&ctx).unwrap();
            expect_equal_with_tolerance(
                &expected.view(),
                &result[0].as_tensor_view().unwrap().view(),
                1e-4,
                0.0,
            )
            .unwrap();
        });
    }

    #[test]
    fn reject_indivisible_hidden_size() {
        let op = RotaryEmbedding {
            interleaved: false,
            num_heads: Some(2),
            rotary_embedding_dim: 0,
        };

        let input_data = Tensor::from([[0., 0., 0., 0., 0.]]).with_new_axis(0);
        let cos_cache = Tensor::from([[[1.0]]]);
        let sin_cache = Tensor::from([[[0.0]]]);
        let mut input_list = InputList::new();
        input_list.push(input_data.view());
        input_list.push(cos_cache.view());
        input_list.push(sin_cache.view());

        let pool = BufferPool::new();
        let ctx = OpRunContext::new(&pool, &input_list);

        assert!(op.run(&ctx).is_err());
    }
}

use rten_tensor::{AsView, Layout, NdTensorView, TensorView};

use crate::operator::{
    OpError, OpRunContext, Operator, OutputList, OutputTypeList, OutputTypesContext,
};
use crate::value::Value;

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

        let input_shape = input.shape();
        let reshaped_input = match input_shape.len() {
            3 => {
                let hidden_size = input_shape[1];
                if matches!(self.num_heads, Some(0) | None) {
                    return Err(OpError::InvalidValue(
                        "num_heads must not be 0 for 3 dimensioned input",
                    ));
                }

                let head_size = hidden_size / self.num_heads.unwrap();
                let tmp = input.reshaped([
                    input_shape[0],
                    input_shape[1],
                    self.num_heads.unwrap(),
                    head_size,
                ]);

                tmp.into_dyn()
            }
            4 => {
                let tmp = input.permuted(&[0, 2, 1, 3]).to_tensor().into_dyn();

                tmp.into_cow()
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

        if let Some(position_ids) = position_ids {
        } else {
        }

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

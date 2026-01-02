//! Operator which computes a tensor shape from static and dynamic values.

use rten_tensor::Tensor;
use rten_tensor::prelude::*;

use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, ValueType};

/// Specifies the source for a tensor shape computed by [`ComputeShape`].
#[derive(Debug, Clone, PartialEq)]
pub enum DimSpec {
    /// Output a fixed value.
    Static(u32),
    /// Copy the size of a tensor dimension from an input.
    Dynamic { input: u32, dim: u32 },
}

/// Compute a tensor shape from a combination of static values and the dynamic
/// shapes of inputs.
///
/// This is a custom internal operator produced by fusions.
#[derive(Debug)]
pub struct ComputeShape {
    /// Specifies the length of the output vector and how to compute each element.
    pub shape: Vec<DimSpec>,
}

impl Operator for ComputeShape {
    fn name(&self) -> &str {
        "ComputeShape"
    }

    fn max_inputs(&self) -> Option<usize> {
        None
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();

        let output = self
            .shape
            .iter()
            .map(|dim| match dim {
                DimSpec::Static(size) => Ok(*size as i32),
                DimSpec::Dynamic {
                    input: input_idx,
                    dim,
                } => {
                    let dim = *dim as usize;
                    let input = inputs.require(*input_idx as usize)?;
                    if input.ndim() > dim {
                        Ok(input.size(dim).min(i32::MAX as usize) as i32)
                    } else {
                        Err(OpError::InvalidValue(
                            "Dim index invalid for input tensor shape",
                        ))
                    }
                }
            })
            .collect::<Result<Vec<i32>, _>>()?;

        Tensor::from(output).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Int32))].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, NdTensorView};

    use super::{ComputeShape, DimSpec};
    use crate::operator::{OpError, OperatorExt};

    #[test]
    fn test_compute_shape() {
        // Valid input with a mix of static and dynamic output values.
        let input_a = NdTensor::<f32, _>::zeros([2, 4, 8]);
        let input_b = NdTensor::<f32, _>::zeros([24]);

        let op = ComputeShape {
            shape: [
                DimSpec::Static(3),
                DimSpec::Dynamic { input: 0, dim: 1 },
                DimSpec::Static(5),
                DimSpec::Dynamic { input: 1, dim: 0 },
            ]
            .into(),
        };
        let result: NdTensor<i32, 1> = op.run_simple((input_a.view(), input_b.view())).unwrap();

        assert_eq!(result, NdTensorView::from(&[3, 4, 5, 24]));

        // Dynamic input with invalid input index.
        let op = ComputeShape {
            shape: [DimSpec::Dynamic { input: 1, dim: 0 }].into(),
        };
        let result: Result<NdTensor<i32, 1>, _> = op.run_simple(input_a.view());
        assert_eq!(result.err().unwrap(), OpError::MissingInputs);

        // Dynamic input with invalid dim index.
        let op = ComputeShape {
            shape: [DimSpec::Dynamic { input: 0, dim: 3 }].into(),
        };
        let result: Result<NdTensor<i32, 1>, _> = op.run_simple(input_a.view());
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("Dim index invalid for input tensor shape")
        );
    }
}

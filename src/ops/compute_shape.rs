//! Operator which computes a tensor shape by evaluating symbolic expressions.

use rten_shape_inference::{SymExpr, SymbolMap};
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, ValueType};

/// Specifies the source for a variable used in symbolic expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolInfo {
    /// Name of the symbol in symbolic expressions.
    pub name: String,

    /// Input index.
    pub input: u32,

    /// Axis index.
    pub axis: u32,
}

/// Compute a tensor shape from a combination of static values and the dynamic
/// shapes of inputs.
#[derive(Debug)]
pub struct ComputeShape {
    /// Specifies how to map dimension sizes of inputs to symbols used by the
    /// `shape` field.
    pub symbols: Vec<SymbolInfo>,

    /// Specifies the symbolic expression to evaluate for each position in the
    /// output vector.
    pub shape: Vec<SymExpr>,
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

        let symbols = self
            .symbols
            .iter()
            .map(|sym| {
                let input = inputs.require(sym.input as usize)?;
                if input.ndim() <= sym.axis as usize {
                    return Err(OpError::InvalidValue("Axis invalid for input shape"));
                }
                let size = input.size(sym.axis as usize) as i32;
                Ok((sym.name.as_str(), size))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let symbols = SymbolMap::new(&symbols);

        let output = self
            .shape
            .iter()
            .map(|expr| {
                expr.eval(&symbols)
                    .map_err(|_| OpError::InvalidValue("Failed to evaluate symbolic shape"))
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
    use rten_shape_inference::{SymExpr, Symbol};
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, NdTensorView};

    use super::{ComputeShape, SymbolInfo};
    use crate::operator::{OpError, OperatorExt};

    #[test]
    fn test_compute_shape() {
        // Valid input with a mix of static and dynamic output values.
        let input_a = NdTensor::<f32, _>::zeros([2, 4, 8]);
        let input_b = NdTensor::<f32, _>::zeros([24]);

        let op = ComputeShape {
            symbols: [
                SymbolInfo {
                    name: "x".to_string(),
                    input: 0,
                    axis: 1,
                },
                SymbolInfo {
                    name: "y".to_string(),
                    input: 1,
                    axis: 0,
                },
            ]
            .to_vec(),
            shape: [
                SymExpr::Value(3),
                SymExpr::Var(
                    Symbol {
                        name: "x".to_string(),
                        positive: true,
                    }
                    .into(),
                ),
                SymExpr::Value(5),
                SymExpr::Var(
                    Symbol {
                        name: "y".to_string(),
                        positive: true,
                    }
                    .into(),
                ),
            ]
            .into(),
        };
        let result: NdTensor<i32, 1> = op.run_simple((input_a.view(), input_b.view())).unwrap();

        assert_eq!(result, NdTensorView::from(&[3, 4, 5, 24]));

        // Dynamic input with invalid input index.
        let op = ComputeShape {
            symbols: [SymbolInfo {
                name: "x".to_string(),
                input: 1,
                axis: 0,
            }]
            .into(),
            shape: Vec::new(),
        };
        let result: Result<NdTensor<i32, 1>, _> = op.run_simple(input_a.view());
        assert_eq!(result.err().unwrap(), OpError::MissingInputs);

        // Dynamic input with invalid axis.
        let op = ComputeShape {
            symbols: [SymbolInfo {
                name: "x".to_string(),
                input: 0,
                axis: 3,
            }]
            .into(),
            shape: Vec::new(),
        };
        let result: Result<NdTensor<i32, 1>, _> = op.run_simple(input_a.view());
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("Axis invalid for input shape")
        );
    }
}

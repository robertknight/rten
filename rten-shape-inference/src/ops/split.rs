use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::ops::resolve_axis;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Split operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Split.html>.
pub struct Split {
    /// Axis to split the tensor along.
    pub axis: i32,

    /// Number of pieces to split the tensor into, if split sizes are not
    /// explicitly provided as an input.
    pub num_outputs: Option<u32>,
}

impl InferShapes for Split {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, rest @ ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(splits) = rest.first() else {
            return Err(InferShapesError::UnknownOutputCount);
        };

        // This currently supports only the case where split sizes are
        // explicitly specified. Otherwise the splits should be determined using
        // the `num_outputs` attribute. If that is not set, this needs to be
        // determined from the number of output values this operator has. That
        // information is not currently exposed to the `infer_shapes` method.
        let Some(split_sizes) = splits.as_vector() else {
            return Err(InferShapesError::UnknownOutputCount);
        };

        let outputs: Result<Vec<SymTensor>, _> = split_sizes
            .iter()
            .map(|size| {
                if let Some(shape) = data.shape() {
                    let axis = resolve_axis(shape.len(), self.axis)?;
                    let mut shape: Vec<_> = shape.collect();
                    shape[axis] = size.clone();
                    Ok(SymTensor::from_shape(shape))
                } else {
                    Ok(SymTensor::unknown("unknown input shape"))
                }
            })
            .collect();

        outputs
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::Split;

    #[test]
    fn test_split() {
        // Split with explicitly specified split sizes.
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", "seq", 2304);
        let splits = sym_vec!(768, 768, 768);
        let op = Split {
            axis: 2,
            num_outputs: None,
        };
        let result = op.infer_shapes(&[data, splits], &mut sym_gen).unwrap();
        assert_eq!(
            result,
            [
                sym_shape!("batch", "seq", 768),
                sym_shape!("batch", "seq", 768),
                sym_shape!("batch", "seq", 768),
            ]
        );
    }
}

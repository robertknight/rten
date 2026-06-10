use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Resize operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Resize.html>.
pub struct Resize;

// Input positions defined by the ONNX Resize spec.
const DATA: usize = 0;
const SCALES: usize = 2;
const SIZES: usize = 3;

impl InferShapes for Resize {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs
            .get(DATA)
            .ok_or(InferShapesError::IncorrectInputCount)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        // `sizes` takes precedence if both are provided.
        let sizes = optional_input(inputs, SIZES);
        let scales = optional_input(inputs, SCALES);

        let out_shape: Vec<SymExpr> = if let Some(sizes) = sizes {
            if let Some(values) = sizes.as_vector() {
                values.to_vec()
            } else {
                sym_gen.gen_shape(data_dims.len())
            }
        } else if let Some(scales) = scales {
            if let Some(values) = scales.as_vector() {
                data_dims
                    .zip(values.iter())
                    .map(|(in_dim, scale)| in_dim * scale.clone())
                    .collect()
            } else {
                sym_gen.gen_shape(data_dims.len())
            }
        } else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Upsample operator.
///
/// This is a deprecated operator that has been replaced by [`Resize`]. See
/// <https://onnx.ai/onnx/operators/onnx__Upsample.html>.
pub struct Upsample;

// Input position of the `scales` input. The data input is at the same position
// (`DATA`) as for `Resize`.
const UPSAMPLE_SCALES: usize = 1;

impl InferShapes for Upsample {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs
            .get(DATA)
            .ok_or(InferShapesError::IncorrectInputCount)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let input_scales = optional_input(inputs, UPSAMPLE_SCALES).and_then(|s| s.as_vector());

        let out_shape: Vec<SymExpr> = if let Some(scales) = input_scales {
            data_dims
                .zip(scales.iter())
                .map(|(in_dim, scale)| in_dim * scale.clone())
                .collect()
        } else {
            sym_gen.gen_shape(data_dims.len())
        };

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Get an optional input.
///
/// As a special case this treats an empty vector as missing. In opset < 13, the
/// ONNX Resize op uses an empty vector to represent missing `sizes`/`scales`
/// inputs.
fn optional_input(inputs: &[SymTensor], idx: usize) -> Option<&SymTensor> {
    let input = inputs.get(idx)?;
    let is_empty_vec = input.as_vector().is_some_and(|v| v.is_empty());
    if is_empty_vec {
        return None;
    }
    Some(input)
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{Resize, Upsample};

    #[test]
    fn test_resize_sizes() {
        let data = sym_shape!(1, 3, 224, 224);
        let sizes = sym_vec!(1, 3, 448, 448);

        let mut sym_gen = SymbolGen::new();
        let result = Resize
            .infer_shapes(&[data, sym_vec!(), sym_vec!(), sizes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(1, 3, 448, 448));
    }

    #[test]
    fn test_resize_scales() {
        let data = sym_shape!("batch", 3, 224, 224);
        let scales = sym_vec!(1, 1, 2, 2);

        let mut sym_gen = SymbolGen::new();
        let result = Resize
            .infer_shapes(&[data, sym_vec!(), scales, sym_vec!()], &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("batch", 3, 448, 448)
        );
    }

    #[test]
    fn test_resize_unknown_input() {
        let data = SymTensor::unknown("unknown");
        let scales = sym_vec!(1, 1, 2, 2);

        let mut sym_gen = SymbolGen::new();
        let result = Resize
            .infer_shapes(&[data, sym_vec!(), scales, sym_vec!()], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_resize_unknown_scales() {
        let data = sym_shape!(1, 3, 224, 224);
        let scales = SymTensor::unknown("unknown");

        let mut sym_gen = SymbolGen::new();
        let result = Resize
            .infer_shapes(&[data, sym_vec!(), scales, sym_vec!()], &mut sym_gen)
            .unwrap();

        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 4);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }
    }

    #[test]
    fn test_resize_unknown_sizes() {
        let data = sym_shape!(1, 3, 224, 224);
        let sizes = SymTensor::unknown("unknown");

        let mut sym_gen = SymbolGen::new();
        let result = Resize
            .infer_shapes(&[data, sym_vec!(), sym_vec!(), sizes], &mut sym_gen)
            .unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 4);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }
    }

    #[test]
    fn test_resize_missing_scales_and_sizes() {
        let data = sym_shape!(1, 3, 224, 224);

        let mut sym_gen = SymbolGen::new();
        let err = Resize
            .infer_shapes(&[data, sym_vec!(), sym_vec!(), sym_vec!()], &mut sym_gen)
            .err()
            .unwrap();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }

    #[test]
    fn test_upsample() {
        let data = sym_shape!("batch", 3, 224, 224);
        let scales = sym_vec!(1, 1, 2, 2);

        let mut sym_gen = SymbolGen::new();
        let result = Upsample
            .infer_shapes(&[data, scales], &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("batch", 3, 448, 448)
        );
    }

    #[test]
    fn test_upsample_unknown_scales() {
        let data = sym_shape!(1, 3, 224, 224);
        let scales = SymTensor::unknown("unknown");

        let mut sym_gen = SymbolGen::new();
        let result = Upsample
            .infer_shapes(&[data, scales], &mut sym_gen)
            .unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 4);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }
    }
}

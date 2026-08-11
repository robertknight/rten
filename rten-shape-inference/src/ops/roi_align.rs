use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// RoiAlign operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__RoiAlign.html>.
pub struct RoiAlign {
    pub output_height: usize,
    pub output_width: usize,
}

impl InferShapes for RoiAlign {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let rois = inputs.require(1)?;
        inputs.require(2)?;

        if data.ndim().is_some_and(|ndim| ndim != 4) || rois.ndim().is_some_and(|ndim| ndim != 2) {
            return Err(InferShapesError::IncorrectRank);
        }

        let out_h: i32 = self
            .output_height
            .try_into()
            .map_err(|_| InferShapesError::InvalidValue)?;
        let out_w: i32 = self
            .output_width
            .try_into()
            .map_err(|_| InferShapesError::InvalidValue)?;

        // Output is `(num_rois, channels, output_height, output_width)`.
        let (Some(num_rois), Some(chans)) = (rois.size(0), data.size(1)) else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let out_shape = vec![
            num_rois,
            chans,
            SymExpr::Value(out_h),
            SymExpr::Value(out_w),
        ];

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::RoiAlign;

    #[test]
    fn test_roi_align() {
        let mut sym_gen = SymbolGen::new();
        let op = RoiAlign {
            output_height: 7,
            output_width: 5,
        };

        // Output is `(num_rois, channels, output_height, output_width)`.
        let data = sym_shape!(1, 256, 64, 64);
        let rois = sym_shape!("num_rois", 4);
        let batch_indices = sym_shape!("num_rois");
        let result = op
            .infer_shapes([data, rois, batch_indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("num_rois", 256, 7, 5));

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let rois = sym_shape!(3, 4);
        let batch_indices = sym_shape!(3);
        let result = op
            .infer_shapes([data, rois, batch_indices].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);

        // Wrong rank.
        let data = sym_shape!(1, 256, 64);
        let rois = sym_shape!(3, 4);
        let batch_indices = sym_shape!(3);
        let err = op
            .infer_shapes([data, rois, batch_indices].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectRank);

        // Missing input.
        let data = sym_shape!(1, 256, 64, 64);
        let rois = sym_shape!(3, 4);
        let err = op
            .infer_shapes([data, rois].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }
}

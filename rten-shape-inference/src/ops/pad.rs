use rten_tensor::Layout;

use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Pad operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Pad.html>.
pub struct Pad;

// Input positions defined by the ONNX Pad spec.
const DATA: usize = 0;
const PADS: usize = 1;
const AXES: usize = 3;

impl InferShapes for Pad {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(DATA)?;
        let pads = inputs.require(PADS)?;
        let axes = inputs.get(AXES);

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let ndim = data_dims.len();

        // Resolve the set of axes that are padded. If `axes` is not provided,
        // all dimensions are padded.
        let resolved_axes: Vec<usize> = if let Some(axes) = axes {
            match axes.to_constant() {
                Some(axes) if axes.ndim() == 1 => axes
                    .into_data()
                    .into_iter()
                    .map(|axis| resolve_axis(ndim, axis))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => {
                    // `axes` is symbolic, so we can't determine which dims are
                    // padded. The output rank is preserved but the sizes are
                    // unknown.
                    let out_dims = sym_gen.gen_shape(ndim);
                    return Ok([SymTensor::from_shape(out_dims)].into());
                }
            }
        } else {
            (0..ndim).collect()
        };

        let mut out_dims: Vec<SymExpr> = data_dims.collect();

        if let Some(pad_values) = pads.as_vector() {
            let n_padded = resolved_axes.len();
            if pad_values.len() != 2 * n_padded {
                return Err(InferShapesError::InvalidValue);
            }
            for (i, axis) in resolved_axes.iter().enumerate() {
                let start = pad_values[i].clone();
                let end = pad_values[n_padded + i].clone();
                out_dims[*axis] = out_dims[*axis].clone() + start + end;
            }
        } else {
            // `pads` is not a known vector. We can preserve the rank but
            // can't determine the sizes of padded dimensions.
            for axis in resolved_axes {
                out_dims[axis] = sym_gen.gen_positive();
            }
        }

        Ok([SymTensor::from_shape(out_dims)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::Pad;

    #[test]
    fn test_pad_fixed_pads() {
        let mut sym_gen = SymbolGen::new();

        // Pad a fully-fixed shape with fixed pads.
        let data = sym_shape!(2, 3);
        let pads = sym_vec!(1, 2, 3, 4);
        let result = Pad.infer_shapes([data, pads].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(6, 9));

        // Zero pads (no-op).
        let data = sym_shape!(2, 3);
        let pads = sym_vec!(0, 0, 0, 0);
        let result = Pad.infer_shapes([data, pads].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(2, 3));
    }

    #[test]
    fn test_pad_symbolic_input_shape() {
        let mut sym_gen = SymbolGen::new();

        // Pad symbolic dims with fixed pads.
        let data = sym_shape!("batch", "seq", 64);
        let pads = sym_vec!(0, 1, 2, 0, 3, 4);
        let result = Pad.infer_shapes([data, pads].into(), &mut sym_gen).unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!("batch", SymExpr::from("seq") + SymExpr::from(4), 70)
        );
    }

    #[test]
    fn test_pad_with_axes() {
        let mut sym_gen = SymbolGen::new();

        // Pad only specified axes.
        let data = sym_shape!(2, 3, 4);
        let pads = sym_vec!(1, 2, 3, 4);
        let const_val = SymTensor::unknown("unknown value");
        let axes = sym_vec!(0, 2);
        let result = Pad
            .infer_shapes([data, pads, const_val, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(6, 3, 10));

        // Negative axis.
        let data = sym_shape!(2, 3, 4);
        let pads = sym_vec!(1, 2);
        let const_val = SymTensor::unknown("unknown value");
        let axes = sym_vec!(-1);
        let result = Pad
            .infer_shapes([data, pads, const_val, axes].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(2, 3, 7));
    }

    #[test]
    fn test_pad_unknown_input_shape() {
        let mut sym_gen = SymbolGen::new();
        let data = SymTensor::unknown("unknown");
        let pads = sym_vec!(1, 1, 1, 1);
        let result = Pad.infer_shapes([data, pads].into(), &mut sym_gen).unwrap();
        assert_eq!(result[0].ndim(), None);
    }

    #[test]
    fn test_pad_unknown_pads() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", 16, 32);
        let pads = SymTensor::unknown("unknown");
        let result = Pad.infer_shapes([data, pads].into(), &mut sym_gen).unwrap();

        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 3);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }
    }

    #[test]
    fn test_pad_symbolic_axes() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!(2, 3, 4);
        let pads = sym_vec!(1, 1);
        let const_val = SymTensor::from_scalar(0.into());
        let axes = SymTensor::unknown("unknown");
        let result = Pad
            .infer_shapes([data, pads, const_val, axes].into(), &mut sym_gen)
            .unwrap();

        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 3);
        for dim in &shape {
            assert!(matches!(dim, SymExpr::Var(_)));
        }
    }

    #[test]
    fn test_pad_invalid_pads_length() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!(2, 3);
        let pads = sym_vec!(1, 1, 1);
        let err = Pad
            .infer_shapes([data, pads].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }

    #[test]
    fn test_pad_missing_inputs() {
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!(2, 3);
        let err = Pad.infer_shapes([data].into(), &mut sym_gen).unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectInputCount);

        let err = Pad
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }
}

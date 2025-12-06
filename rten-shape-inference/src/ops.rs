//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{InferShapes, InferShapesError, resolve_axis};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{SymElem, SymTensor};

/// Concat operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Concat.html>.
pub struct Concat {
    pub axis: i32,
}

impl InferShapes for Concat {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [first, rest @ ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(first_dims) = first.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let axis = resolve_axis(first_dims.len(), self.axis)
            .map_err(|_| InferShapesError::IncorrectRank)?;

        // If input is a constant or symbolic vector, return a constant or
        // symbolic vector by concatenating each input.
        if axis == 0 && inputs.iter().all(|inp| inp.values().is_some()) {
            let value = {
                let mut values = Vec::new();
                for inp in inputs {
                    values.extend(inp.values().expect("should have values").to_vec());
                }
                SymTensor::from_vec(values)
            };
            return Ok([value].into());
        }

        let mut out_shape: Vec<_> = first_dims.collect();

        for input in rest {
            if let Some(dim) = input.shape().and_then(|mut dims| dims.nth(axis)) {
                out_shape[axis] += dim;
            } else {
                out_shape[axis] += sym_gen.gen_positive();
            }
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Unsqueeze operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>.
pub struct Unsqueeze;

impl InferShapes for Unsqueeze {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, axes] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        // If data is a constant or symbolic scalar and axes is 0, the output
        // will be a length-1 vector. We can't handle higher-rank symbolic
        // values yet.
        //
        // Otherwise if the input shape is known and `axes` is a constant, the
        // output is a symbolic shape.
        //
        // Otherwise the output is unknown.

        let axes_vec = axes.to_constant().map(|c| c.into_vec());

        let value = if let Some(var) = data.as_scalar()
            && axes_vec.as_deref().map(|v| v == [0]).unwrap_or(false)
        {
            SymTensor::from_vec([var.clone()].into())
        } else if let Some(dims) = data.shape()
            && let Some(mut axes) = axes_vec
        {
            let mut dims: Vec<_> = dims.collect();
            axes.sort();

            for (i, axis) in axes.into_iter().enumerate() {
                let axis = resolve_axis(dims.len() + 1, axis)
                    .map_err(|_| InferShapesError::IncorrectRank)?;
                dims.insert(axis + i, SymElem::Value(1));
            }

            SymTensor::from_shape(dims)
        } else {
            SymTensor::unknown("unknown data shape or axes value")
        };

        Ok([value].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_elems, sym_shape, sym_vec};

    use super::{Concat, Unsqueeze};

    fn extract_shape(mut result: Vec<SymTensor>) -> Vec<SymElem> {
        result.remove(0).shape().unwrap().collect()
    }

    #[test]
    fn test_concat() {
        // Concatenation of fixed dims.
        let a = sym_shape!("batch", 16, 64);
        let b = sym_shape!("batch", 16, 64);

        let mut sym_gen = SymbolGen::new();
        let op = Concat { axis: 1 };
        let result = op.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymElem::from(16) + SymElem::from(16), 64)
        );

        // Concatenation of symbolic dims.
        let a = sym_shape!("batch", "foo", 64);
        let b = sym_shape!("batch", "bar", 64);

        let op = Concat { axis: 1 };
        let result = op.infer_shapes(&[a, b], &mut sym_gen).unwrap();
        let shape = extract_shape(result);
        assert_eq!(
            shape,
            sym_elems!("batch", SymElem::from("foo") + SymElem::from("bar"), 64)
        );

        // Concatenation of symbolic vectors.
        let bc_dims = sym_vec!("batch", "chans");
        let hw_dims = sym_vec!("height", "width");
        let op = Concat { axis: 0 };
        let mut result = op.infer_shapes(&[bc_dims, hw_dims], &mut sym_gen).unwrap();
        assert_eq!(
            result.remove(0).as_vector().unwrap(),
            sym_elems!("batch", "chans", "height", "width")
        );
    }

    #[test]
    fn test_unsqueeze() {
        let mut sym_gen = SymbolGen::new();

        // Unsqueeze into an ND-tensor.
        let shape = sym_shape!("batch", 16, 64);
        let axes = sym_vec!(1);
        let expected = sym_shape!("batch", 1, 16, 64);

        let result = Unsqueeze
            .infer_shapes(&[shape, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);

        // Unsqueeze a symbolic scalar into a symbolic vec.
        let scalar = SymTensor::from_scalar(1.into());
        let axes = sym_vec!(0);
        let expected = sym_vec!(1);
        let result = Unsqueeze
            .infer_shapes(&[scalar, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], expected);
    }
}

use rten_tensor::SliceRange;

use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::ops::resolve_axis;
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{Constant, SymTensor};

/// Slice operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Slice.html>.
pub struct Slice;

impl InferShapes for Slice {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, starts, ends, rest @ ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        let axes = rest
            .first()
            .map(|axes| axes.to_constant())
            .unwrap_or_else(|| {
                let axes = (0..data_dims.len()).map(|i| i as i32).collect();
                Some(Constant::Vector(axes))
            });

        let steps = rest.get(1);

        let sliced_shape = if let Some(axes) = axes {
            let mut dims: Vec<_> = data_dims.collect();

            let starts = starts.as_vector();
            let ends = ends.as_vector();
            let steps = steps.and_then(|s| s.as_vector());

            for (i, axis) in axes.values().iter().copied().enumerate() {
                let axis =
                    resolve_axis(dims.len(), axis).map_err(|_| InferShapesError::IncorrectRank)?;

                let start = starts.and_then(|s| s.get(i));
                let end = ends.and_then(|e| e.get(i));
                let step = steps.and_then(|s| s.get(i)).unwrap_or(&SymExpr::Value(1));

                if let Some(SymExpr::Value(start)) = start
                    && let Some(SymExpr::Value(end)) = end
                    && let SymExpr::Value(step) = step
                    && let SymExpr::Value(size) = dims[axis]
                {
                    let end = match *end {
                        i32::MAX => None,
                        end => Some(end as isize),
                    };

                    let range = SliceRange::new(*start as isize, end, *step as isize);

                    // When slicing a symbolic vec along axis 0, the result can
                    // also be a symbolic vec.
                    if let Some(vals) = data.values()
                        && *step == 1
                    {
                        let clamped_range = range.resolve_clamped(size as usize);
                        return Ok([SymTensor::from_vec(vals[clamped_range].to_vec())].into());
                    }

                    let slice_size = range.steps(size as usize) as i32;
                    dims[axis] = SymExpr::Value(slice_size);
                } else if let Some(start) = start
                    && (start == &SymExpr::Value(0) || *start == -dims[axis].clone())
                    && let Some(end) = end
                    && (end == &SymExpr::Value(i32::MAX) || *end == dims[axis])
                    && step == &SymExpr::Value(1)
                {
                    // This is a no-op slice that doesn't alter the dimension
                    // size.
                } else if let Some(start) = start
                    && start.is_positive()
                    && let Some(end) = end
                    && end.is_positive()
                    && step == &SymExpr::Value(1)
                {
                    // nb. This assumes start <= end.
                    let size = dims[axis].clone();
                    dims[axis] = end.min(&size) - start.min(&size);
                } else {
                    dims[axis] = sym_gen.gen_positive();
                }
            }

            SymTensor::from_shape(dims)
        } else {
            let shape = (0..data_dims.len())
                .map(|_| sym_gen.gen_positive())
                .collect();
            SymTensor::from_shape(shape)
        };

        Ok([sliced_shape].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::Slice;

    #[test]
    fn test_slice() {
        let mut sym_gen = SymbolGen::new();

        // Slice of a fixed-sized dimension with fixed starts, ends and axes.
        let data = sym_shape!("batch", 64, 8);
        let starts = sym_vec!(32);
        let ends = sym_vec!(i32::MAX);
        let axes = sym_vec!(1);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 32, 8));

        // Slice of a symbolic dimension with unknown values.
        //
        // In this case an expression for the output size would be something
        // like `min(batch, end) - min(batch, start)`. We can't express that
        // yet, so a new unknown dimension is created.
        let batch = SymExpr::from("batch");
        let start = SymExpr::from("start");
        let end = SymExpr::from("end");

        let data = sym_shape!(batch.clone(), 64, 8);
        let starts = sym_vec!(start.clone());
        let ends = sym_vec!(end.clone());
        let axes = sym_vec!(0);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes], &mut sym_gen)
            .unwrap();
        let batch_expr = end.min(&batch) - start.min(&batch);
        assert_eq!(result[0], sym_shape!(batch_expr, 64, 8));

        // Slice of a symbolic dimension with a 0..i32::MAX range.
        //
        // This is a no-op that doesn't alter the dimension size.
        let data = sym_shape!("batch", 64, 8);
        let starts = sym_vec!(0);
        let ends = sym_vec!(i32::MAX);
        let axes = sym_vec!(0);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 64, 8));

        // Slice where axes are symbolic.
        //
        // In this case all we can infer is that the output will have a rank
        // equal to the input.
        let mut sym_gen = SymbolGen::new();
        let data = sym_shape!("batch", 64, 8);
        let starts = sym_vec!(0);
        let ends = sym_vec!(i32::MAX);
        let axes = sym_vec!("axes");
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", "unknown_2", "unknown_3"));

        // Slice with unknown input produces unknown output.
        let mut sym_gen = SymbolGen::new();
        let data = SymTensor::unknown("unknown");
        let starts = sym_vec!(0);
        let ends = sym_vec!(0);
        let result = Slice
            .infer_shapes(&[data, starts, ends], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], SymTensor::unknown("unknown input shape"));

        // Slice of a symbolic vector along axis 0.
        let mut sym_gen = SymbolGen::new();
        let data = sym_vec!("batch", 3, "height", "width");
        let starts = sym_vec!(1);
        let ends = sym_vec!(i32::MAX);
        let result = Slice
            .infer_shapes(&[data, starts, ends], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_vec!(3, "height", "width"));

        // Slice of a symbolic vector with negative starts and ends.
        let data = sym_vec!("s6", 6, 400, 64);
        let starts = sym_vec!(i32::MIN);
        let ends = sym_vec!(-2);
        let result = Slice
            .infer_shapes(&[data, starts, ends], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_vec!("s6", 6));

        // Slice from 0..size, where "size" is the dimension size.
        let data = sym_shape!("batch", 1, "seq");
        let starts = sym_vec!(0);
        let ends = sym_vec!("seq");
        let axes = sym_vec!(2);
        let steps = sym_vec!(1);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes, steps], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 1, "seq"));

        // Slice from -size..size, where "size" is the dimension size.
        let data = sym_shape!("batch", 1, "seq");
        let starts = sym_vec!(-SymExpr::from("seq"));
        let ends = sym_vec!("seq");
        let axes = sym_vec!(2);
        let steps = sym_vec!(1);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes, steps], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 1, "seq"));
    }
}

use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::ops::resolve_axis;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{Constant, SymElem, SymTensor};

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
            .get(0)
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
                let axis = resolve_axis(dims.len(), axis as i32)
                    .map_err(|_| InferShapesError::IncorrectRank)?;

                let start = starts.and_then(|s| s.get(i));
                let end = ends.and_then(|e| e.get(i));
                let step = steps.and_then(|s| s.get(i)).unwrap_or(&SymElem::Value(1));

                if let Some(SymElem::Value(start)) = start
                    && let Some(SymElem::Value(end)) = end
                    && let SymElem::Value(step) = step
                    && let SymElem::Value(size) = dims[axis]
                {
                    let end = match *end {
                        i32::MAX => None,
                        end => Some(end as isize),
                    };

                    let slice_size =
                        calculate_slice_size(*start as isize, end, *step as isize, size as usize);
                    dims[axis] = SymElem::Value(slice_size);
                } else if start == Some(&SymElem::Value(0))
                    && let Some(end) = end
                    && end == &SymElem::Value(i32::MAX)
                    && step == &SymElem::Value(1)
                {
                    // This is a no-op slice that doesn't alter the dimension
                    // size.
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

/// Calculate the size of a dimension of size `dim_size` after slicing it with
/// a range that has the given start, end and step values.
///
/// This code is copied from the `SliceRange` type in the rten-tensor crate to
/// avoid a dependency. If a dependency on rten-tensor is added at some point,
/// this can be simplified.
fn calculate_slice_size(start: isize, end: Option<isize>, step: isize, dim_size: usize) -> i32 {
    struct SliceRange {
        pub start: isize,
        pub end: Option<isize>,
        step: isize,
    }

    impl SliceRange {
        fn new(start: isize, end: Option<isize>, step: isize) -> SliceRange {
            assert!(step != 0, "Slice step cannot be 0");
            SliceRange { start, end, step }
        }

        fn steps(&self, dim_size: usize) -> usize {
            let clamped = self.clamp(dim_size);

            let start_idx = Self::offset_from_start(clamped.start, dim_size);
            let end_idx = clamped
                .end
                .map(|index| Self::offset_from_start(index, dim_size))
                .unwrap_or(if self.step > 0 { dim_size as isize } else { -1 });

            if (clamped.step > 0 && end_idx <= start_idx)
                || (clamped.step < 0 && end_idx >= start_idx)
            {
                return 0;
            }

            let steps = if clamped.step > 0 {
                1 + (end_idx - start_idx - 1) / clamped.step
            } else {
                1 + (start_idx - end_idx - 1) / -clamped.step
            };

            steps.max(0) as usize
        }

        fn clamp(&self, dim_size: usize) -> SliceRange {
            let len = dim_size as isize;

            let min_idx;
            let max_idx;

            if self.step > 0 {
                // When traversing forwards, the range of valid +ve indexes is `[0,
                // len]` and for -ve indexes `[-len, -1]`.
                min_idx = -len;
                max_idx = len;
            } else {
                // When traversing backwards, the range of valid +ve indexes are
                // `[0, len-1]` and for -ve indexes `[-len-1, -1]`.
                min_idx = -len - 1;
                max_idx = len - 1;
            }

            SliceRange::new(
                self.start.clamp(min_idx, max_idx),
                self.end.map(|e| e.clamp(min_idx, max_idx)),
                self.step,
            )
        }

        fn offset_from_start(index: isize, dim_size: usize) -> isize {
            if index >= 0 {
                index
            } else {
                dim_size as isize + index
            }
        }
    }

    SliceRange::new(start, end, step).steps(dim_size) as i32
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_shape, sym_vec};

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
        let data = sym_shape!("batch", 64, 8);
        let starts = sym_vec!("start");
        let ends = sym_vec!("end");
        let axes = sym_vec!(0);
        let result = Slice
            .infer_shapes(&[data, starts, ends, axes], &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("unknown_1", 64, 8));

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
    }
}

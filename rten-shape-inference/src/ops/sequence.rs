//! Shape inference for operators that consume or produce sequences.

use crate::infer_shapes::{
    InferShapes, InferShapesContext, InferShapesError, resolve_axis, resolve_index,
};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_value::{Constant, SymValue};

use super::Concat;

/// Maximum length of a sequence for which shape inference tracks the shape of
/// each item.
///
/// Sequences longer than this are treated as having an unknown length. This
/// avoids generating a large amount of shape information from a single
/// operator.
const MAX_SEQUENCE_LEN: usize = 1024;

/// Resolve the position of an existing item in a sequence of length `len`.
///
/// Positions are specified as a value in `[-len, len)`, where negative values
/// count backwards from the end of the sequence.
fn resolve_position(len: usize, pos: i32) -> Result<usize, InferShapesError> {
    resolve_index(len, pos).ok_or(InferShapesError::InvalidValue)
}

/// SequenceEmpty operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html>.
pub struct SequenceEmpty;

impl InferShapes for SequenceEmpty {
    fn infer_shapes(
        &self,
        _inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        Ok([SymValue::from_sequence(Vec::new())].into())
    }
}

/// SequenceAt operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceAt.html>.
pub struct SequenceAt;

impl InferShapes for SequenceAt {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let seq = inputs.require(0)?;
        let pos = inputs.require(1)?;

        let (Some(items), Some(Constant::Scalar(pos))) = (seq.as_sequence(), pos.to_constant())
        else {
            return Ok([SymValue::unknown("unknown sequence or position")].into());
        };
        let pos = resolve_position(items.len(), pos)?;

        Ok([items[pos].clone()].into())
    }
}

/// SequenceConstruct operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html>.
pub struct SequenceConstruct;

impl InferShapes for SequenceConstruct {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        // The element type of the sequence comes from the first input, so at
        // least one input is required.
        inputs.require(0)?;

        let items = inputs
            .iter()
            .map(|item| {
                item.cloned()
                    .unwrap_or_else(|| SymValue::unknown("missing sequence item"))
            })
            .collect();

        Ok([SymValue::from_sequence(items)].into())
    }
}

/// SequenceErase operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceErase.html>.
pub struct SequenceErase;

impl InferShapes for SequenceErase {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let seq = inputs.require(0)?;
        let Some(items) = seq.as_sequence() else {
            return Ok([SymValue::unknown("unknown sequence")].into());
        };

        let pos = match inputs.get(1) {
            Some(pos) => {
                let Some(Constant::Scalar(pos)) = pos.to_constant() else {
                    return Ok([SymValue::unknown("unknown sequence position")].into());
                };
                resolve_position(items.len(), pos)?
            }
            // The last item is removed if no position is given.
            None => items
                .len()
                .checked_sub(1)
                .ok_or(InferShapesError::InvalidValue)?,
        };

        let mut items = items.to_vec();
        items.remove(pos);

        Ok([SymValue::from_sequence(items)].into())
    }
}

/// SequenceInsert operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceInsert.html>.
pub struct SequenceInsert;

impl InferShapes for SequenceInsert {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let seq = inputs.require(0)?;
        let value = inputs.require(1)?;
        let Some(items) = seq.as_sequence() else {
            return Ok([SymValue::unknown("unknown sequence")].into());
        };

        if items.len() >= MAX_SEQUENCE_LEN {
            return Ok([SymValue::unknown("sequence is too long")].into());
        }

        let pos = match inputs.get(2) {
            Some(pos) => {
                let Some(Constant::Scalar(pos)) = pos.to_constant() else {
                    return Ok([SymValue::unknown("unknown sequence position")].into());
                };
                resolve_position(items.len() + 1, pos)?
            }
            // The value is appended if no position is given.
            None => items.len(),
        };

        let mut items = items.to_vec();
        items.insert(pos, value.clone());

        Ok([SymValue::from_sequence(items)].into())
    }
}

/// SequenceLength operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SequenceLength.html>.
pub struct SequenceLength;

impl InferShapes for SequenceLength {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let seq = inputs.require(0)?;

        let len = match seq.as_sequence() {
            Some(items) => SymValue::from_scalar(SymExpr::Value(items.len() as i32)),
            // The length is a scalar, even if its value is unknown.
            None => SymValue::from_shape(Vec::new()),
        };

        Ok([len].into())
    }
}

/// ConcatFromSequence operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html>.
pub struct ConcatFromSequence {
    pub axis: i32,

    /// Stack the items along a new axis inserted at `axis`, instead of
    /// concatenating them along an existing axis.
    pub new_axis: bool,
}

impl InferShapes for ConcatFromSequence {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let seq = inputs.require(0)?;
        let Some(items) = seq.as_sequence() else {
            return Ok([SymValue::unknown("unknown sequence")].into());
        };
        if items.is_empty() {
            // Concatenating an empty sequence fails at runtime.
            return Err(InferShapesError::InvalidValue);
        }

        // Add the new axis to each item, then concatenate along it.
        let items: Vec<SymValue> = if self.new_axis {
            items
                .iter()
                .map(|item| {
                    let Some(dims) = item.shape() else {
                        return Ok(SymValue::unknown("unknown sequence item shape"));
                    };
                    let axis = resolve_axis(dims.len() + 1, self.axis)?;
                    let mut dims: Vec<SymExpr> = dims.collect();
                    dims.insert(axis, SymExpr::Value(1));
                    Ok(SymValue::from_shape(dims))
                })
                .collect::<Result<_, InferShapesError>>()?
        } else {
            items.to_vec()
        };

        Concat { axis: self.axis }.infer_shapes(items.into(), sym_gen)
    }
}

/// SplitToSequence operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__SplitToSequence.html>.
pub struct SplitToSequence {
    pub axis: i32,

    /// Whether to keep the split axis in each item, if the split sizes are not
    /// specified.
    pub keep_dims: bool,
}

/// Return the sizes of the chunks produced by splitting a dimension of size
/// `dim_size` into chunks of `chunk_size` items. The final chunk is smaller if
/// the dimension is not evenly divisible.
///
/// Returns `None` if the number of chunks cannot be determined or exceeds
/// [`MAX_SEQUENCE_LEN`].
fn chunk_sizes(dim_size: &SymExpr, chunk_size: i32) -> Option<Vec<SymExpr>> {
    let SymExpr::Value(dim_size) = *dim_size else {
        return None;
    };
    if dim_size < 0 || chunk_size < 1 {
        return None;
    }
    let (dim_size, chunk_size) = (dim_size as usize, chunk_size as usize);

    let n_chunks = dim_size.div_ceil(chunk_size);
    if n_chunks > MAX_SEQUENCE_LEN {
        return None;
    }

    let sizes = (0..n_chunks)
        .map(|chunk| SymExpr::Value(chunk_size.min(dim_size - chunk * chunk_size) as i32))
        .collect();

    Some(sizes)
}

impl InferShapes for SplitToSequence {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymValue>, InferShapesError> {
        let input = inputs.require(0)?;
        let Some(dims) = input.shape() else {
            return Ok([SymValue::unknown("unknown input shape")].into());
        };
        let axis = resolve_axis(dims.len(), self.axis)?;
        let dims: Vec<SymExpr> = dims.collect();

        let (split_sizes, keep_dims) = if let Some(splits) = inputs.get(1) {
            let split_sizes = match splits.to_constant() {
                Some(Constant::Scalar(size)) => {
                    if size < 1 {
                        return Err(InferShapesError::InvalidValue);
                    }
                    chunk_sizes(&dims[axis], size)
                }
                Some(Constant::Vector(sizes)) if sizes.len() <= MAX_SEQUENCE_LEN => {
                    Some(sizes.into_iter().map(SymExpr::Value).collect())
                }
                // The split sizes are not known, or the sequence would be
                // longer than shape inference tracks.
                _ => None,
            };
            let Some(split_sizes) = split_sizes else {
                return Ok([SymValue::unknown("unknown split sizes")].into());
            };
            // `keep_dims` is ignored if the split sizes are specified. The axis
            // can only be removed if every split has size 1.
            (split_sizes, true)
        } else {
            let Some(split_sizes) = chunk_sizes(&dims[axis], 1) else {
                return Ok([SymValue::unknown("unknown split count")].into());
            };
            (split_sizes, self.keep_dims)
        };

        let items = split_sizes
            .into_iter()
            .map(|size| {
                let mut dims = dims.clone();
                if keep_dims {
                    dims[axis] = size;
                } else {
                    dims.remove(axis);
                }
                SymValue::from_shape(dims)
            })
            .collect();

        Ok([SymValue::from_sequence(items)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_value::{SymValue, sym_scalar, sym_shape, sym_vec};

    use super::{
        ConcatFromSequence, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase,
        SequenceInsert, SequenceLength, SplitToSequence,
    };

    /// Run shape inference for an operator which has a single output.
    fn infer_one(
        op: &dyn InferShapes,
        inputs: Vec<Option<SymValue>>,
    ) -> Result<SymValue, InferShapesError> {
        let mut sym_gen = SymbolGen::new();
        let mut outputs = op.infer_shapes(inputs.into(), &mut sym_gen)?;
        assert_eq!(outputs.len(), 1);
        Ok(outputs.remove(0))
    }

    /// Return true if nothing is known about a value's shape.
    fn is_unknown(value: &SymValue) -> bool {
        value.ndim().is_none() && value.as_sequence().is_none()
    }

    #[test]
    fn test_sequence_empty() {
        let result = infer_one(&SequenceEmpty, [].into()).unwrap();
        assert_eq!(result.as_sequence(), Some([].as_slice()));
    }

    #[test]
    fn test_sequence_construct() {
        let items = [sym_shape!("batch", 16), sym_shape!("batch", 32)];
        let result = infer_one(&SequenceConstruct, items.clone().map(Some).into()).unwrap();
        assert_eq!(result.as_sequence(), Some(items.as_slice()));

        // Inputs with unknown shapes are still tracked as items.
        let result = infer_one(
            &SequenceConstruct,
            [Some(sym_shape!("batch", 16)), None].into(),
        )
        .unwrap();
        let items = result.as_sequence().unwrap();
        assert_eq!(items.len(), 2);
        assert!(is_unknown(&items[1]));

        // At least one input is required.
        let err = infer_one(&SequenceConstruct, [].into()).unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }

    #[test]
    fn test_sequence_at() {
        #[derive(Debug)]
        struct Case {
            seq: SymValue,
            pos: SymValue,
            /// Expected output, or `None` if the output is unknown.
            expected: Result<Option<SymValue>, InferShapesError>,
        }

        let seq = SymValue::from_sequence(vec![sym_shape!("batch", 16), sym_shape!("batch", 32)]);

        let cases = [
            Case {
                seq: seq.clone(),
                pos: sym_scalar!(0),
                expected: Ok(Some(sym_shape!("batch", 16))),
            },
            // Position counted from the end of the sequence.
            Case {
                seq: seq.clone(),
                pos: sym_scalar!(-1),
                expected: Ok(Some(sym_shape!("batch", 32))),
            },
            // Out-of-range position.
            Case {
                seq: seq.clone(),
                pos: sym_scalar!(2),
                expected: Err(InferShapesError::InvalidValue),
            },
            // Position which is not a known value.
            Case {
                seq: seq.clone(),
                pos: sym_scalar!("pos"),
                expected: Ok(None),
            },
            // Sequence whose items are not known.
            Case {
                seq: SymValue::unknown("unknown sequence"),
                pos: sym_scalar!(0),
                expected: Ok(None),
            },
        ];

        cases.test_each(|case| {
            let result = infer_one(
                &SequenceAt,
                [Some(case.seq.clone()), Some(case.pos.clone())].into(),
            )
            .map(|value| (!is_unknown(&value)).then_some(value));
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_sequence_erase() {
        let items = vec![sym_shape!(2), sym_shape!(3), sym_shape!(4)];
        let seq = SymValue::from_sequence(items.clone());

        // The last item is erased if no position is given.
        let result = infer_one(&SequenceErase, [Some(seq.clone()), None].into()).unwrap();
        assert_eq!(result.as_sequence(), Some(&items[..2]));

        // Erase an item at an explicit position.
        let result = infer_one(
            &SequenceErase,
            [Some(seq.clone()), Some(sym_scalar!(0))].into(),
        )
        .unwrap();
        assert_eq!(result.as_sequence(), Some(&items[1..]));

        // Erasing from an empty sequence fails at runtime.
        let empty = SymValue::from_sequence(Vec::new());
        let err = infer_one(&SequenceErase, [Some(empty), None].into()).unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);

        // Erasing at an unknown position leaves the items unknown.
        let result =
            infer_one(&SequenceErase, [Some(seq), Some(sym_scalar!("pos"))].into()).unwrap();
        assert!(is_unknown(&result));
    }

    #[test]
    fn test_sequence_insert() {
        let seq = SymValue::from_sequence(vec![sym_shape!(2), sym_shape!(3)]);
        let item = sym_shape!(4);

        // The item is appended if no position is given.
        let result = infer_one(
            &SequenceInsert,
            [Some(seq.clone()), Some(item.clone()), None].into(),
        )
        .unwrap();
        assert_eq!(
            result.as_sequence(),
            Some([sym_shape!(2), sym_shape!(3), sym_shape!(4)].as_slice())
        );

        // Insert at an explicit position.
        let result = infer_one(
            &SequenceInsert,
            [Some(seq.clone()), Some(item.clone()), Some(sym_scalar!(0))].into(),
        )
        .unwrap();
        assert_eq!(
            result.as_sequence(),
            Some([sym_shape!(4), sym_shape!(2), sym_shape!(3)].as_slice())
        );

        // Positions are resolved against the length of the new sequence, so
        // the item can be inserted after the last existing item.
        let result = infer_one(
            &SequenceInsert,
            [Some(seq), Some(item), Some(sym_scalar!(-1))].into(),
        )
        .unwrap();
        assert_eq!(
            result.as_sequence(),
            Some([sym_shape!(2), sym_shape!(3), sym_shape!(4)].as_slice())
        );
    }

    #[test]
    fn test_sequence_length() {
        let seq = SymValue::from_sequence(vec![sym_shape!(2), sym_shape!(3)]);
        let result = infer_one(&SequenceLength, [Some(seq)].into()).unwrap();
        assert_eq!(result, sym_scalar!(2));

        // The length of an unknown sequence is a scalar with an unknown value.
        let result = infer_one(
            &SequenceLength,
            [Some(SymValue::unknown("unknown sequence"))].into(),
        )
        .unwrap();
        assert_eq!(result.ndim(), Some(0));
        assert_eq!(result.to_constant(), None);
    }

    #[test]
    fn test_concat_from_sequence() {
        #[derive(Debug)]
        struct Case {
            seq: SymValue,
            axis: i32,
            new_axis: bool,
            /// Expected output, or `None` if the output is unknown.
            expected: Result<Option<SymValue>, InferShapesError>,
        }

        let seq = SymValue::from_sequence(vec![sym_shape!(2, "chans"), sym_shape!(3, "chans")]);

        let cases = [
            // Concat along an existing axis.
            Case {
                seq: seq.clone(),
                axis: 0,
                new_axis: false,
                expected: Ok(Some(sym_shape!(5, "chans"))),
            },
            // Concat along a new axis.
            Case {
                seq: SymValue::from_sequence(vec![sym_shape!(2, "chans"); 3]),
                axis: 0,
                new_axis: true,
                expected: Ok(Some(sym_shape!(3, 2, "chans"))),
            },
            // Concat along a new axis appended to the item shape.
            Case {
                seq: SymValue::from_sequence(vec![sym_shape!(2, "chans"); 3]),
                axis: -1,
                new_axis: true,
                expected: Ok(Some(sym_shape!(2, "chans", 3))),
            },
            // Concatenating an empty sequence fails at runtime.
            Case {
                seq: SymValue::from_sequence(Vec::new()),
                axis: 0,
                new_axis: false,
                expected: Err(InferShapesError::InvalidValue),
            },
            // Sequence whose items are not known.
            Case {
                seq: SymValue::unknown("unknown sequence"),
                axis: 0,
                new_axis: false,
                expected: Ok(None),
            },
        ];

        cases.test_each(|case| {
            let op = ConcatFromSequence {
                axis: case.axis,
                new_axis: case.new_axis,
            };
            let result = infer_one(&op, [Some(case.seq.clone())].into())
                .map(|value| value.simplify())
                .map(|value| (!is_unknown(&value)).then_some(value));
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_split_to_sequence() {
        #[derive(Debug)]
        struct Case {
            input: SymValue,
            splits: Option<SymValue>,
            axis: i32,
            keep_dims: bool,
            /// Expected sequence items, or `None` if the sequence is unknown.
            expected: Result<Option<Vec<SymValue>>, InferShapesError>,
        }

        let cases = [
            // Split into chunks of a given size.
            Case {
                input: sym_shape!(5, "chans"),
                splits: Some(sym_scalar!(2)),
                axis: 0,
                keep_dims: true,
                expected: Ok(Some(vec![
                    sym_shape!(2, "chans"),
                    sym_shape!(2, "chans"),
                    sym_shape!(1, "chans"),
                ])),
            },
            // Split into chunks with explicit sizes.
            Case {
                input: sym_shape!(5, "chans"),
                splits: Some(sym_vec!(3, 2)),
                axis: 0,
                keep_dims: false,
                expected: Ok(Some(vec![sym_shape!(3, "chans"), sym_shape!(2, "chans")])),
            },
            // Split into chunks of size one, retaining the split axis.
            Case {
                input: sym_shape!("batch", 3),
                splits: None,
                axis: 1,
                keep_dims: true,
                expected: Ok(Some(vec![sym_shape!("batch", 1); 3])),
            },
            // Split into chunks of size one, removing the split axis.
            Case {
                input: sym_shape!("batch", 3),
                splits: None,
                axis: -1,
                keep_dims: false,
                expected: Ok(Some(vec![sym_shape!("batch"); 3])),
            },
            // Split sizes which are not known.
            Case {
                input: sym_shape!(5, "chans"),
                splits: Some(SymValue::unknown("computed splits")),
                axis: 0,
                keep_dims: true,
                expected: Ok(None),
            },
            // Split of an axis with a symbolic size into chunks of size one.
            // The number of chunks is unknown.
            Case {
                input: sym_shape!("batch", 3),
                splits: None,
                axis: 0,
                keep_dims: true,
                expected: Ok(None),
            },
            // Split which produces more items than shape inference tracks.
            Case {
                input: sym_shape!(2048, 3),
                splits: None,
                axis: 0,
                keep_dims: true,
                expected: Ok(None),
            },
            // Invalid chunk size.
            Case {
                input: sym_shape!(5, "chans"),
                splits: Some(sym_scalar!(0)),
                axis: 0,
                keep_dims: true,
                expected: Err(InferShapesError::InvalidValue),
            },
            // Invalid axis.
            Case {
                input: sym_shape!(5, "chans"),
                splits: None,
                axis: 2,
                keep_dims: true,
                expected: Err(InferShapesError::IncorrectRank),
            },
        ];

        cases.test_each(|case| {
            let op = SplitToSequence {
                axis: case.axis,
                keep_dims: case.keep_dims,
            };
            let result = infer_one(&op, [Some(case.input.clone()), case.splits.clone()].into())
                .map(|value| value.as_sequence().map(|items| items.to_vec()));
            assert_eq!(result, case.expected);
        });
    }
}

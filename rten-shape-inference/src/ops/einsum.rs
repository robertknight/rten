use crate::einsum_parser::{EinsumExpr, ValidateError, expand_ellipsis};
use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Einsum operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Einsum.html>.
pub struct Einsum<'a> {
    pub equation: &'a str,
}

impl InferShapes for Einsum<'_> {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let Ok(expr) = EinsumExpr::parse(self.equation) else {
            return Err(InferShapesError::InvalidValue);
        };

        // Validate the inputs and determine the number of dimensions
        // represented by `...` across inputs which contain an ellipsis.
        let broadcast_ndim =
            match expr.validate_inputs(inputs.iter().map(|input| input.and_then(|t| t.ndim()))) {
                Ok(n) => n,
                // We can't determine the rank of the output without knowing the
                // rank of an ellipsis input.
                Err(ValidateError::UnknownRank) => {
                    return Ok([SymTensor::unknown("unknown einsum input shape")].into());
                }
                Err(ValidateError::IncorrectInputCount) => {
                    return Err(InferShapesError::IncorrectInputCount);
                }
                Err(ValidateError::RankMismatch) => return Err(InferShapesError::IncorrectRank),
                Err(ValidateError::TooManyDims) => return Err(InferShapesError::InvalidValue),
                Err(ValidateError::BroadcastMismatch) => {
                    return Err(InferShapesError::IncompatibleShapes);
                }
            };

        // Expand "..." in each input and output term to digit placeholders.
        let expanded_inputs: Vec<String> = expr
            .inputs
            .iter()
            .map(|t| expand_ellipsis(t, broadcast_ndim))
            .collect();
        let expanded_output = expand_ellipsis(&expr.output, broadcast_ndim);

        let out_shape: Vec<SymExpr> = expanded_output
            .chars()
            .map(|output_ch| {
                let mut out_dim: Option<SymExpr> = None;
                for (term, input) in expanded_inputs.iter().zip(inputs.iter()) {
                    let Some(dims) = input.and_then(|t| t.shape()) else {
                        continue;
                    };
                    debug_assert_eq!(term.chars().count(), dims.len());
                    for (ch, dim) in term.chars().zip(dims) {
                        if ch != output_ch {
                            continue;
                        }
                        if let Some(out_dim) = out_dim.as_mut() {
                            *out_dim = out_dim.broadcast(&dim);
                        } else {
                            out_dim = Some(dim);
                        }
                    }
                }
                out_dim.unwrap_or(sym_gen.gen_positive())
            })
            .collect();

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::Einsum;

    fn infer(equation: &str, inputs: &[SymTensor]) -> SymTensor {
        let mut sym_gen = SymbolGen::new();
        Einsum { equation }
            .infer_shapes(inputs.to_vec().into(), &mut sym_gen)
            .unwrap()
            .remove(0)
            // Simplify broadcasts in shape expression.
            .simplify()
    }

    #[test]
    fn test_einsum() {
        #[derive(Debug)]
        struct Case<'a> {
            equation: &'a str,
            inputs: Vec<SymTensor>,
            expected: SymTensor,
        }

        let cases = [
            // Identity.
            Case {
                equation: "ij->ij",
                inputs: vec![sym_shape!(2, 3)],
                expected: sym_shape!(2, 3),
            },
            // Transpose.
            Case {
                equation: "ij->ji",
                inputs: vec![sym_shape!(2, 3)],
                expected: sym_shape!(3, 2),
            },
            // No `->`: implicit output is the alphabetically-sorted set of
            // unique labels (here "ik" — j is repeated and so dropped).
            Case {
                equation: "ij,jk",
                inputs: vec![sym_shape!(2, 3), sym_shape!(3, 4)],
                expected: sym_shape!(2, 4),
            },
            // Matmul.
            Case {
                equation: "ij,jk->ik",
                inputs: vec![sym_shape!("M", "K"), sym_shape!("K", "N")],
                expected: sym_shape!("M", "N"),
            },
            // Reduction.
            Case {
                equation: "ij->i",
                inputs: vec![sym_shape!("M", "N")],
                expected: sym_shape!("M"),
            },
            // Diagonal of a square matrix.
            Case {
                equation: "ii->i",
                inputs: vec![sym_shape!(4, 4)],
                expected: sym_shape!(4),
            },
            // Scalar output.
            Case {
                equation: "i,i->",
                inputs: vec![sym_shape!(4), sym_shape!(4)],
                expected: sym_shape!(),
            },
            // `...ij->...ji` — ellipsis stands for leading batch dims; the
            // trailing two dims are transposed.
            Case {
                equation: "...ij->...ji",
                inputs: vec![sym_shape!("A", "B", 3, 4)],
                expected: sym_shape!("A", "B", 4, 3),
            },
            // No `->` and inputs have ellipsis: implicit output is `...` plus
            // sorted unique letters.
            Case {
                equation: "...ij",
                inputs: vec![sym_shape!(2, "I", "J")],
                expected: sym_shape!(2, "I", "J"),
            },
            // Outer product.
            Case {
                equation: "i,j->ij",
                inputs: vec![sym_shape!(4), sym_shape!(5)],
                expected: sym_shape!(4, 5),
            },
        ];

        cases.test_each(|case| {
            assert_eq!(infer(case.equation, &case.inputs), case.expected);
        });
    }

    #[test]
    fn test_einsum_unknown_input_shape() {
        // Without ellipsis, an unknown input doesn't prevent us from
        // determining the rank when other inputs cover the output labels.
        let a = sym_shape!("M", "K");
        let b = SymTensor::unknown("unknown");
        let result = infer("ij,jk->ik", &[a, b]);
        let shape: Vec<_> = result.shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], SymExpr::from("M"));
        // The size for 'k' isn't known.
        assert!(matches!(shape[1], SymExpr::Var(_)));
    }

    #[test]
    fn test_einsum_unknown_ellipsis_input() {
        // When the operand expanding `...` has unknown rank, the output
        // shape is reported as unknown.
        let x = SymTensor::unknown("unknown");
        let result = infer("...ij->...ji", &[x]);
        assert_eq!(result.ndim(), None);
    }
}

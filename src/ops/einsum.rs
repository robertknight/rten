use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use smallvec::SmallVec;

use crate::ops::{matmul, InputList, IntoOpResult, OpError, Operator, OutputList};
use crate::tensor_pool::TensorPool;

/// A parsed equation for an Einsum operator.
///
/// Einsum expressions have the form `abc,def,...->xyz` where the `->xyz` part
/// is optional. If ommitted, it is inferred as the alphabetically ordered
/// set of letters from the left hand side that do not repeat.
struct EinsumExpr {
    inputs: Vec<String>,
    output: String,
}

impl EinsumExpr {
    /// Parse an [Einsum expression][einsum].
    ///
    /// [einsum]: https://onnx.ai/onnx/operators/onnx__Einsum.html
    fn parse(expr: &str) -> Result<EinsumExpr, OpError> {
        let mut parts = expr.trim().splitn(2, "->").map(|part| part.trim());

        let lhs = match parts.next() {
            Some(lhs) if !lhs.is_empty() => lhs,
            _ => {
                return Err(OpError::InvalidValue("Invalid equation"));
            }
        };

        let inputs: Vec<_> = lhs.split(',').map(|term| term.trim().to_string()).collect();
        if inputs.iter().any(|term| !is_valid_einsum_term(term)) {
            return Err(OpError::InvalidValue(
                "Einsum terms must contain only lowercase letters",
            ));
        }

        let output: String = match parts.next() {
            Some(rhs) => rhs.to_string(),
            None => {
                const N_LETTERS: usize = 26;

                // Count occurences of each lowercase ASCII letter.
                let mut char_count = [0; N_LETTERS];
                for ch in inputs.iter().flat_map(|term| term.chars()) {
                    let ascii_idx = ch as u8 - b'a';
                    char_count[ascii_idx as usize] += 1;
                }

                // Generate output as sequence of alphabetically ordered
                // letters which appear only once in the input.
                let mut output = String::with_capacity(N_LETTERS);
                for i in 0..N_LETTERS as u8 {
                    if char_count[i as usize] == 1 {
                        let ascii_ch = b'a' + i;
                        output.push(ascii_ch as char);
                    }
                }

                output
            }
        };

        if !is_valid_einsum_term(&output) {
            return Err(OpError::InvalidValue(
                "Einsum terms must contain only lowercase letters",
            ));
        }

        Ok(EinsumExpr { inputs, output })
    }

    /// Return the dimensions in the expression which are summed over.
    fn reduced_dims(&self) -> Vec<char> {
        let mut terms = Vec::new();
        for in_term in &self.inputs {
            for in_ch in in_term.chars() {
                if !terms.contains(&in_ch) && !self.output.contains(in_ch) {
                    terms.push(in_ch);
                }
            }
        }
        terms
    }
}

fn is_valid_einsum_term(term: &str) -> bool {
    term.chars().all(|c| c.is_ascii_lowercase())
}

#[derive(Debug)]
pub struct Einsum {
    pub equation: String,
}

impl Operator for Einsum {
    fn name(&self) -> &str {
        "Einsum"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let mut typed_inputs: SmallVec<[TensorView; 2]> = SmallVec::with_capacity(inputs.len());
        for i in 0..inputs.len() {
            typed_inputs.push(inputs.require_as(i)?);
        }
        einsum(pool, &typed_inputs, &self.equation).into_op_result()
    }
}

pub fn einsum(pool: &TensorPool, inputs: &[TensorView], equation: &str) -> Result<Tensor, OpError> {
    let equation = EinsumExpr::parse(equation)?;
    if equation.inputs.len() != inputs.len() {
        return Err(OpError::InvalidValue(
            "Einsum equation input count does not match operator inputs",
        ));
    }

    if inputs.len() == 1 {
        let in_order = &equation.inputs[0];
        let out_order = &equation.output;
        if !is_valid_permute_spec(in_order, out_order) {
            return Err(OpError::UnsupportedValue("Unsupported Einsum equation"));
        }
        let output = permuted_by_labels(&inputs[0], in_order, out_order).to_tensor_in(pool);
        return Ok(output);
    }

    if inputs.len() != 2 {
        return Err(OpError::UnsupportedValue(
            "Einsum implementation only supports two inputs",
        ));
    }

    let x = &inputs[0];
    let y = &inputs[1];
    let term1 = &equation.inputs[0];
    let term2 = &equation.inputs[1];

    let reduced_dims = equation.reduced_dims();
    if reduced_dims.len() != 1 {
        return Err(OpError::UnsupportedValue(
            "Einsum only supports equations with one reduced dimension",
        ));
    }
    let matmul_k = reduced_dims[0];

    // Find a term that can be used as the `N` dimension of a matmul.
    let Some(matmul_n) = term2.chars().find(|c| !term1.contains(*c)) else {
        return Err(OpError::UnsupportedValue(
            "Cannot evaluate Einsum using matmul",
        ));
    };

    // Find a term that can be used as the `M` dimension of a matmul.
    let Some(matmul_m) = term1.chars().rev().find(|c| !term2.contains(*c)) else {
        return Err(OpError::UnsupportedValue(
            "Cannot evaluate Einsum using matmul",
        ));
    };

    // Find the terms that will be used as the batch dimensions of a matmul.
    let batch_dims = term1
        .chars()
        .filter(|c| *c != matmul_k && *c != matmul_n && *c != matmul_m);

    let mut x_order: String = batch_dims.clone().collect();
    x_order.push(matmul_m);
    x_order.push(matmul_k);

    let mut y_order: String = batch_dims
        .clone()
        .filter(|bc| term2.contains(*bc))
        .collect();
    y_order.push(matmul_k);
    y_order.push(matmul_n);

    let mut out_order: String = batch_dims.collect();
    out_order.push(matmul_m);
    out_order.push(matmul_n);

    let xp = permuted_by_labels(x, term1, &x_order);
    let yp = permuted_by_labels(y, term2, &y_order);
    let out = matmul(pool, xp, yp)?;

    if out_order == equation.output {
        Ok(out)
    } else {
        let out_permuted = permuted_by_labels(&out.view(), &out_order, &equation.output);
        Ok(out_permuted.to_tensor_in(pool))
    }
}

/// Return true if `src` and `dest` contain the same set of unique letters,
/// ignoring the order.
///
/// Both strings are assumed to be short.
fn is_valid_permute_spec(src: &str, dest: &str) -> bool {
    if src.len() != dest.len() {
        return false;
    }
    for src_ch in src.chars() {
        let src_count = src.chars().filter(|c| *c == src_ch).count();
        let dest_count = dest.chars().filter(|c| *c == src_ch).count();
        if src_count != 1 || dest_count != 1 {
            return false;
        }
    }
    true
}

/// Permute a tensor by specifying label strings specifying the input and output
/// order of dimensions.
///
/// For example `permute(tensor, "xy", "yx")` will transpose a matrix.
fn permuted_by_labels<'a, T>(
    tensor: &TensorView<'a, T>,
    in_order: &str,
    out_order: &str,
) -> TensorView<'a, T> {
    assert!(
        is_valid_permute_spec(in_order, out_order),
        "invalid permute spec {}->{}",
        in_order,
        out_order
    );
    assert!(
        tensor.ndim() == in_order.len(),
        "input order does not match tensor ndim"
    );
    let perm: Vec<usize> = out_order
        .chars()
        .map(|c| in_order.chars().position(|ic| ic == c).unwrap())
        .collect();
    tensor.permuted(&perm)
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, TensorView};

    use crate::ops::tests::new_pool;
    use crate::ops::{einsum, matmul, OpError};

    #[test]
    fn test_einsum() {
        struct Case<'a> {
            equation: &'a str,
            inputs: Vec<TensorView<'a>>,
            expected: Result<Tensor, OpError>,
        }

        let pool = new_pool();
        let vec_a = Tensor::arange(1., 10., None);
        let mat_a = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
        let mat_b = Tensor::from([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]);
        let matmul_ab = matmul(&pool, mat_a.view(), mat_b.view()).unwrap();
        let matmul_ba = matmul_ab.transposed().to_tensor();

        let bhwc = mat_a
            .clone()
            .into_shape([1, 1, mat_a.size(0), mat_a.size(1)]);
        let hck = mat_b.clone().into_shape([1, mat_b.size(0), mat_b.size(1)]);

        let bhwk = matmul_ab
            .clone()
            .into_shape([1, 1, mat_a.size(0), mat_b.size(1)]);

        let cases = [
            // Identity
            Case {
                equation: "ij->ij",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.clone()),
            },
            // Transpose
            Case {
                equation: "ij->ji",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.transposed().to_tensor()),
            },
            // Transpose with ignored spaces
            Case {
                equation: " ij -> ji ",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.transposed().to_tensor()),
            },
            // Transpose with implicit output
            Case {
                equation: "ba",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.transposed().to_tensor()),
            },
            // Matmul
            Case {
                equation: "ij,jk->ik",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Ok(matmul_ab.clone()),
            },
            // Matmul with implicit output
            Case {
                equation: "ij,jk",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Ok(matmul_ab.clone()),
            },
            // Matmul with transposed inputs
            Case {
                equation: "ji,kj->ik",
                inputs: vec![mat_a.transposed(), mat_b.transposed()],
                expected: Ok(matmul_ab),
            },
            // Matmul with transposed output
            Case {
                equation: "ij,jk->ki",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Ok(matmul_ba),
            },
            // Matmul with batch dimensions.
            // Example taken from image encoder of https://huggingface.co/facebook/sam-vit-base.
            Case {
                equation: "bhwc,hkc->bhwk",
                inputs: vec![bhwc.as_dyn(), hck.permuted([0, 2, 1]).as_dyn()],
                expected: Ok(bhwk.into_dyn()),
            },
            // Incorrect input count
            Case {
                equation: "ij,jk->ik",
                inputs: vec![mat_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum equation input count does not match operator inputs",
                )),
            },
            // Unsupported dot product
            Case {
                equation: "i,i->",
                inputs: vec![vec_a.view(), vec_a.view()],
                expected: Err(OpError::UnsupportedValue(
                    "Cannot evaluate Einsum using matmul",
                )),
            },
            // Unsupported matrix-vector product
            Case {
                equation: "ij,j->i",
                inputs: vec![mat_a.view(), mat_b.slice_dyn(0)],
                expected: Err(OpError::UnsupportedValue(
                    "Cannot evaluate Einsum using matmul",
                )),
            },
            // Unsupported vector-matrix product
            Case {
                equation: "j,jk->k",
                inputs: vec![mat_a.slice_dyn(0), mat_b.view()],
                expected: Err(OpError::UnsupportedValue(
                    "Cannot evaluate Einsum using matmul",
                )),
            },
            // Unsupported number of reduced dimensions
            Case {
                equation: "ij,kl->ijkl",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Err(OpError::UnsupportedValue(
                    "Einsum only supports equations with one reduced dimension",
                )),
            },
            // Empty equation
            Case {
                equation: "",
                inputs: vec![],
                expected: Err(OpError::InvalidValue("Invalid equation")),
            },
            // Invalid input terms
            Case {
                equation: "IJ,JK",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum terms must contain only lowercase letters",
                )),
            },
            // Invalid output term
            Case {
                equation: "ij,jk->IK",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum terms must contain only lowercase letters",
                )),
            },
        ];

        for Case {
            equation,
            inputs,
            expected,
        } in cases
        {
            let output = einsum(&pool, inputs.as_slice(), equation);
            assert_eq!(output, expected);
        }
    }
}

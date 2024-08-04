use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use smallvec::SmallVec;

use crate::ops::layout::expand_to;
use crate::ops::{
    matmul, mul, mul_in_place, reduce_sum, InputList, IntoOpResult, OpError, Operator, OutputList,
};
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
                return Err(OpError::InvalidValue(
                    "Einsum equation must have at least one term",
                ));
            }
        };

        let inputs: Vec<String> = lhs
            .split(',')
            .map(|term| non_whitespace_chars(term).collect())
            .collect();
        if inputs.iter().any(|term| !is_valid_einsum_term(term)) {
            return Err(OpError::InvalidValue(
                "Einsum terms must contain only lowercase letters",
            ));
        }

        let output: String = match parts.next() {
            Some(rhs) => non_whitespace_chars(rhs).collect(),
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

fn non_whitespace_chars(s: &str) -> impl Iterator<Item = char> + '_ {
    s.chars().filter(|c| !c.is_ascii_whitespace())
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
            "Number of terms in Einsum equation does not match input tensor count",
        ));
    }
    for (term, view) in equation.inputs.iter().zip(inputs) {
        if term.len() != view.ndim() {
            return Err(OpError::InvalidValue(
                "Einsum term dimension count does not match input tensor",
            ));
        }
    }

    // For einsum equations with a single input or no reduced dimensions, the
    // naive implementation is efficient.
    //
    // For einsum equations with more than two inputs we fall back to the naive
    // implementation, which is much slower if reductions are required.
    let reduced_dims = equation.reduced_dims();
    if inputs.len() != 2 || reduced_dims.is_empty() {
        return naive_einsum(pool, inputs, &equation);
    }

    let x = &inputs[0];
    let y = &inputs[1];
    let term1 = &equation.inputs[0];
    let term2 = &equation.inputs[1];

    if reduced_dims.len() == 1 {
        einsum_matmul(pool, x, y, term1, term2, &equation.output, reduced_dims[0])
    } else {
        // Re-arrange input views as `[output_dims][reduced_dims]`. This makes
        // the reduced dimensions adjacent.
        let reduced_dims = equation.reduced_dims();
        let common_order: String = equation
            .output
            .chars()
            .chain(reduced_dims.iter().copied())
            .collect();
        let xp = permute_and_insert_axes(x, term1, &common_order);
        let yp = permute_and_insert_axes(y, term2, &common_order);

        // Expand the reduced dimensions of each input if needed, so they are
        // the same size. Note that the non-reduced dimensions are not expanded,
        // they will be broadcast if needed during the matmul.
        let mut tmp_x_shape = xp.shape().to_vec();
        let mut tmp_y_shape = yp.shape().to_vec();
        for i in xp.ndim() - reduced_dims.len()..xp.ndim() {
            tmp_x_shape[i] = tmp_x_shape[i].max(tmp_y_shape[i]);
            tmp_y_shape[i] = tmp_y_shape[i].max(tmp_x_shape[i]);
        }
        let x = if tmp_x_shape == xp.shape() {
            x.to_contiguous_in(pool)
        } else {
            expand_to(pool, x.view(), &tmp_x_shape).into_cow()
        };
        let y = if tmp_y_shape == yp.shape() {
            y.to_contiguous_in(pool)
        } else {
            expand_to(pool, y.view(), &tmp_y_shape).into_cow()
        };

        // Reshape the adjacent reduced dimensions into a single dimension.
        let reduced_dims_start_index = xp.ndim() - reduced_dims.len();
        let reduced_size: usize = xp.shape()[reduced_dims_start_index..].iter().product();

        tmp_x_shape.truncate(reduced_dims_start_index);
        tmp_x_shape.push(reduced_size);
        let x = x.reshaped(tmp_x_shape.as_slice());

        tmp_y_shape.truncate(reduced_dims_start_index);
        tmp_y_shape.push(reduced_size);
        let y = y.reshaped(tmp_y_shape.as_slice());

        // Evaluate the equation with the simplified input shapes using a
        // matmul.
        let reduced_dim = 'K'; // Upper-case to avoid conflict with equation
                               // terms.
        let term_simplified: String = equation
            .output
            .chars()
            .chain(std::iter::once(reduced_dim))
            .collect();
        einsum_matmul(
            pool,
            &x,
            &y,
            &term_simplified,
            &term_simplified,
            &equation.output,
            reduced_dim,
        )
    }
}

fn is_valid_permute_insert_spec(src: &str, dest: &str) -> bool {
    if src.len() > dest.len() {
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

/// Permute a tensor by using label strings to specify the input and output
/// order of dimensions.
///
/// All dimensions listed in the input order must occur in the output order. The
/// output order may contain dimensions that are missing from the input order.
/// In that case a 1-sized dimension will be inserted.
///
/// Examples of input and output orders:
///
/// `"xy", "yx"` - Transpose a matrix
/// `"x", "axb"` - Insert two 1-sized dimensions
fn permute_and_insert_axes<'a, T>(
    tensor: &TensorView<'a, T>,
    in_order: &str,
    out_order: &str,
) -> TensorView<'a, T> {
    assert!(
        is_valid_permute_insert_spec(in_order, out_order),
        "invalid permute-and-insert spec {}->{}",
        in_order,
        out_order
    );
    assert!(
        tensor.ndim() == in_order.len(),
        "input order does not match tensor ndim"
    );
    let perm: Vec<usize> = out_order
        .chars()
        .filter_map(|c| in_order.chars().position(|ic| ic == c))
        .collect();
    let mut permuted = tensor.permuted(&perm);

    for (i, c) in out_order.chars().enumerate() {
        if !in_order.contains(c) {
            permuted.insert_axis(i);
        }
    }

    permuted
}

/// Simple non-optimized Einsum implementation.
///
/// This uses a combination of transpose, multiplication and reduce-sum
/// operations. It is efficient for cases where only multiplication or only
/// reduction (plus transpose) is needed. It is not efficient when both
/// multiplication and reduction are needed. Such cases are much more
/// efficiently handled as matrix multiplication.
fn naive_einsum(
    pool: &TensorPool,
    inputs: &[TensorView],
    equation: &EinsumExpr,
) -> Result<Tensor, OpError> {
    // Re-arrange input views as `[output_dims][reduced_dims]`.
    let reduced_dims = equation.reduced_dims();
    let common_order: String = equation
        .output
        .chars()
        .chain(reduced_dims.iter().copied())
        .collect();
    let mut broadcasted_inputs: Vec<_> = inputs
        .iter()
        .zip(&equation.inputs)
        .map(|(inp, term)| permute_and_insert_axes(inp, term, &common_order))
        .collect();

    // Multiply corresponding elements of each input.
    let mut output = broadcasted_inputs.remove(0).to_tensor_in(pool);
    for rhs in broadcasted_inputs.into_iter() {
        if rhs.can_broadcast_to(output.shape()) {
            mul_in_place(output.view_mut(), rhs)
        } else {
            output = mul(pool, output.view(), rhs)?;
        }
    }

    // Sum over the reduced dimensions.
    if reduced_dims.is_empty() {
        return Ok(output);
    }

    let reduced_dim_indices: Vec<i32> = (0..reduced_dims.len()).map(|i| i as i32 - 1).collect();
    return reduce_sum(
        pool,
        output.view(),
        Some(reduced_dim_indices.as_slice()),
        false, /* keep_dims */
    );
}

/// Reduce inputs of an Einsum equation with two terms using matrix
/// multiplication.
///
/// The equation must have a single reduced dimension.
fn einsum_matmul(
    pool: &TensorPool,
    x: &TensorView,
    y: &TensorView,
    term1: &str,
    term2: &str,
    output: &str,
    reduced_dim: char,
) -> Result<Tensor, OpError> {
    let matmul_k = reduced_dim;

    // Find terms that can be used as the `N` and `M` dimensions of a matmul.
    //
    // If there aren't suitable dimensions, we'll insert them. Upper-case
    // letters are used to denote inserted dimensions since these cannot
    // conflict with dimensions in the einsum equation.
    let matmul_n = term2.chars().find(|c| !term1.contains(*c)).unwrap_or('N');
    let matmul_m = term1
        .chars()
        .rev()
        .find(|c| !term2.contains(*c))
        .unwrap_or('M');

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
    if matmul_m.is_ascii_lowercase() {
        out_order.push(matmul_m);
    }
    if matmul_n.is_ascii_lowercase() {
        out_order.push(matmul_n);
    }

    let xp = permute_and_insert_axes(x, term1, &x_order);
    let yp = permute_and_insert_axes(y, term2, &y_order);
    let mut out = matmul(pool, xp, yp)?;

    if !matmul_m.is_ascii_lowercase() {
        out.remove_axis(out.ndim() - 2);
    }
    if !matmul_n.is_ascii_lowercase() {
        out.remove_axis(out.ndim() - 1);
    }

    if out_order == output {
        Ok(out)
    } else {
        let out_permuted = permute_and_insert_axes(&out.view(), &out_order, output);
        Ok(out_permuted.to_tensor_in(pool))
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, TensorView};

    use crate::ops::tests::new_pool;
    use crate::ops::{einsum, matmul, mul, reduce_sum, OpError};

    #[test]
    fn test_einsum() {
        struct Case<'a> {
            equation: &'a str,
            inputs: Vec<TensorView<'a>>,
            expected: Result<Tensor, OpError>,
        }

        let pool = new_pool();

        let vec_a = Tensor::arange(1., 10., None);
        let vec_b = Tensor::arange(1., 5., None);

        let mat_a = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
        let mat_b = Tensor::from([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]);
        let matmul_ab = matmul(&pool, mat_a.view(), mat_b.view()).unwrap();
        let matmul_ba = matmul_ab.transposed().to_tensor();
        let outer_mat_ab = mul(
            &pool,
            mat_a
                .reshaped([mat_a.size(0), mat_a.size(1), 1, 1])
                .as_dyn(),
            mat_b
                .reshaped([1, 1, mat_b.size(0), mat_b.size(1)])
                .as_dyn(),
        )
        .unwrap();

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
            // Spaces between letters
            Case {
                equation: "i j -> i j",
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
            // Reduction of a single input
            Case {
                equation: "ij->i",
                inputs: vec![mat_a.view()],
                expected: Ok(reduce_sum(
                    &pool,
                    mat_a.view(),
                    Some(&[-1]),
                    false, /* keep_dims */
                )
                .unwrap()),
            },
            // Outer product of two vectors
            Case {
                equation: "i,j->ij",
                inputs: vec![vec_a.view(), vec_b.view()],
                expected: Ok(mul(
                    &pool,
                    vec_a.reshaped([vec_a.len(), 1]).as_dyn(),
                    vec_b.reshaped([1, vec_b.len()]).as_dyn(),
                )
                .unwrap()),
            },
            // Outer product of two matrices
            Case {
                equation: "ij,kl->ijkl",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Ok(outer_mat_ab),
            },
            // Outer product with transpose
            Case {
                equation: "a,b->ba",
                inputs: vec![vec_a.view(), vec_b.view()],
                expected: Ok(mul(
                    &pool,
                    vec_b.reshaped([vec_b.len(), 1]).as_dyn(),
                    vec_a.reshaped([1, vec_a.len()]).as_dyn(),
                )
                .unwrap()),
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
                    "Number of terms in Einsum equation does not match input tensor count",
                )),
            },
            // Dot product
            Case {
                equation: "i,i->",
                inputs: vec![vec_a.view(), vec_a.view()],
                expected: Ok(Tensor::from(vec_a.iter().map(|a| a * a).sum::<f32>())),
            },
            // Matrix-vector product
            Case {
                equation: "ij,j->i",
                inputs: vec![mat_a.view(), mat_b.slice_dyn((.., 0))],
                expected: Ok(matmul(&pool, mat_a.view(), mat_b.slice_dyn((.., ..1)))
                    .unwrap()
                    .into_shape([mat_a.size(0)].as_slice())),
            },
            // Vector-matrix product
            Case {
                equation: "j,jk->k",
                inputs: vec![mat_a.slice_dyn(0), mat_b.view()],
                expected: Ok(matmul(&pool, mat_a.slice_dyn((..1, ..)), mat_b.view())
                    .unwrap()
                    .into_shape([mat_b.size(1)].as_slice())),
            },
            // Reduction over two dimensions
            Case {
                equation: "ij,ij->",
                inputs: vec![mat_a.view(), mat_a.view()],
                expected: Ok(Tensor::from(mat_a.iter().map(|x| x * x).sum::<f32>())),
            },
            // Reduction over four dimensions
            Case {
                equation: "bhwc,bhwc->",
                inputs: vec![bhwc.as_dyn(), bhwc.as_dyn()],
                expected: Ok(Tensor::from(bhwc.iter().map(|x| x * x).sum::<f32>())),
            },
            // Reduction over multiple dimensions where the reduced dimensions
            // are not present in all tensors.
            Case {
                equation: "ij,j->",
                inputs: vec![mat_a.view(), mat_b.slice_dyn((.., 0))],
                expected: Ok(Tensor::from(
                    mat_a
                        .iter()
                        .zip(mat_b.slice_dyn((.., 0)).broadcast(mat_a.shape()).iter())
                        .map(|(x, y)| x * y)
                        .sum::<f32>(),
                )),
            },
            // Empty equation
            Case {
                equation: "",
                inputs: vec![],
                expected: Err(OpError::InvalidValue(
                    "Einsum equation must have at least one term",
                )),
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
            // Mismatch between input ndim and term dimension count
            Case {
                equation: "ij",
                inputs: vec![vec_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum term dimension count does not match input tensor",
                )),
            },
            // Three input dot product
            Case {
                equation: "i,i,i->",
                inputs: vec![vec_a.view(), vec_a.view(), vec_a.view()],
                expected: Ok(Tensor::from(vec_a.map(|x| x * x * x).iter().sum::<f32>())),
            },
        ];

        for Case {
            equation,
            inputs,
            expected,
        } in cases
        {
            let output = einsum(&pool, inputs.as_slice(), equation);
            assert_eq!(
                output, expected,
                "result mismatch for equation {}",
                equation
            );
        }
    }
}

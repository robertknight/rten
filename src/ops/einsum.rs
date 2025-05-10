use std::collections::HashMap;

use rten_tensor::prelude::*;
use rten_tensor::{DynLayout, OverlapPolicy, Tensor, TensorView};

use smallvec::SmallVec;

use crate::ops::layout::expand_to;
use crate::ops::{
    matmul, mul, reduce_sum, IntoOpResult, OpError, OpRunContext, Operator, OutputList,
};
use crate::tensor_pool::{AutoReturn, PoolRef, TensorPool};

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
    /// The expression must contain at least one input term.
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
            return Err(OpError::InvalidValue("Input term is invalid"));
        }

        let output: String = match parts.next() {
            Some(rhs) => non_whitespace_chars(rhs).collect(),
            None => {
                const N_LETTERS: usize = 26;

                // Count occurences of each lowercase ASCII letter.
                let mut char_count = [0; N_LETTERS];
                for ch in inputs
                    .iter()
                    .flat_map(|term| term.chars().filter(|c| c.is_ascii_lowercase()))
                {
                    let ascii_idx = ch as u8 - b'a';
                    char_count[ascii_idx as usize] += 1;
                }

                // Generate output as sequence of alphabetically ordered
                // letters which appear only once in the input.
                let mut output = String::with_capacity(N_LETTERS);

                if inputs.iter().any(|term| term.contains("...")) {
                    output.push_str("...");
                }

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
            return Err(OpError::InvalidValue("Output term is invalid"));
        }
        if contains_repeated_chars(&output) {
            return Err(OpError::InvalidValue(
                "Einsum output term contains repeated labels",
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
    if let Some((lhs, rhs)) = term.split_once("...") {
        is_valid_einsum_term(lhs) && !rhs.contains("...") && is_valid_einsum_term(rhs)
    } else {
        term.chars().all(|c| c.is_ascii_lowercase())
    }
}

fn non_whitespace_chars(s: &str) -> impl Iterator<Item = char> + '_ {
    s.chars().filter(|c| !c.is_ascii_whitespace())
}

fn contains_repeated_chars(term: &str) -> bool {
    term.chars()
        .filter(|c| *c != '.')
        .any(|c1| term.chars().filter(|c2| c1 == *c2).count() > 1)
}

#[derive(Debug)]
pub struct Einsum {
    pub equation: String,
}

impl Operator for Einsum {
    fn name(&self) -> &str {
        "Einsum"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let mut typed_inputs: SmallVec<[TensorView; 2]> = SmallVec::with_capacity(inputs.len());
        for i in 0..inputs.len() {
            typed_inputs.push(inputs.require_as(i)?);
        }
        einsum(ctx.pool(), &typed_inputs, &self.equation).into_op_result()
    }
}

pub fn einsum(
    pool: &TensorPool,
    inputs: &[TensorView],
    equation_str: &str,
) -> Result<Tensor, OpError> {
    let equation = EinsumExpr::parse(equation_str)?;
    if equation.inputs.len() != inputs.len() {
        return Err(OpError::InvalidValue(
            "Number of terms in Einsum equation does not match input tensor count",
        ));
    }

    // Maximum number of dimensions allowed. This value is chosen to make it
    // easy to use single digits to represent broadcasting dimensions in terms.
    const MAX_DIMS: usize = 10;

    // Number of dimensions represented by "..." in equation. This must be the
    // same for every term.
    let mut broadcast_ndim = None;
    for (term, view) in equation.inputs.iter().zip(inputs) {
        let non_broadcast_ndim = term.split("...").fold(0, |len, term| len + term.len());
        if view.ndim() < non_broadcast_ndim {
            return Err(OpError::InvalidValue(
                "Einsum term dimension count does not match input tensor",
            ));
        }
        if non_broadcast_ndim > MAX_DIMS || view.ndim() > MAX_DIMS {
            return Err(OpError::UnsupportedValue(
                "Einsum input or term has too many dimensions",
            ));
        }

        if term.contains("...") {
            let new_broadcast_ndim = (view.ndim() - non_broadcast_ndim) as u8;
            match broadcast_ndim {
                None => {
                    broadcast_ndim = Some(new_broadcast_ndim);
                }
                Some(b) if b == new_broadcast_ndim => {}
                _ => {
                    return Err(OpError::InvalidValue(
                        "Number of broadcast dims does not match across inputs",
                    ));
                }
            }
        } else if term.len() != view.ndim() {
            return Err(OpError::InvalidValue(
                "Einsum term dimension count does not match input tensor",
            ));
        }
    }

    let path = einsum_path(&equation, broadcast_ndim.unwrap_or(0));

    let mut output: Option<PoolRef<Tensor>> = None;
    for step in &path {
        let output_view = output.as_ref().map(|o| o.view());
        let x = match step.lhs.input {
            EinsumInput::Index(idx) => &inputs[idx as usize],
            EinsumInput::PrevOutput => output_view.as_ref().expect("invalid einsum path"),
        };
        let y = step.rhs.as_ref().map(|rhs| match rhs.input {
            EinsumInput::Index(idx) => &inputs[idx as usize],
            EinsumInput::PrevOutput => output_view.as_ref().expect("invalid einsum path"),
        });
        let new_output = einsum_step(pool, step, x, y)?.auto_return(pool);
        output = Some(new_output);
    }

    // EinsumExpr ensures that equations have at least one input, so the path
    // should never be empty.
    Ok(output.expect("empty path").take())
}

/// Take diagonals over dimensions which are repeated in an einsum term.
///
/// `term` is a sequence of dimension labels. For any labels that are repeated,
/// the corresponding dimensions in `x` are replaced with a single dimension
/// that is the diagonal.
///
/// For example, `take_diagonals("ii", x)` takes a matrix as input and returns
/// a 1D view that is the diagonal. `take_diagonals("iji", x)` takes a 3D
/// tensor as input and returns a 2D view.
///
/// Dimensions over which diagonals are taken must be the same size. An error
/// is returned if this is not the case.
///
/// Returns a tuple of `(unique_labels, diagonal_view)`.
fn take_diagonals<'a>(term: &str, x: &TensorView<'a>) -> Result<(String, TensorView<'a>), OpError> {
    assert!(term.chars().count() == x.ndim());

    let mut out_shape: Vec<usize> = Vec::new();
    let mut out_strides: Vec<usize> = Vec::new();
    let mut unique_dims = String::with_capacity(term.len());

    for (i, label) in (0..x.ndim()).zip(term.chars()) {
        if unique_dims.contains(label) {
            // We have already added the diagonal for this label to the output.
            continue;
        }
        unique_dims.push(label);

        let dim_size = x.size(i);
        out_shape.push(dim_size);

        let mut diagonal_stride = 0;
        for (k, other_label) in (0..x.ndim()).zip(term.chars()) {
            if label != other_label {
                continue;
            }
            if x.size(k) != dim_size {
                return Err(OpError::InvalidValue(
                    "Dimension sizes for repeated labels in term do not match",
                ));
            }
            diagonal_stride += x.stride(k);
        }
        out_strides.push(diagonal_stride);
    }

    let out_layout = DynLayout::try_from_shape_and_strides(
        &out_shape,
        &out_strides,
        OverlapPolicy::AllowOverlap,
    )
    .expect("failed to create diagonal layout");
    let out_view = TensorView::from_storage_and_layout(x.storage(), out_layout);

    Ok((unique_dims, out_view))
}

/// Evaluate a single step in an einsum path.
fn einsum_step(
    pool: &TensorPool,
    step: &EinsumStep,
    x: &TensorView,
    y: Option<&TensorView>,
) -> Result<Tensor, OpError> {
    let (lhs_term, x) = take_diagonals(&step.lhs.term, x)?;

    let (Some(y), Some(rhs)) = (y, &step.rhs) else {
        // Re-arrange input views as `[output_dims][reduced_dims]`.
        let reduced_dims = step.reduced_dims();
        let common_order: String = step
            .output
            .chars()
            .chain(reduced_dims.iter().copied())
            .collect();

        let xp = permute_and_insert_axes(&x, &lhs_term, &common_order);
        if reduced_dims.is_empty() {
            return Ok(xp.to_tensor_in(pool));
        }

        let reduced_dim_indices: Vec<i32> = (0..reduced_dims.len()).map(|i| i as i32 - 1).collect();
        return reduce_sum(
            pool,
            xp,
            Some(reduced_dim_indices.as_slice()),
            false, /* keep_dims */
        );
    };

    let (rhs_term, y) = take_diagonals(&rhs.term, y)?;
    let reduced_dims = step.reduced_dims();

    if reduced_dims.len() == 1 {
        einsum_matmul(
            pool,
            &x,
            &y,
            &lhs_term,
            &rhs_term,
            &step.output,
            reduced_dims[0],
        )
    } else {
        // Re-arrange input views as `[output_dims][reduced_dims]`. This makes
        // the reduced dimensions adjacent.
        let common_order: String = step
            .output
            .chars()
            .chain(reduced_dims.iter().copied())
            .collect();
        let xp = permute_and_insert_axes(&x, &lhs_term, &common_order);
        let yp = permute_and_insert_axes(&y, &rhs_term, &common_order);

        // If there are no reduced dimensions, fall back to a simple multiply
        // with broadcasting.
        if reduced_dims.is_empty() {
            let output = mul(pool, xp, yp)?;
            return Ok(output);
        }

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
        let term_simplified: String = step
            .output
            .chars()
            .chain(std::iter::once(reduced_dim))
            .collect();
        einsum_matmul(
            pool,
            &x.view(),
            &y.view(),
            &term_simplified,
            &term_simplified,
            &step.output,
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
    let mut out = matmul(pool, xp, yp, None)?;

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

/// Specifies the input tensor to use when processing a term in an Einsum
/// equation.
#[derive(Copy, Clone, Debug, PartialEq)]
enum EinsumInput {
    /// Use the nth input tensor, from the list of inputs for the complete
    /// einsum equation.
    Index(u32),
    /// Use the output from the previous step.
    PrevOutput,
}

/// A term in an Einsum equation which specifies the input to use and labels
/// for the dimensions.
#[derive(Clone, Debug, PartialEq)]
struct EinsumTerm {
    term: String,
    input: EinsumInput,
}

/// A processing step in an Einsum path which handles one or two terms.
#[derive(Clone, Debug, PartialEq)]
struct EinsumStep {
    lhs: EinsumTerm,
    rhs: Option<EinsumTerm>,
    output: String,
}

impl EinsumStep {
    /// Return a list of chars which appear in the input terms but not in the
    /// output of this step.
    fn reduced_dims(&self) -> Vec<char> {
        let in_terms = [
            self.lhs.term.as_str(),
            self.rhs.as_ref().map(|rhs| rhs.term.as_str()).unwrap_or(""),
        ];

        let mut terms = Vec::new();
        for in_term in in_terms {
            for in_ch in in_term.chars() {
                if !terms.contains(&in_ch) && !self.output.contains(in_ch) {
                    terms.push(in_ch);
                }
            }
        }
        terms
    }
}

/// Replace an ellipsis representing a fixed number of dimensions with a
/// sequence of numbers.
///
/// Numbers are used because they are not allowed as dimension labels in
/// input Einsum equations.
///
/// eg. `replace_ellipsis("i...j", 3)` returns `"i012j"`.
fn replace_ellipsis(term: &str, broadcast_ndim: u8) -> String {
    assert!(broadcast_ndim <= 10);

    let zero = b'0';
    if let Some((lhs, rhs)) = term.split_once("...") {
        lhs.chars()
            .chain((0..broadcast_ndim).map(|i| (zero + i) as char))
            .chain(rhs.chars())
            .collect()
    } else {
        term.to_string()
    }
}

/// Convert an Einsum expression with many inputs into a sequence of steps which
/// each processes one or two inputs.
///
/// `broadcast_ndim` specifies how many dimensions ellipses in input and output
/// terms stand for. The ellipses are replaced with digit labels in the path.
fn einsum_path(expr: &EinsumExpr, broadcast_ndim: u8) -> Vec<EinsumStep> {
    let output = replace_ellipsis(&expr.output, broadcast_ndim);
    let input_term = |term: &str, index: u32| EinsumTerm {
        term: replace_ellipsis(term, broadcast_ndim),
        input: EinsumInput::Index(index),
    };

    match &expr.inputs[..] {
        // This case shouldn't happen since Einsum equations must have at least
        // one input term.
        [] => Vec::new(),
        [term] => {
            let step = EinsumStep {
                lhs: input_term(term, 0),
                rhs: None,
                output,
            };
            [step].into()
        }
        [term_a, term_b] => {
            let step = EinsumStep {
                lhs: input_term(term_a, 0),
                rhs: Some(input_term(term_b, 1)),
                output,
            };
            [step].into()
        }
        all_terms @ [term_a, term_b, rest @ ..] => {
            let mut steps = Vec::with_capacity(all_terms.len() - 1);

            // Count how many terms use each reduced dimension.
            let mut reduced_dims: HashMap<char, usize> = expr
                .reduced_dims()
                .into_iter()
                .map(|dim| {
                    (
                        dim,
                        all_terms.iter().filter(|term| term.contains(dim)).count(),
                    )
                })
                .collect();

            // Add step for first two terms.
            for dim in term_a.chars() {
                if let Some(count) = reduced_dims.get_mut(&dim) {
                    *count -= 1;
                }
            }
            for dim in term_b.chars() {
                if let Some(count) = reduced_dims.get_mut(&dim) {
                    *count -= 1;
                }
            }

            // The output for each step consists of the unique input dim labels
            // which either appear in the final output, or are reduced
            // dimensions that appear in subsequent steps.
            let mut next_output: String = term_a
                .chars()
                .chain(term_b.chars().filter(|c| !term_a.contains(*c)))
                .filter(|dim| {
                    expr.output.contains(*dim) || reduced_dims.get(dim).copied().unwrap_or(0) > 0
                })
                .collect();

            steps.push(EinsumStep {
                lhs: input_term(term_a, 0),
                rhs: Some(input_term(term_b, 1)),
                output: next_output.clone(),
            });

            // Add a step for each remaining term.
            for (term_idx, term) in rest.iter().enumerate() {
                for dim in term.chars() {
                    if let Some(count) = reduced_dims.get_mut(&dim) {
                        *count -= 1;
                    }
                }
                let prev_output = next_output;
                if term_idx == rest.len() - 1 {
                    next_output = output.clone();
                } else {
                    next_output = prev_output
                        .chars()
                        .chain(term.chars().filter(|c| !prev_output.contains(*c)))
                        .filter(|dim| {
                            output.contains(*dim) || reduced_dims.get(dim).copied().unwrap_or(0) > 0
                        })
                        .collect();
                }
                steps.push(EinsumStep {
                    lhs: EinsumTerm {
                        term: prev_output,
                        input: EinsumInput::PrevOutput,
                    },
                    // The first two inputs are used in the first step.
                    // Each subsequent step uses one term from the input
                    // plus the output from the previous step.
                    rhs: Some(input_term(term, term_idx as u32 + 2)),
                    output: next_output.clone(),
                });
            }

            steps
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{einsum_path, EinsumExpr, EinsumInput, EinsumStep, EinsumTerm};
    use crate::ops::tests::new_pool;
    use crate::ops::{einsum, matmul, mul, reduce_sum, OpError};

    #[test]
    fn test_einsum() {
        #[derive(Debug)]
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
        let matmul_ab = matmul(&pool, mat_a.view(), mat_b.view(), None).unwrap();
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
        let square_mat = Tensor::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let cube = Tensor::arange(1., 28., None).into_shape([3, 3, 3].as_slice());

        let bhwc = mat_a
            .clone()
            .into_shape([1, 1, mat_a.size(0), mat_a.size(1)]);
        let hck = mat_b.clone().into_shape([1, mat_b.size(0), mat_b.size(1)]);

        let bhwk = matmul_ab
            .clone()
            .into_shape([1, 1, mat_a.size(0), mat_b.size(1)]);

        // 3D tensor with each dimension having a different size.
        let ijk = Tensor::zeros(&[10, 5, 8]);

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
                inputs: vec![mat_a.view(), mat_b.slice((.., 0))],
                expected: Ok(matmul(&pool, mat_a.view(), mat_b.slice((.., ..1)), None)
                    .unwrap()
                    .into_shape([mat_a.size(0)].as_slice())),
            },
            // Vector-matrix product
            Case {
                equation: "j,jk->k",
                inputs: vec![mat_a.slice(0), mat_b.view()],
                expected: Ok(matmul(&pool, mat_a.slice((..1, ..)), mat_b.view(), None)
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
                inputs: vec![mat_a.view(), mat_b.slice((.., 0))],
                expected: Ok(Tensor::from(
                    mat_a
                        .iter()
                        .zip(mat_b.slice((.., 0)).broadcast(mat_a.shape()).iter())
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
                equation: "IJ,JK", // Upper-case letters
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Err(OpError::InvalidValue("Input term is invalid")),
            },
            Case {
                equation: "i.j", // Period that is not part of an ellipsis
                inputs: vec![mat_a.view()],
                expected: Err(OpError::InvalidValue("Input term is invalid")),
            },
            Case {
                equation: "i...j...", // Multiple ellipses in a term
                inputs: vec![mat_a.view()],
                expected: Err(OpError::InvalidValue("Input term is invalid")),
            },
            // Repeated labels in input term take the diagonal.
            Case {
                equation: "ii->i",
                inputs: vec![square_mat.view()],
                expected: Ok(Tensor::from([1., 5., 9.])),
            },
            Case {
                equation: "iii->i",
                inputs: vec![cube.view()],
                expected: Ok(Tensor::from([1., 14., 27.])),
            },
            // Matrix trace
            Case {
                equation: "ii->",
                inputs: vec![square_mat.view()],
                expected: Ok(Tensor::from([1., 5., 9.].iter().sum::<f32>())),
            },
            // Repeated labels when dimensions are not the same size
            Case {
                equation: "ii->i",
                inputs: vec![mat_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Dimension sizes for repeated labels in term do not match",
                )),
            },
            // Invalid output term
            Case {
                equation: "ij,jk->IK",
                inputs: vec![mat_a.view(), mat_b.view()],
                expected: Err(OpError::InvalidValue("Output term is invalid")),
            },
            // Repeated labels in output term
            Case {
                equation: "ij->ii",
                inputs: vec![mat_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum output term contains repeated labels",
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
            Case {
                equation: "i...j",
                inputs: vec![vec_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Einsum term dimension count does not match input tensor",
                )),
            },
            // Too many dimensions in term
            Case {
                equation: "abcdefghijkl...",
                inputs: vec![TensorView::from_data([0; 12].as_slice(), &[])],
                expected: Err(OpError::UnsupportedValue(
                    "Einsum input or term has too many dimensions",
                )),
            },
            // Too many dimensions in input
            Case {
                equation: "...",
                inputs: vec![TensorView::from_data([0; 11].as_slice(), &[])],
                expected: Err(OpError::UnsupportedValue(
                    "Einsum input or term has too many dimensions",
                )),
            },
            // Three input dot product
            Case {
                equation: "i,i,i->",
                inputs: vec![vec_a.view(), vec_a.view(), vec_a.view()],
                expected: Ok(Tensor::from(vec_a.map(|x| x * x * x).iter().sum::<f32>())),
            },
            // Ellipsis for broadcasting control
            Case {
                equation: "...",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.clone()),
            },
            Case {
                equation: "i...j->i...j",
                inputs: vec![mat_a.view()],
                expected: Ok(mat_a.clone()),
            },
            Case {
                equation: "i...j->j...i",
                inputs: vec![ijk.view()],
                expected: Ok(ijk.transposed().to_tensor()),
            },
            Case {
                // Implicit output is "...ij". Ellipsis is inserted at front
                // and remaining letters are in alphabetical order.
                equation: "i...j",
                inputs: vec![ijk.view()],
                expected: Ok(ijk.permuted(&[1, 0, 2]).to_tensor()),
            },
            Case {
                equation: "...i->...",
                inputs: vec![mat_a.view()],
                expected: reduce_sum(&pool, mat_a.view(), Some(&[-1]), false /* keep_dims */),
            },
            // Mismatch of dimension count for ellipsis
            Case {
                equation: "...,...->...",
                inputs: vec![vec_a.view(), mat_a.view()],
                expected: Err(OpError::InvalidValue(
                    "Number of broadcast dims does not match across inputs",
                )),
            },
        ];

        cases.test_each(|case| {
            let Case {
                equation,
                inputs,
                expected,
            } = case;

            let pool = new_pool();
            let output = einsum(&pool, inputs.as_slice(), equation);
            assert_eq!(
                &output, expected,
                "result mismatch for equation {}",
                equation
            );
        });
    }

    #[test]
    fn test_einsum_path() {
        #[derive(Debug)]
        struct Case<'a> {
            equation: &'a str,
            broadcast_ndim: u8,
            path: Vec<EinsumStep>,
        }

        let new_term = |term: &str, index: Option<u32>| EinsumTerm {
            term: term.to_string(),
            input: index
                .map(EinsumInput::Index)
                .unwrap_or(EinsumInput::PrevOutput),
        };

        let cases = [
            // Single input term
            Case {
                equation: "i->i",
                broadcast_ndim: 0,
                path: [EinsumStep {
                    lhs: new_term("i", Some(0)),
                    rhs: None,
                    output: "i".to_string(),
                }]
                .into(),
            },
            // Two input terms
            Case {
                equation: "ij,jk->ik",
                broadcast_ndim: 0,
                path: [EinsumStep {
                    lhs: new_term("ij", Some(0)),
                    rhs: Some(new_term("jk", Some(1))),
                    output: "ik".to_string(),
                }]
                .into(),
            },
            // 3+ input terms.
            //
            // Each term has one "new" dimension and one that occurs in earlier
            // steps.
            Case {
                equation: "ab,bc,cd,de->ea",
                broadcast_ndim: 0,
                path: [
                    EinsumStep {
                        lhs: new_term("ab", Some(0)),
                        rhs: Some(new_term("bc", Some(1))),
                        output: "ac".to_string(),
                    },
                    EinsumStep {
                        lhs: new_term("ac", None),
                        rhs: Some(new_term("cd", Some(2))),
                        output: "ad".to_string(),
                    },
                    EinsumStep {
                        lhs: new_term("ad", None),
                        rhs: Some(new_term("de", Some(3))),
                        output: "ea".to_string(),
                    },
                ]
                .into(),
            },
            // 3+ input terms.
            //
            // Each input's terms are unique, so there are no reductions.
            Case {
                equation: "ab,cd,ef",
                broadcast_ndim: 0,
                path: [
                    EinsumStep {
                        lhs: new_term("ab", Some(0)),
                        rhs: Some(new_term("cd", Some(1))),
                        output: "abcd".to_string(),
                    },
                    EinsumStep {
                        lhs: new_term("abcd", None),
                        rhs: Some(new_term("ef", Some(2))),
                        output: "abcdef".to_string(),
                    },
                ]
                .into(),
            },
            // Input terms with ellipses
            Case {
                equation: "i...j->j...i",
                broadcast_ndim: 3,
                path: [EinsumStep {
                    lhs: new_term("i012j", Some(0)),
                    rhs: None,
                    output: "j012i".to_string(),
                }]
                .into(),
            },
        ];

        cases.test_each(|case| {
            let expr = EinsumExpr::parse(case.equation).unwrap();
            assert_eq!(einsum_path(&expr, case.broadcast_ndim), case.path);
        })
    }
}

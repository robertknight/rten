use std::fmt::Debug;
use std::iter::zip;

use crate::linalg::gemm;
use crate::tensor::{from_data, zero_tensor, Tensor};

mod conv;

pub use conv::{conv_2d, conv_transpose_2d};
pub use conv::{Conv2d, ConvTranspose2d, Padding};

/// Enum of the different types of input tensor that an operator can accept.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    FloatTensor(&'a Tensor<f32>),
    IntTensor(&'a Tensor<i32>),
}

impl<'a> Input<'a> {
    pub fn shape(&self) -> &'a [usize] {
        match self {
            Input::FloatTensor(t) => t.shape(),
            Input::IntTensor(t) => t.shape(),
        }
    }

    pub fn as_float(&self) -> Option<&'a Tensor<f32>> {
        if let Input::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<&'a Tensor<i32>> {
        if let Input::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }
}

impl<'a> From<&'a Tensor<f32>> for Input<'a> {
    fn from(t: &'a Tensor<f32>) -> Input {
        Input::FloatTensor(t)
    }
}

impl<'a> From<&'a Tensor<i32>> for Input<'a> {
    fn from(t: &'a Tensor<i32>) -> Input {
        Input::IntTensor(t)
    }
}

/// An Operator is a computation step in a graph.
pub trait Operator: Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator with the inputs.
    fn run(&self, input: &[Input]) -> Tensor;

    /// Return true if this operator supports in-place execution via
    /// `run_in_place`.
    ///
    /// In-place execution writes outputs to an existing tensor rather than
    /// allocating a new tensor. This can speed up execution by reducing the
    /// number of allocations during execution of a computation graph.
    fn can_run_in_place(&self) -> bool {
        false
    }

    /// Execute this operator in-place on an existing tensor.
    fn run_in_place(&self, _input: &mut Tensor) {}
}

/// Enum of all the built-in operators
pub enum OpType {
    Add,
    Concat(Concat),
    Conv2d(Conv2d),
    ConvTranspose2d(ConvTranspose2d),
    MatMul,
    MaxPool2d(MaxPool2d),
    Pad2d(Pad2d),
    ReLU,
    Reshape,
    Sigmoid,
    Slice(Slice),
}

/// Given the shapes of two inputs to a binary operation, choose the one that
/// will be used as the output shape. The other tensor will be broadcasted
/// to match.
fn choose_broadcast_shape<'a>(a: &'a [usize], b: &'a [usize]) -> &'a [usize] {
    if a.len() != b.len() {
        if a.len() < b.len() {
            b
        } else {
            a
        }
    } else if a < b {
        b
    } else {
        a
    }
}

/// Compute the result of applying the binary operation `op` to corresponding
/// elements of `a` and `b`. The shapes of `a` and `b` are broadcast to a
/// matching shape if necessary.
fn binary_op<T: Copy + Debug, F: Fn(T, T) -> T>(a: &Tensor<T>, b: &Tensor<T>, op: F) -> Tensor<T> {
    let out_shape = choose_broadcast_shape(a.shape(), b.shape());
    let a_elts = a.broadcast_elements(out_shape);
    let b_elts = b.broadcast_elements(out_shape);
    let out_data = zip(a_elts, b_elts).map(|(a, b)| op(a, b)).collect();
    from_data(out_shape.into(), out_data)
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op(a, b, |x, y| x + y)
}

#[derive(Debug)]
pub struct Add {}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn run(&self, inputs: &[Input]) -> Tensor {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        add(a, b)
    }
}

pub fn max_pool_2d(input: &Tensor, kernel_size: usize) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let out_h = in_h / kernel_size;
    let out_w = in_w / kernel_size;
    let mut output = zero_tensor::<f32>(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for chan in 0..in_c {
            let mut out_view = output.unchecked_view_mut([n, chan, 0, 0]);
            let in_view = input.unchecked_view([n, chan, 0, 0]);

            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for k_y in 0..kernel_size {
                        for k_x in 0..kernel_size {
                            let val =
                                in_view[[out_y * kernel_size + k_y, out_x * kernel_size + k_x]];
                            max_val = max_val.max(val);
                        }
                    }
                    out_view[[out_y, out_x]] = max_val;
                }
            }
        }
    }

    output
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();

    if a_cols != b_rows {
        panic!("Columns of first matrix does not match rows of second matrix")
    }

    let mut output = zero_tensor(&[a_rows, b_cols]);
    gemm(&mut output, a, b);

    output
}

#[derive(Debug)]
pub struct MatMul {}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn run(&self, inputs: &[Input]) -> Tensor {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        matmul(a, b)
    }
}

#[derive(Debug)]
pub struct MaxPool2d {
    pub kernel_size: usize,
}

impl Operator for MaxPool2d {
    fn name(&self) -> &str {
        "MaxPool2d"
    }

    /// Run `sigmoid` operator with `[input]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        max_pool_2d(input, self.kernel_size)
    }
}

pub fn relu_in_place(x: &mut Tensor) {
    for val in x.data_mut().iter_mut() {
        *val = val.max(0f32);
    }
}

pub fn relu(x: &Tensor) -> Tensor {
    x.map(|e| e.max(0f32))
}

#[derive(Debug)]
pub struct ReLU {}
impl Operator for ReLU {
    fn name(&self) -> &str {
        "ReLU"
    }

    /// Run `relu` operator with `[input]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        relu(input)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: &mut Tensor) {
        relu_in_place(input);
    }
}

#[derive(Debug)]
pub struct Reshape {}
impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        let shape = inputs[1].as_int().unwrap();
        let shape_values: Vec<_> = shape.elements().map(|e| e as usize).collect();
        input.clone_with_shape(&shape_values)
    }

    fn can_run_in_place(&self) -> bool {
        // The ability to reshape in place depends on input and target types.
        // If the planned inputs were passed to this method, we could do an
        // in-place reshape if the inputs/targets were compatible.
        false
    }
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|e| 1. / (1. + (-e).exp()))
}

pub fn sigmoid_in_place(x: &mut Tensor) {
    for val in x.data_mut().iter_mut() {
        *val = 1. / (1. + (-*val).exp());
    }
}

#[derive(Debug)]
pub struct Sigmoid {}
impl Operator for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        sigmoid(input)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: &mut Tensor) {
        sigmoid_in_place(input);
    }
}

pub fn concat(a: &Tensor, b: &Tensor, dim: usize) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != b_shape.len() {
        panic!("Tensors must have the same number of dimensions");
    }
    if dim >= a_shape.len() {
        panic!("Dimension {} is outside of range 0..{}", dim, a_shape.len());
    }
    for d in 0..a_shape.len() {
        if d != dim && a_shape[d] != b_shape[d] {
            panic!("Dimensions must be the same except for concat dim");
        }
    }

    if a_shape[dim] == 0 {
        return b.clone();
    } else if b_shape[dim] == 0 {
        return a.clone();
    }

    let mut out_data = Vec::with_capacity(a.len() + b.len());

    let a_step_size = a_shape[dim..].iter().product();
    let b_step_size = b_shape[dim..].iter().product();

    let mut a_pos = 0;
    let mut b_pos = 0;

    let mut a_elts = a.elements();
    let mut b_elts = b.elements();

    while a_pos < a.len() && b_pos < b.len() {
        out_data.extend(a_elts.by_ref().take(a_step_size));
        a_pos += a_step_size;

        out_data.extend(b_elts.by_ref().take(b_step_size));
        b_pos += b_step_size;
    }

    let mut out_shape: Vec<_> = a_shape.into();
    out_shape[dim] += b_shape[dim];

    from_data(out_shape, out_data)
}

#[derive(Debug)]
pub struct Concat {
    pub dim: usize,
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    /// Run `concat` operator with `[a, b]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        concat(a, b, self.dim)
    }
}

/// Pad an NCHW tensor in the height and width dimensions.
///
/// `padding` specifies the amount of left, top, right and bottom padding to add.
pub fn pad_2d(input: &Tensor, padding: [usize; 4]) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();

    let pad_left = padding[0];
    let pad_top = padding[1];
    let pad_right = padding[2];
    let pad_bottom = padding[3];

    let out_h = in_h + pad_top + pad_bottom;
    let out_w = in_w + pad_left + pad_right;
    let mut output = zero_tensor::<f32>(&[batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for y in pad_top..(out_h - pad_bottom) {
            for x in pad_left..(out_w - pad_right) {
                for c in 0..in_c {
                    output[[n, c, y, x]] = input[[n, c, y - pad_top, x - pad_left]];
                }
            }
        }
    }

    output
}

#[derive(Debug)]
pub struct Pad2d {
    pub padding: [usize; 4],
}

impl Operator for Pad2d {
    fn name(&self) -> &str {
        "Pad2d"
    }

    /// Run `pad` operator with `[input]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        pad_2d(input, self.padding)
    }
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice(input: &Tensor, dim: usize, start: usize, end: usize) -> Tensor {
    let mut out_shape: Vec<_> = input.shape().into();
    out_shape[dim] = end - start;

    let out_len = out_shape.iter().sum();
    let mut out_data = Vec::with_capacity(out_len);

    let dim_stride = input.stride(dim);
    let steps = if dim == 0 {
        1
    } else {
        input.shape()[0..dim].iter().product()
    };
    let parent_dim_stride = if dim == 0 {
        input.len()
    } else {
        input.stride(dim - 1)
    };

    let elts: Vec<f32> = input.elements().collect();
    for i in 0..steps {
        let offset = i * parent_dim_stride + start * dim_stride;
        let len = (end - start) * dim_stride;
        out_data.extend_from_slice(&elts[offset..offset + len]);
    }

    from_data(out_shape, out_data)
}

#[derive(Debug)]
pub struct Slice {
    pub dim: usize,
    pub start: usize,
    pub end: usize,
}

impl Operator for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    /// Run `slice` operator with `[input]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        slice(input, self.dim, self.start, self.end)
    }

    fn can_run_in_place(&self) -> bool {
        self.start == 0
    }

    fn run_in_place(&self, input: &mut Tensor) {
        input.resize_dim(self.dim, self.end);
    }
}

// Expectated values of operations in tests should be computed from the
// corresponding operations in PyTorch, since that is the framework being used
// to train the models that will initially be executed with this library.
#[cfg(test)]
mod tests {
    use crate::linalg::gemm;
    use crate::ops::{
        add, concat, matmul, max_pool_2d, pad_2d, relu, relu_in_place, sigmoid, sigmoid_in_place,
        slice, Operator, Reshape,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, random_tensor, zero_tensor};
    use crate::test_util::expect_equal;

    #[test]
    fn test_add() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        let result = add(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_add_broadcasted() -> Result<(), String> {
        // Simple case where comparing ordering of tensor shapes tells us
        // target shape.
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![1], vec![10.]);
        let expected = from_data(vec![2, 2], vec![11., 12., 13., 14.]);
        let result = add(&a, &b);
        expect_equal(&result, &expected)?;

        // Try alternative ordering for inputs.
        let result = add(&b, &a);
        expect_equal(&result, &expected)?;

        // Case where the length of tensor shapes needs to be compared before
        // the ordering, since ([5] > [1,5]).
        let a = from_data(vec![5], vec![1., 2., 3., 4., 5.]);
        let b = from_data(vec![1, 5], vec![1., 2., 3., 4., 5.]);
        let expected = from_data(vec![1, 5], vec![2., 4., 6., 8., 10.]);

        let result = add(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_matmul() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = random_tensor(&[3, 10], &mut rng);
        let b = random_tensor(&[10, 8], &mut rng);

        let mut expected = zero_tensor(&[3, 8]);
        gemm(&mut expected, &a, &b);

        let result = matmul(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_max_pool_2d() -> Result<(), String> {
        let height = 4;
        let width = 8;
        let mut input = zero_tensor(&[1, 1, height, width]);

        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;

        input[[0, 0, 0, 2]] = 0.1;
        input[[0, 0, 0, 3]] = 0.2;
        input[[0, 0, 1, 2]] = 0.3;
        input[[0, 0, 1, 3]] = 0.4;

        let expected = from_data(
            vec![1, 1, 2, 4],
            vec![4.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let result = max_pool_2d(&input, 2);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_relu() -> Result<(), String> {
        let input = from_data(vec![2, 2, 1], vec![-0.5, 0.5, 3.0, -5.5]);
        let expected = from_data(vec![2, 2, 1], vec![0.0, 0.5, 3.0, 0.0]);

        let result = relu(&input);
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        relu_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_reshape() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_data(vec![1], vec![4]);
        let expected = input.clone_with_shape(&[4]);

        let op = Reshape {};
        let result = op.run(&[(&input).into(), (&shape).into()]);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_sigmoid() -> Result<(), String> {
        let input = from_data(
            vec![9],
            vec![-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0],
        );
        let expected = from_data(
            vec![9],
            vec![
                0.0000, 0.0474, 0.2689, 0.3775, 0.5000, 0.6225, 0.7311, 0.9526, 1.0000,
            ],
        );

        let result = sigmoid(&input);
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        sigmoid_in_place(&mut result);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_concat() -> Result<(), String> {
        let a = from_data(vec![2, 2, 1], vec![0.1, 0.2, 0.3, 0.4]);
        let b = from_data(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);

        // Test concatenation along the first dimension
        let expected = from_data(vec![4, 2, 1], vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0]);
        let result = concat(&a, &b, 0);
        expect_equal(&result, &expected)?;

        // Test concatenation along a non-first dimension
        let expected = from_data(vec![2, 2, 2], vec![0.1, 1.0, 0.2, 2.0, 0.3, 3.0, 0.4, 4.0]);
        let result = concat(&a, &b, 2);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_pad_2d() -> Result<(), String> {
        let input = from_data(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = from_data(
            vec![1, 1, 4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let result = pad_2d(&input, [1, 1, 1, 1]);
        expect_equal(&result, &expected)?;

        let result = pad_2d(&input, [0, 0, 0, 0]);
        expect_equal(&result, &input)
    }

    #[test]
    fn test_slice_not_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[2, 2, 5, 3], &mut rng);

        let dim = 2;
        let start = 2;
        let end = 4;

        let sliced = slice(&input, dim, start, end);
        let shape = sliced.shape();

        assert_eq!(sliced.shape(), vec![2, 2, end - start, 3]);
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(sliced[[w, x, y, z]], input[[w, x, y + start, z]]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[5, 2, 5, 3], &mut rng);

        let dim = 0;
        let start = 2;
        let end = 4;

        let sliced = slice(&input, dim, start, end);
        let shape = sliced.shape();

        assert_eq!(shape, vec![end - start, 2, 5, 3]);
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(sliced[[w, x, y, z]], input[[w + start, x, y, z]]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_noop() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[5, 2, 5, 3], &mut rng);

        for dim in 0..input.shape().len() {
            let sliced = slice(&input, dim, 0, input.shape()[dim]);
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }
}

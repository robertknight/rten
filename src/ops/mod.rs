use std::fmt::Debug;
use std::iter::zip;

use crate::linalg::{gemm, gemm_slice, Matrix};
use crate::tensor::{from_data, from_scalar, zero_tensor, Tensor};

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

/// Enum of the different types of output tensor that an operator can produce.
pub enum Output {
    FloatTensor(Tensor<f32>),
    IntTensor(Tensor<i32>),
}

impl Output {
    pub fn as_int(self) -> Option<Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_int_ref(&self) -> Option<&Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(&t)
        } else {
            None
        }
    }

    pub fn as_float(self) -> Option<Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_float_ref(&self) -> Option<&Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(&t)
        } else {
            None
        }
    }
}

impl<'a> From<Tensor<f32>> for Output {
    fn from(t: Tensor<f32>) -> Output {
        Output::FloatTensor(t)
    }
}

impl<'a> From<Tensor<i32>> for Output {
    fn from(t: Tensor<i32>) -> Output {
        Output::IntTensor(t)
    }
}

/// An Operator is a computation step in a graph.
pub trait Operator: Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator with the inputs.
    fn run(&self, input: &[Input]) -> Output;

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
    ///
    /// `input` is the first input, which the implementation may modify and
    /// return as the output. `other` are the remaining inputs.
    ///
    /// The default implementation just returns the input without modifying it.
    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        input
    }
}

/// Enum of all the built-in operators
pub enum OpType {
    Add,
    BatchNormalization(BatchNormalization),
    Clip(Clip),
    Concat(Concat),
    Conv2d(Conv2d),
    ConvTranspose2d(ConvTranspose2d),
    Gather(Gather),
    Gemm(Gemm),
    GlobalAveragePool,
    MatMul,
    MaxPool2d(MaxPool2d),
    Mul,
    Pad2d(Pad2d),
    ReLU,
    Reshape,
    Shape,
    Sigmoid,
    Slice,
    Unsqueeze(Unsqueeze),
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

/// Return true if an elementwise binary operation can be performed in-place
/// on `a` given `b` as the other argument.
fn can_run_binary_op_in_place<T: Copy>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()
}

/// Perform an elementwise binary operation in-place.
///
/// This currently only supports the case where both inputs have exactly the
/// same shape, so no broadcasting is required, and the inputs are contigious.
fn binary_op_in_place<T: Copy + Debug, F: Fn(&mut T, T)>(a: &mut Tensor<T>, b: &Tensor<T>, op: F) {
    assert!(a.is_contiguous());
    assert!(b.is_contiguous());
    for (a_elt, b_elt) in zip(a.data_mut().iter_mut(), b.data().iter()) {
        op(a_elt, *b_elt);
    }
}

/// Perform elementwise addition of two tensors.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op(a, b, |x, y| x + y)
}

/// Perform in-place elementwise addition of two tensors.
pub fn add_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |x, y| *x += y);
}

#[derive(Debug)]
pub struct Add {}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        add(a, b).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Output {
        let mut a = input.as_float().unwrap();
        let b = other[0].as_float().unwrap();

        if can_run_binary_op_in_place(&a, &b) {
            add_in_place(&mut a, &b);
            a.into()
        } else {
            self.run(&[(&a).into(), b.into()])
        }
    }
}

pub fn clip(input: &Tensor, min: f32, max: f32) -> Tensor {
    input.map(|x| x.max(min).min(max))
}

pub fn clip_in_place(input: &mut Tensor, min: f32, max: f32) {
    for val in input.data_mut().iter_mut() {
        *val = val.max(min).min(max)
    }
}

/// Perform in-place batch normalization on the NCHW tensor `out`.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization
pub fn batch_norm_in_place(
    out: &mut Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) {
    let [batch, chans, in_h, in_w] = out.dims();
    for n in 0..batch {
        for c in 0..chans {
            let chan_mean = mean[[c]];
            let chan_var = var[[c]];
            let chan_scale = scale[[c]];
            let chan_bias = bias[[c]];

            for y in 0..in_h {
                for x in 0..in_w {
                    let mut el = &mut out[[n, c, y, x]];
                    *el = (*el - chan_mean) / (chan_var + epsilon).sqrt() * chan_scale + chan_bias;
                }
            }
        }
    }
}

/// Perform batch normalization on the NCHW tensor `input`.
///
/// See https://github.com/onnx/onnx/blob/main/docs/Operators.md#batchnormalization
pub fn batch_norm(
    input: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Tensor {
    let mut output = input.clone();
    batch_norm_in_place(&mut output, scale, bias, mean, var, epsilon);
    output
}

#[derive(Debug)]
pub struct BatchNormalization {
    pub epsilon: f32,
}

impl Operator for BatchNormalization {
    fn name(&self) -> &str {
        "BatchNormalization"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        let scale = inputs[1].as_float().unwrap();
        let bias = inputs[2].as_float().unwrap();
        let mean = inputs[3].as_float().unwrap();
        let var = inputs[4].as_float().unwrap();
        batch_norm(input, scale, bias, mean, var, self.epsilon).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        let scale = other[0].as_float().unwrap();
        let bias = other[1].as_float().unwrap();
        let mean = other[2].as_float().unwrap();
        let var = other[3].as_float().unwrap();

        batch_norm_in_place(&mut output, scale, bias, mean, var, self.epsilon);

        output.into()
    }
}

#[derive(Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        clip(input, self.min, self.max).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        clip_in_place(&mut output, self.min, self.max);
        output.into()
    }
}

/// Gather elements from `input` specified by `indices`.
///
/// This currently only supports one common use case for Gather operators,
/// which is to index into a vector with a scalar.
pub fn gather<T: Copy + Default>(
    input: &Tensor<T>,
    axis: usize,
    indices: &Tensor<i32>,
) -> Tensor<T> {
    match (input.shape().len(), axis, indices.item()) {
        (1, 0, Some(index)) => from_scalar(input[[index as usize]]),
        _ => panic!("Gather operator only supports indexing into a 1D tensor with a scalar"),
    }
}

#[derive(Debug)]
pub struct Gather {
    pub axis: usize,
}

impl Operator for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let indices = inputs[1].as_int().unwrap();
        match inputs[0] {
            Input::IntTensor(input) => gather(input, self.axis, &indices).into(),
            Input::FloatTensor(input) => gather(input, self.axis, &indices).into(),
        }
    }
}

#[derive(Debug)]
pub struct Gemm {
    pub alpha: f32,
    pub beta: f32,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// Compute the General Matrix Multiplication (GEMM) `c = alpha * (ab) + beta * c`.
///
/// If `transpose_a` or `transpose_b` are set, the `a` and `b` inputs
/// respectively are transposed before multiplying them.
///
/// nb. This is named `gemm_op` to avoid confusion with `linalg::gemm`.
pub fn gemm_op(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
) -> Tensor {
    if alpha != 1.0 {
        panic!("Gemm only supports `alpha` value of 1.0");
    }
    if beta != 0.0 && beta != 1.0 {
        panic!("Gemm only supports `beta` values of 0.0 and 1.0");
    }

    let (a_rows, a_cols, a_row_stride, a_col_stride) = if transpose_a {
        (a.shape()[1], a.shape()[0], a.stride(1), a.stride(0))
    } else {
        (a.shape()[0], a.shape()[1], a.stride(0), a.stride(1))
    };
    let (b_rows, b_cols, b_row_stride, b_col_stride) = if transpose_b {
        (b.shape()[1], b.shape()[0], b.stride(1), b.stride(0))
    } else {
        (b.shape()[0], b.shape()[1], b.stride(0), b.stride(1))
    };

    let out_shape = &[a_rows, b_cols][..];
    let mut output = if c.is_some() && beta == 1.0 {
        let out_data = c.unwrap().broadcast_elements(out_shape).collect();
        from_data(out_shape.into(), out_data)
    } else {
        zero_tensor(out_shape)
    };

    let out_row_stride = output.stride(0);

    gemm_slice(
        output.data_mut(),
        out_row_stride,
        Matrix {
            data: a.data(),
            rows: a_rows,
            cols: a_cols,
            row_stride: a_row_stride,
            col_stride: a_col_stride,
        },
        Matrix {
            data: b.data(),
            rows: b_rows,
            cols: b_cols,
            row_stride: b_row_stride,
            col_stride: b_col_stride,
        },
    );

    output
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        let c = inputs.get(2).map(|c| c.as_float().unwrap());
        gemm_op(
            &a,
            &b,
            c,
            self.alpha,
            self.beta,
            self.transpose_a,
            self.transpose_b,
        )
        .into()
    }
}

pub fn global_average_pool(input: &Tensor) -> Tensor {
    let [batch, chans, in_h, in_w] = input.dims();
    let mut output = zero_tensor(&[batch, chans, 1, 1]);

    let hw_float = (in_h * in_w) as f32;

    for n in 0..batch {
        for c in 0..chans {
            let in_view = input.unchecked_view([n, c, 0, 0]);
            let mut sum = 0.0;
            for y in 0..in_h {
                for x in 0..in_w {
                    sum += in_view[[y, x]];
                }
            }
            output[[n, c, 0, 0]] = sum / hw_float;
        }
    }

    output
}

#[derive(Debug)]
pub struct GlobalAveragePool {}

impl Operator for GlobalAveragePool {
    fn name(&self) -> &str {
        "GlobalAveragePool"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        global_average_pool(input).into()
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

    fn run(&self, inputs: &[Input]) -> Output {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        matmul(a, b).into()
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
    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        max_pool_2d(input, self.kernel_size).into()
    }
}

/// Multiply two tensors elementwise.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op(a, b, |x, y| x * y)
}

/// Perform in-place elementwise multiplication of two tensors.
pub fn mul_in_place(a: &mut Tensor, b: &Tensor) {
    binary_op_in_place(a, b, |a_elt, b_elt| *a_elt *= b_elt);
}

#[derive(Debug)]
pub struct Mul {}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let a = inputs[0].as_float().unwrap();
        let b = inputs[1].as_float().unwrap();
        mul(a, b).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Output {
        let mut a = input.as_float().unwrap();
        let b = other[0].as_float().unwrap();

        if can_run_binary_op_in_place(&a, &b) {
            mul_in_place(&mut a, &b);
            a.into()
        } else {
            self.run(&[(&a).into(), b.into()])
        }
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
    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        relu(input).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        relu_in_place(&mut output);
        output.into()
    }
}

pub fn reshape<T: Copy>(input: &Tensor<T>, shape: &Tensor<i32>) -> Tensor<T> {
    // If exactly one of the new shape's dimensions is -1, infer the size
    // from the input length and the sizes of the other dimensions.
    let mut unspecified_dim = None;
    let mut specified_dims_size = 1;
    for (dim, size) in shape.elements().enumerate() {
        if size < -1 {
            panic!("Invalid dimension size {} in shape", size);
        } else if size != -1 {
            specified_dims_size *= size as usize;
        } else if unspecified_dim.is_some() {
            panic!("Multiple dimensions in new shape set to -1");
        } else {
            unspecified_dim = Some(dim);
        }
    }
    let (unspecified_dim_size, remainder) = match input.len() {
        0 => (0, 0),
        _ => (
            input.len() / specified_dims_size,
            input.len() % specified_dims_size,
        ),
    };
    if remainder != 0 {
        panic!("Input length must be a multiple of specified dimensions");
    }

    let complete_shape: Vec<_> = shape
        .elements()
        .map(|size| match size {
            -1 => unspecified_dim_size,
            valid => valid as usize,
        })
        .collect();

    input.clone_with_shape(&complete_shape)
}

#[derive(Debug)]
pub struct Reshape {}
impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        let shape = inputs[1].as_int().unwrap();
        reshape(&input, &shape).into()
    }

    fn can_run_in_place(&self) -> bool {
        // The ability to reshape in place depends on input and target types.
        // If the planned inputs were passed to this method, we could do an
        // in-place reshape if the inputs/targets were compatible.
        false
    }
}

#[derive(Debug)]
pub struct Shape {}

impl Operator for Shape {
    fn name(&self) -> &str {
        "Shape"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        let shape = from_data(
            vec![input.shape().len()],
            input.shape().iter().map(|&el| el as i32).collect(),
        );
        shape.into()
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

    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        sigmoid(input).into()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        sigmoid_in_place(&mut output);
        output.into()
    }
}

pub fn concat<T: Copy>(a: &Tensor<T>, b: &Tensor<T>, dim: usize) -> Tensor<T> {
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
    fn run(&self, inputs: &[Input]) -> Output {
        let a = inputs[0];
        let b = inputs[1];

        match (a, b) {
            (Input::FloatTensor(a), Input::FloatTensor(b)) => concat(a, b, self.dim).into(),
            (Input::IntTensor(a), Input::IntTensor(b)) => concat(a, b, self.dim).into(),
            _ => panic!("Incompatible input tensor types for Concat"),
        }
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
    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0].as_float().unwrap();
        pad_2d(input, self.padding).into()
    }
}

fn slice_ranges(
    input_shape: &[usize],
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> =
        input_shape.iter().map(|dim_size| (0, *dim_size)).collect();
    for (i, (start, end)) in zip(starts.elements(), ends.elements()).enumerate() {
        let axis = if let Some(axes) = axes {
            axes[[i]] as usize
        } else {
            i
        };
        ranges[axis] = (start as usize, end as usize);
    }
    ranges
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice<T: Copy>(
    input: &Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Tensor<T> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes);
    let sliced_data = input.slice_elements(&ranges).collect();
    let sliced_shape = ranges.iter().map(|(start, end)| end - start).collect();
    from_data(sliced_shape, sliced_data).into()
}

/// Clip the dimensions of the input tensor specified by `axes` to the ranges
/// given by `starts` and `ends`. If `axes` is
/// not set, dimensions
pub fn slice_in_place<T: Copy>(
    input: &mut Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) {
    let ranges = slice_ranges(input.shape(), starts, ends, axes);
    for (dim, (start, end)) in ranges.iter().copied().enumerate() {
        input.clip_dim(dim, start, end);
    }
}

#[derive(Debug)]
pub struct Slice {}

impl Operator for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    /// Run `slice` operator with `[input, starts, ends, axes]` inputs.
    fn run(&self, inputs: &[Input]) -> Output {
        let input = inputs[0];
        let starts = inputs[1].as_int().unwrap();
        let ends = inputs[2].as_int().unwrap();
        let axes = inputs.get(3).map(|t| t.as_int().unwrap());
        match input {
            Input::FloatTensor(input) => slice(input, starts, ends, axes).into(),
            Input::IntTensor(input) => slice(input, starts, ends, axes).into(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Output {
        let mut output = input.as_float().unwrap();
        let starts = other[0].as_int().unwrap();
        let ends = other[1].as_int().unwrap();
        let axes = other.get(2).map(|t| t.as_int().unwrap());
        slice_in_place(&mut output, starts, ends, axes);
        output.into()
    }
}

pub fn unsqueeze<T: Copy>(input: &Tensor<T>, axes: &[usize]) -> Tensor<T> {
    let mut new_shape: Vec<_> = input.shape().iter().copied().collect();
    let mut sorted_axes: Vec<_> = axes.iter().collect();
    sorted_axes.sort();
    for &axis in sorted_axes {
        new_shape.insert(axis, 1);
    }
    input.clone_with_shape(&new_shape)
}

#[derive(Debug)]
pub struct Unsqueeze {
    pub axes: Vec<usize>,
}

impl Operator for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn run(&self, inputs: &[Input]) -> Output {
        match inputs[0] {
            Input::FloatTensor(input) => unsqueeze(&input, &self.axes).into(),
            Input::IntTensor(input) => unsqueeze(&input, &self.axes).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::gemm;
    use crate::ops::{
        add, add_in_place, batch_norm, batch_norm_in_place, clip, clip_in_place, concat, gather,
        gemm_op, global_average_pool, matmul, max_pool_2d, mul, mul_in_place, pad_2d, relu,
        relu_in_place, reshape, sigmoid, sigmoid_in_place, slice, slice_in_place, unsqueeze, Add,
        Operator, Output, Reshape, Shape,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_scalar, from_vec, random_tensor, zero_tensor, Tensor};
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
        expect_equal(&result, &expected)?;

        // Case where one of the inputs is a scalar.
        let a = from_scalar(3.0);
        let b = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = add(&a, &b);
        let expected = from_data(vec![2, 2], vec![4.0, 5.0, 6.0, 7.0]);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_add_in_place() -> Result<(), String> {
        // Invoke `add_in_place` directly.
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let a_copy = a.clone();
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![11., 22., 33., 44.]);
        add_in_place(&mut a, &b);
        expect_equal(&a, &expected)?;

        // Run `Add` operator in place with inputs that support in-place addition.
        let op = Add {};
        let result = op.run_in_place(Output::FloatTensor(a_copy), &[(&b).into()]);
        expect_equal(result.as_float_ref().unwrap(), &expected)?;

        // Run `Add` operator in-place with inputs that don't support in-place
        // addition. The operator should fall back to creating a new output tensor.
        let scalar = from_scalar(1.0);
        let expected = from_data(vec![2, 2], vec![11., 21., 31., 41.]);
        let result = op.run_in_place(Output::FloatTensor(scalar), &[(&b).into()]);
        expect_equal(result.as_float_ref().unwrap(), &expected)
    }

    #[test]
    fn test_batch_norm() -> Result<(), String> {
        let input = from_data(vec![1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(vec![2], vec![3.0, 3.0]);
        let bias = from_data(vec![2], vec![0.1, 0.2]);
        let mean = from_data(vec![2], vec![0.5, -0.5]);
        let var = from_data(vec![2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(vec![1, 2, 1, 1], vec![y1, y2]);
        let result = batch_norm(&input, &scale, &bias, &mean, &var, epsilon);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_batch_norm_in_place() -> Result<(), String> {
        let mut input = from_data(vec![1, 2, 1, 1], vec![1.0, 2.0]);
        let scale = from_data(vec![2], vec![3.0, 3.0]);
        let bias = from_data(vec![2], vec![0.1, 0.2]);
        let mean = from_data(vec![2], vec![0.5, -0.5]);
        let var = from_data(vec![2], vec![1.0, 2.0]);

        let epsilon = 1e-5 as f32;

        let y1 = (input[[0, 0, 0, 0]] - mean[[0]]) / (var[[0]] + epsilon).sqrt() * scale[[0]]
            + bias[[0]];
        let y2 = (input[[0, 1, 0, 0]] - mean[[1]]) / (var[[1]] + epsilon).sqrt() * scale[[1]]
            + bias[[1]];
        let expected = from_data(vec![1, 2, 1, 1], vec![y1, y2]);

        batch_norm_in_place(&mut input, &scale, &bias, &mean, &var, epsilon);

        expect_equal(&input, &expected)
    }

    #[test]
    fn test_clip() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let expected = from_data(vec![2, 2], vec![1., 1., 3., 5.]);
        let result = clip(&input, 1.0, 5.0);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_clip_in_place() -> Result<(), String> {
        let mut input = from_data(vec![2, 2], vec![-5., -2., 3., 20.]);
        let expected = from_data(vec![2, 2], vec![1., 1., 3., 5.]);
        clip_in_place(&mut input, 1.0, 5.0);
        expect_equal(&input, &expected)
    }

    #[test]
    fn test_gather() {
        // We currently support only one common use of Gather, which is to
        // index into a vector with a scalar, eg. to extract one dimension from
        // a tensor shape.
        let input = from_vec(vec![1, 20, 30]);
        let indices = from_scalar(1);
        let result = gather(&input, 0, &indices);
        assert_eq!(result.item(), Some(20))
    }

    #[test]
    fn test_gemm_op() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = random_tensor(&[3, 10], &mut rng);
        let b = random_tensor(&[10, 8], &mut rng);

        let mut expected = zero_tensor(&[3, 8]);
        gemm(&mut expected, &a, &b);

        let result = gemm_op(&a, &b, None, 1.0, 1.0, false, false);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = random_tensor(&[10, 3], &mut rng);
        let b = random_tensor(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = zero_tensor(&[3, 8]);
        gemm(&mut expected, &a_transposed, &b_transposed);

        let result = gemm_op(&a, &b, None, 1.0, 1.0, true, true);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = random_tensor(&[3, 10], &mut rng);
        let b = random_tensor(&[10, 8], &mut rng);
        let c = random_tensor(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm(&mut expected, &a, &b);

        let result = gemm_op(&a, &b, Some(&c), 1.0, 1.0, false, false);

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_global_average_pool() -> Result<(), String> {
        let input = from_data(vec![1, 2, 2, 2], vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let expected = from_data(vec![1, 2, 1, 1], vec![2.5, 25.]);
        let result = global_average_pool(&input);
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
    fn test_mul() -> Result<(), String> {
        let a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        let result = mul(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_mul_in_place() -> Result<(), String> {
        let mut a = from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = from_data(vec![2, 2], vec![10., 20., 30., 40.]);
        let expected = from_data(vec![2, 2], vec![10., 40., 90., 160.]);
        mul_in_place(&mut a, &b);
        expect_equal(&a, &expected)
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
    fn test_reshape_with_unspecified_dim() -> Result<(), String> {
        // Reshape with an unspecified (-1) dim and nonzero-length input
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![1, -1, 2]);
        let expected = input.clone_with_shape(&[1, 2, 2]);
        let result = reshape(&input, &shape);
        expect_equal(&result, &expected)?;

        // Reshape with an unspecified (-1) dim and zero-length input
        let zero_sized_input = from_data(vec![4, 0, 1], vec![]);
        let shape = from_vec(vec![100, -1]);
        let result = reshape(&zero_sized_input, &shape);
        let expected = zero_sized_input.clone_with_shape(&[100, 0]);
        expect_equal(&result, &expected)
    }

    #[test]
    #[should_panic(expected = "Multiple dimensions in new shape set to -1")]
    fn test_reshape_with_multiple_unspecified_dims() {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![1, -1, -1]);
        reshape(&input, &shape);
    }

    #[test]
    #[should_panic(expected = "Input length must be a multiple of specified dimensions")]
    fn test_reshape_with_unsolvable_unspecified_dim() {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![5, -1]);
        reshape(&input, &shape);
    }

    #[test]
    fn test_reshape_op() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_data(vec![1], vec![4]);
        let expected = input.clone_with_shape(&[4]);

        let op = Reshape {};
        let result = op
            .run(&[(&input).into(), (&shape).into()])
            .as_float()
            .unwrap();

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
    fn test_shape() {
        let input = from_data(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        let op = Shape {};
        let result = op.run(&[(&input).into()]).as_int().unwrap();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data(), &[1, 1, 2, 2]);
    }

    fn from_slice<T: Copy>(data: &[T]) -> Tensor<T> {
        from_data(vec![data.len()], data.into())
    }

    #[test]
    fn test_slice_in_place() {
        let mut rng = XorShiftRNG::new(5678);
        let mut input = random_tensor(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        slice_in_place(&mut input, &starts, &ends, Some(&axes));

        assert_eq!(
            input.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
    }

    #[test]
    fn test_slice_not_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        let sliced = slice(&input, &starts, &ends, Some(&axes));
        let shape = sliced.shape();

        assert_eq!(
            sliced.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w, x, y + starts[[0]] as usize, z]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[5, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[0]);

        let sliced = slice(&input, &starts, &ends, Some(&axes));
        let shape = sliced.shape();

        assert_eq!(
            shape,
            vec![ends[[0]] as usize - starts[[0]] as usize, 2, 5, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w + starts[[0]] as usize, x, y, z]]
                        );
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
            let dim_size = input.shape()[dim] as i32;

            let starts = from_slice(&[0]);
            let ends = from_slice(&[dim_size]);
            let axes = from_slice(&[dim as i32]);

            let sliced = slice(&input, &starts, &ends, Some(&axes));
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }

    #[test]
    fn test_unsqueeze() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[3, 4, 5], &mut rng);

        // Unsqueeze with axes in increasing order
        let output = unsqueeze(&input, &[0, 4]);
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze with axes in decreasing order
        let output = unsqueeze(&input, &[4, 0]);
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze a scalar into a 1-item vec
        let scalar = from_scalar(2.0);
        let output = unsqueeze(&scalar, &[0]);
        assert_eq!(output.shape(), &[1]);
        assert_eq!(output.data(), &[2.0]);
    }
}

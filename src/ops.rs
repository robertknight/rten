use std::fmt::Debug;
use std::iter::zip;

use crate::gemm::{gemm, gemm_slice};
use crate::tensor::{from_data, zero_tensor, Tensor};

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

/// Unroll patches from an image into a matrix.
///
/// The input has shape NCHW. The result has shape (GHW)xP where G is the subset
/// of image channels from `start_chan` to `end_chan` and P is the number of
/// patches that the padded input divides into.
fn im2col(
    output: &mut Tensor,
    input: &Tensor,
    image_index: usize,
    patch_h: usize,
    patch_w: usize,
    start_chan: usize,
    end_chan: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let [_, out_w] = output.dims();
    let [_, _, in_h, in_w] = input.dims();
    let y_patches = (in_h + pad_h * 2) - (patch_h - 1);
    let x_patches = (in_w + pad_w * 2) - (patch_w - 1);
    let n_chans = end_chan - start_chan;

    for c in 0..n_chans {
        // The loop ordering here is chosen to maximize the number of
        // consecutive steps that we read/write the same rows of the inputs and
        // outputs. This is more efficient assuming the tensors are stored in
        // row-major order.
        for py in 0..y_patches {
            let out_col_left = py * x_patches;

            // Calculate range of kernel rows that will lead to valid input
            // row coordinates. For other rows zero padding is used, meaning
            // the output will be zero.
            let min_ky = pad_h.saturating_sub(py);
            let max_ky = (in_h + pad_h).saturating_sub(py).min(patch_h);

            for k_y in min_ky..max_ky {
                let img_y = py + k_y;
                let out_row_top = c * patch_h * patch_w + k_y * patch_w;
                let in_row =
                    input.last_dim_slice([image_index, start_chan + c, img_y - pad_h, 0], in_w);

                for k_x in 0..patch_w {
                    let out_row = out_row_top + k_x;

                    // Calculate range of patches that will lead to valid input
                    // column coordinates. For other patches zero padding is used,
                    // meaning the output will be zero.
                    let min_px = pad_w.saturating_sub(k_x);
                    let max_px = (in_w + pad_w).saturating_sub(k_x).min(x_patches);
                    let out_row_data =
                        &mut output.last_dim_slice_mut([out_row, 0], out_w)[out_col_left..];

                    for px in min_px..max_px {
                        out_row_data[px] = in_row[px + k_x - pad_w]
                    }
                }
            }
        }
    }
}

/// Unroll a subset of channels from a convolution kernel into a matrix of shape
/// O x (CHW) where O is the number of output channels and C is the number of
/// input channels.
fn unroll_kernel(output: &mut Tensor, kernel: &Tensor, out_chan_start: usize, out_chan_end: usize) {
    let [_, k_in_c, k_h, k_w] = kernel.dims();
    for out_c in out_chan_start..out_chan_end {
        for in_c in 0..k_in_c {
            for y in 0..k_h {
                for x in 0..k_w {
                    let out_row = out_c - out_chan_start;
                    let out_col = in_c * (k_h * k_w) + y * k_w + x;
                    output[[out_row, out_col]] = kernel[[out_c, in_c, y, x]];
                }
            }
        }
    }
}

/// Convert a matrix of image patches back into an image.
///
/// The output tensor has shape NCHW. The input image has shape GP where G is a
/// subset of channels C given by `start_chan..start_chan+G` and P is the number
/// of patches.
fn col2im(
    output: &mut Tensor,
    input: &Tensor,
    image_index: usize,
    start_chan: usize,
    y_patches: usize,
    x_patches: usize,
) {
    let [group_chans, n_patches] = input.dims();

    for c in 0..group_chans {
        let mut out_view = output.unchecked_view_mut([image_index, start_chan + c, 0, 0]);
        let in_row = input.last_dim_slice([c, 0], n_patches);

        for y in 0..y_patches {
            for x in 0..x_patches {
                let patch = y * x_patches + x;
                out_view[[y, x]] = in_row[patch];
            }
        }
    }
}

/// Add per-channel biases to an NCHW tensor. Bias is a C-length vector.
fn add_channel_bias(output: &mut Tensor, bias: &Tensor) {
    let [batch, chans, height, width] = output.dims();

    for n in 0..batch {
        for c in 0..chans {
            let mut out_view = output.unchecked_view_mut([n, c, 0, 0]);
            let chan_bias = bias[[c]];

            for y in 0..height {
                for x in 0..width {
                    out_view[[y, x]] += chan_bias;
                }
            }
        }
    }
}

/// Specialization of conv_2d for pointwise convolutions over one image. This
/// can be reduced to tensor reshaping and matrix multiplication.
fn conv_2d_pointwise(input: &Tensor, kernel: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let [_, _, in_h, in_w] = input.dims();
    let [out_c, in_c, _, _] = kernel.dims();

    let mut output = zero_tensor(&[out_c, in_h * in_w]);
    let out_row_stride = output.stride(0);

    // Use the low-level gemm_slice API to simplicitly reshape the input and
    // kernel to `in_c x in_h*in_w` and `out_c x in_c` matrices respectively.
    //
    // If this package supported creating reshaped views of existing tensors,
    // we could use that instead.
    gemm_slice(
        output.data_mut(),
        out_row_stride,
        kernel.data(),
        out_c, /* a rows */
        in_c,  /* a columns */
        in_c,  /* a row stride */
        input.data(),
        in_c,        /* b rows */
        in_h * in_w, /* b columns */
        in_h * in_w, /* b row stride */
    );

    output.reshape(&[1, out_c, in_h, in_w]);

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

/// Specialization of conv_2d for depthwise convolutions.
///
/// Depthwise convolutions operate over a single input/output channel at
/// a time and hence the transformation of convolution to matrix multiplication
/// doesn't pay off. An optimized direct method works better.
fn conv_2d_depthwise(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, _, k_h, k_w] = kernel.dims();
    let (pad_h, pad_w) = padding;

    let out_h = in_h - k_h + 1 + 2 * pad_h;
    let out_w = in_w - k_w + 1 + 2 * pad_w;

    let mut output = zero_tensor::<f32>(&[batch, out_c, out_h, out_w]);

    for n in 0..batch {
        for c in 0..in_c {
            let kernel_view = kernel.unchecked_view([c, 0, 0, 0]);

            // The loops here are ordered so that the inner-most loop is as
            // efficient as possible and runs for as long as possible over a
            // contiguous slice of memory.
            for out_y in 0..out_h {
                let out_row = output.last_dim_slice_mut([n, c, out_y, 0], out_w);

                for k_y in 0..k_h {
                    let in_y = out_y + k_y;
                    if in_y < pad_h || in_y >= in_h + pad_h {
                        continue;
                    }
                    let in_row = input.last_dim_slice([n, c, in_y - pad_h, 0], in_w);

                    for k_x in 0..k_w {
                        let kernel_val = kernel_view[[k_y, k_x]];

                        // Calculate range of out X coords that are in 0..out_w
                        // and map to valid input X coords.
                        let min_out_x = pad_w.saturating_sub(k_x);
                        let max_out_x = (in_w + pad_w).saturating_sub(k_x).min(out_w);

                        let out_row_slice = &mut out_row[min_out_x..max_out_x];
                        let in_row_slice =
                            &in_row[min_out_x + k_x - pad_w..max_out_x + k_x - pad_w];

                        for i in 0..out_row_slice.len() {
                            out_row_slice[i] += in_row_slice[i] * kernel_val;
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

/// Perform a 2D convolution of `input` with `kernel`.
///
/// `input` has dimensions NCHW while `kernel` has OGHW where `G` is `C / groups`.
///
/// - `padding` specifies the amount of horizontal and vertical padding respectively
///   that is added to each side.
/// - `groups` controls which input and output channels are convolved. It must
///   be a positive integer that divides the input and output channel count.
///   A value of 1 convolves every input channel with every output channel.
///   A value of 2 convolves each half of the input channels with the corresponding
///   half of the output channels.
///   A value equal to the input channel count convolves each input channel
///   separately with `output_channels / groups` outputs. This is known as
///   depthwise convolution.
pub fn conv_2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    padding: (usize, usize),
    groups: usize,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, k_in_c, k_h, k_w] = kernel.dims();
    let (pad_h, pad_w) = padding;

    if batch == 1 && k_h == 1 && k_w == 1 && pad_h == 0 && pad_w == 0 && groups == 1 {
        return conv_2d_pointwise(input, kernel, bias);
    }

    let out_channels_per_group = out_c / groups;
    let in_channels_per_group = in_c / groups;

    if in_channels_per_group != k_in_c {
        panic!(
            "Input channels (per group) {} does not match kernel input channels {}",
            in_channels_per_group, k_in_c
        );
    }

    if groups == 0 || in_c % groups != 0 || out_c % groups != 0 {
        panic!(
            "Input channels {} and output channels {} must be divisible by group count {}",
            in_c, out_c, groups
        );
    }

    if in_c == out_c && groups == in_c {
        return conv_2d_depthwise(input, kernel, bias, padding);
    }

    let out_h = in_h - k_h + 1 + 2 * pad_h;
    let out_w = in_w - k_w + 1 + 2 * pad_w;

    let y_patches = (in_h + pad_h * 2) - (k_h - 1);
    let x_patches = (in_w + pad_w * 2) - (k_w - 1);

    let n_patches = y_patches * x_patches;
    let mut im2col_mat = zero_tensor(&[in_channels_per_group * k_h * k_w, n_patches]);
    let mut output = zero_tensor(&[batch, out_c, out_h, out_w]);
    let mut kernel_mat = zero_tensor(&[out_channels_per_group, in_channels_per_group * k_h * k_w]);
    let mut output_mat = zero_tensor(&[out_channels_per_group, n_patches]);

    for n in 0..batch {
        for group in 0..groups {
            let in_chan_start = group * in_channels_per_group;
            let in_chan_end = in_chan_start + in_channels_per_group;
            let out_chan_start = group * out_channels_per_group;
            let out_chan_end = out_chan_start + out_channels_per_group;

            // Perform convolution for group. This uses an indirect method,
            // where image patches and the kernel are first packed into
            // matrices. The matrices are multiplied, and the results unpacked
            // into the output tensor.
            im2col(
                &mut im2col_mat,
                input,
                n,
                k_h,
                k_w,
                in_chan_start,
                in_chan_end,
                pad_h,
                pad_w,
            );
            unroll_kernel(&mut kernel_mat, kernel, out_chan_start, out_chan_end);
            gemm(&mut output_mat, &kernel_mat, &im2col_mat);
            col2im(
                &mut output,
                &output_mat,
                n,
                out_chan_start,
                y_patches,
                x_patches,
            );

            // `gemm` currently accumulates into the output buffer, so we need
            // to clear it between iterations.
            output_mat.data_mut().fill(0.0);
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

#[derive(Copy, Clone, Debug)]
pub enum Padding {
    /// Apply enough padding such that the output and input have the same size.
    Same,

    /// Apply an even amount of padding to the start and end of the height and
    /// width dimensions respectively.
    Fixed((usize, usize)),
}

/// Calculate the specific amount of padding required for an operation that
/// will receive an NCHW input tensor and apply a kernel of a given size.
///
/// The kernel size must be >= the input size.
fn calc_fixed_padding(
    pad: Padding,
    input_shape: &[usize],
    kernel_size: (usize, usize),
) -> (usize, usize) {
    match pad {
        Padding::Fixed(pads) => pads,
        Padding::Same => {
            let [_, _, in_h, in_w]: [usize; 4] = input_shape.try_into().unwrap();
            let (k_h, k_w) = kernel_size;

            let unpadded_h = (in_h - k_h) + 1;
            let unpadded_w = (in_w - k_w) + 1;

            let pad_h = (in_h - unpadded_h) / 2;
            let pad_w = (in_w - unpadded_w) / 2;

            (pad_h, pad_w)
        }
    }
}

#[derive(Debug)]
pub struct Conv2d {
    pub padding: Padding,
    pub groups: usize,
}

impl Operator for Conv2d {
    fn name(&self) -> &str {
        "Conv2d"
    }

    /// Run `conv_2d` operator with `[input, weight, bias?]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        let weight = inputs[1].as_float().unwrap();
        let bias = inputs.get(2).map(|t| t.as_float().unwrap());

        let [_, _, k_h, k_w] = weight.dims();

        conv_2d(
            input,
            weight,
            bias,
            calc_fixed_padding(self.padding, input.shape(), (k_h, k_w)),
            self.groups,
        )
    }
}

// Compute `dest += src * scale`, where `dest_stride` is the distance between
// elements of `dest` to update.
fn add_scaled_vector_strided(dest: &mut [f32], src: &[f32], dest_stride: usize, scale: f32) {
    // Check size once to skip bounds check on each loop iteration below.
    if (src.len() - 1) * dest_stride >= dest.len() {
        panic!("Dest vector is too small");
    }

    const N: usize = 4;
    let n_blocks = src.len() / N;
    let mut val = [0.0; N];

    for b in 0..n_blocks {
        for i in 0..N {
            unsafe {
                val[i] = src.get_unchecked(b * N + i) * scale;
            }
        }

        for i in 0..N {
            unsafe {
                *dest.get_unchecked_mut((b * N + i) * dest_stride) += val[i];
            }
        }
    }

    for i in n_blocks * N..src.len() {
        unsafe {
            *dest.get_unchecked_mut(i * dest_stride) += src[i] * scale;
        }
    }
}

/// Perform a transposed 2D convolution of a tensor by a kernel.
///
/// `input` has dimensions NCHW and kernel has dimensions COHW where `O` is
/// the number of output channels.
pub fn conv_transpose_2d(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [k_in_c, out_c, k_h, k_w] = kernel.dims();

    if in_c != k_in_c {
        panic!(
            "Input channels {} does not match kernel input channels {}",
            in_c, k_in_c
        )
    }

    let out_h = (in_h - 1) * stride + k_h;
    let out_w = (in_w - 1) * stride + k_w;

    let mut output = zero_tensor::<f32>(&[batch, out_c, out_h, out_w]);

    for n in 0..batch {
        for out_chan in 0..out_c {
            for in_chan in 0..in_c {
                let kernel_view = kernel.unchecked_view([in_chan, out_chan, 0, 0]);

                for in_y in 0..in_h {
                    let in_row = input.last_dim_slice([n, in_chan, in_y, 0], in_w);

                    for k_y in 0..k_h {
                        let out_y = in_y * stride + k_y;
                        let out_row = output.last_dim_slice_mut([n, out_chan, out_y, 0], out_w);

                        for k_x in 0..k_w {
                            add_scaled_vector_strided(
                                &mut out_row[k_x..],
                                in_row,
                                stride,
                                kernel_view[[k_y, k_x]],
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        add_channel_bias(&mut output, bias);
    }

    output
}

#[derive(Debug)]
pub struct ConvTranspose2d {
    pub stride: usize,
}

impl Operator for ConvTranspose2d {
    fn name(&self) -> &str {
        "ConvTranspose2d"
    }

    /// Run `conv_2d` operator with `[input, weight]` inputs.
    fn run(&self, inputs: &[Input]) -> Tensor {
        let input = inputs[0].as_float().unwrap();
        let weight = inputs[1].as_float().unwrap();
        let bias = inputs.get(2).map(|t| t.as_float().unwrap());
        conv_transpose_2d(input, weight, bias, self.stride)
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
    use crate::gemm::gemm;
    use crate::ops::{
        add, concat, conv_2d, conv_transpose_2d, matmul, max_pool_2d, pad_2d, relu, relu_in_place,
        sigmoid, sigmoid_in_place, slice, Conv2d, Operator, Padding, Reshape,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, random_tensor, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    /// Un-optimized reference implementation of convolution.
    fn reference_conv(
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        padding: (usize, usize),
        groups: usize,
    ) -> Tensor {
        let [batch, in_chans, in_h, in_w] = input.dims();
        let [out_chans, k_in_chans, k_h, k_w] = kernel.dims();
        let (pad_h, pad_w) = padding;

        let in_channels_per_group = in_chans / groups;
        let out_channels_per_group = out_chans / groups;
        assert_eq!(in_channels_per_group, k_in_chans);

        let out_h = in_h - k_h + 1 + 2 * pad_h;
        let out_w = in_w - k_w + 1 + 2 * pad_w;

        let mut output = zero_tensor(&[batch, out_chans, out_h, out_w]);

        for n in 0..batch {
            for group in 0..groups {
                let in_chan_start = group * in_channels_per_group;
                let in_chan_end = in_chan_start + in_channels_per_group;
                let out_chan_start = group * out_channels_per_group;
                let out_chan_end = out_chan_start + out_channels_per_group;

                for out_chan in out_chan_start..out_chan_end {
                    let chan_bias = if let Some(bias) = bias {
                        bias[[out_chan]]
                    } else {
                        0.0
                    };
                    for out_y in 0..out_h {
                        for out_x in 0..out_w {
                            let mut accum = 0.0;
                            for in_chan in in_chan_start..in_chan_end {
                                for k_y in 0..k_h {
                                    for k_x in 0..k_w {
                                        let in_y = out_y + k_y;
                                        let in_x = out_x + k_x;

                                        if in_y >= pad_h
                                            && in_y < in_h + pad_h
                                            && in_x >= pad_w
                                            && in_x < in_w + pad_w
                                        {
                                            accum += input
                                                [[n, in_chan, in_y - pad_h, in_x - pad_w]]
                                                * kernel
                                                    [[out_chan, in_chan - in_chan_start, k_y, k_x]];
                                        }
                                    }
                                }
                            }
                            output[[n, out_chan, out_y, out_x]] = accum + chan_bias;
                        }
                    }
                }
            }
        }

        output
    }

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

    /// Basic tests for conv_2d. These compare the results against values
    /// computed from PyTorch as well as the reference implementation.
    #[test]
    fn test_conv_2d() -> Result<(), String> {
        let kernel = from_data(
            vec![1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = from_data(
            vec![1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let expected_with_same_padding = from_data(
            vec![1, 1, 3, 3],
            vec![
                1.5202, 1.5592, 0.9939, 1.7475, 2.6358, 1.3428, 1.0165, 1.1806, 0.8685,
            ],
        );

        let result = conv_2d(&input, &kernel, None, (1, 1), 1 /* groups */);
        let reference_result = reference_conv(&input, &kernel, None, (1, 1), 1 /* groups */);
        expect_equal(&result, &expected_with_same_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_no_padding = from_data(vec![1, 1, 1, 1], vec![2.6358]);

        let result = conv_2d(&input, &kernel, None, (0, 0), 1 /* groups */);
        let reference_result = reference_conv(&input, &kernel, None, (0, 0), 1 /* groups */);
        expect_equal(&result, &expected_with_no_padding)?;
        expect_equal(&result, &reference_result)?;

        let expected_with_bias = from_data(vec![1, 1, 1, 1], vec![3.6358]);
        let bias = from_data(vec![1], vec![1.0]);
        let result = conv_2d(&input, &kernel, Some(&bias), (0, 0), 1 /* groups */);
        let reference_result =
            reference_conv(&input, &kernel, Some(&bias), (0, 0), 1 /* groups */);
        expect_equal(&result, &expected_with_bias)?;
        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_2d_same_padding() -> Result<(), String> {
        let kernel = &from_data(
            vec![1, 1, 3, 3],
            vec![
                0.3230, 0.7632, 0.4616, 0.8837, 0.5898, 0.3424, 0.2101, 0.7821, 0.6861,
            ],
        );

        let input = &from_data(
            vec![1, 1, 3, 3],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 0.8862,
            ],
        );

        let op = Conv2d {
            padding: Padding::Same,
            groups: 1,
        };
        let result = op.run(&[input.into(), kernel.into()]);
        let reference_result = reference_conv(input, kernel, None, (1, 1), 1 /* groups */);

        expect_equal(&result, &reference_result)
    }

    // Specific tests for convolutions with a 1x1 kernel.
    #[test]
    fn test_conv_2d_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[10, 5, 1, 1], &mut rng);
        let input = random_tensor(&[1, 5, 20, 20], &mut rng);

        let result = conv_2d(&input, &kernel, None, (0, 0), 1 /* groups */);
        let reference_result = reference_conv(&input, &kernel, None, (0, 0), 1 /* groups */);

        assert_eq!(result.shape(), [1, 10, 20, 20]);
        expect_equal(&result, &reference_result)
    }

    // Specific tests for convolutions that operate over one output channel and
    // one input channel at a time.
    #[test]
    fn test_conv_2d_depthwise() -> Result<(), String> {
        let input = from_data(
            vec![1, 3, 2, 2],
            vec![
                0.5946, 0.8249, 0.0448, 0.9552, 0.2041, 0.2501, 0.2693, 0.1007, 1.5202, 1.5592,
                0.9939, 1.7475,
            ],
        );
        let kernel = from_data(
            vec![3, 1, 2, 2],
            vec![
                -0.0862, -0.4111, 0.0813, 0.4993, -0.4641, 0.1715, -0.0532, -0.2429, -0.4325,
                0.4273, 0.4180, 0.4338,
            ],
        );
        let expected = from_data(vec![1, 3, 1, 1], vec![0.09020272, -0.09061745, 1.1822754]);
        let reference_result = reference_conv(&input, &kernel, None, (0, 0), 3 /* groups */);

        let result = conv_2d(&input, &kernel, None, (0, 0), 3 /* groups */);

        expect_equal(&result, &expected)?;
        expect_equal(&result, &reference_result)
    }

    // Tests for convolutions that are neither pointwise nor depthwise. In
    // other words, the kernel has a spatial size > 1x1 and a channel depth > 1.
    #[test]
    fn test_conv_2d_not_depthwise_or_pointwise() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let kernel = random_tensor(&[4, 3, 3, 3], &mut rng);
        let input = random_tensor(&[2, 3, 20, 20], &mut rng);

        let result = conv_2d(&input, &kernel, None, (1, 1), 1 /* groups */);
        let reference_result = reference_conv(&input, &kernel, None, (1, 1), 1 /* groups */);

        expect_equal(&result, &reference_result)
    }

    #[test]
    fn test_conv_transpose_2d() -> Result<(), String> {
        let input = from_data(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = from_data(vec![1, 1, 2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let expected = from_data(
            vec![1, 1, 4, 4],
            vec![
                0.1000, 0.2000, 0.2000, 0.4000, 0.3000, 0.4000, 0.6000, 0.8000, 0.3000, 0.6000,
                0.4000, 0.8000, 0.9000, 1.2000, 1.2000, 1.6000,
            ],
        );

        let result = conv_transpose_2d(&input, &kernel, None, 2);
        expect_equal(&result, &expected)?;

        let mut expected_with_bias = from_data(expected.shape().into(), expected.data().into());
        for i in 0..expected_with_bias.len() {
            expected_with_bias.data_mut()[i] += 1.234;
        }
        let bias = from_data(vec![1], vec![1.234]);
        let result = conv_transpose_2d(&input, &kernel, Some(&bias), 2);
        expect_equal(&result, &expected_with_bias)
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

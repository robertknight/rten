use crate::tensor::{from_data, zero_tensor, Tensor};

/// An Operator is a computation step in a graph.
pub trait Operator {
    /// Execute the operator with the inputs.
    fn run(&self, input: &[&Tensor]) -> Tensor;
}

/// Enum of all the built-in operators
pub enum OpType {
    Concat(Concat),
    Conv2d(Conv2d),
    ConvTranspose2d(ConvTranspose2d),
    MaxPool2d(MaxPool2d),
    Pad2d(Pad2d),
    ReLU,
    Sigmoid,
    Slice(Slice),
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

    let (pad_h, pad_w) = padding;
    let out_h = in_h - k_h + 1 + 2 * pad_h;
    let out_w = in_w - k_w + 1 + 2 * pad_w;

    let mut output = zero_tensor::<f32>(vec![batch, out_c, out_h, out_w]);
    for n in 0..batch {
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                for group in 0..groups {
                    let in_chan_start = group * in_channels_per_group;
                    let in_chan_end = in_chan_start + in_channels_per_group;
                    let out_chan_start = group * out_channels_per_group;
                    let out_chan_end = out_chan_start + out_channels_per_group;

                    for out_chan in out_chan_start..out_chan_end {
                        for k_y in 0..k_h {
                            for k_x in 0..k_w {
                                let in_y = out_y + k_y;
                                let in_x = out_x + k_x;

                                if in_y < pad_h || in_x < pad_w {
                                    continue;
                                }

                                let in_y = in_y - pad_h;
                                let in_x = in_x - pad_w;

                                if in_y >= in_h || in_x >= in_w {
                                    continue;
                                }

                                for in_chan in in_chan_start..in_chan_end {
                                    output[[n, out_chan, out_y, out_x]] += input
                                        [[n, in_chan, in_y, in_x]]
                                        * kernel[[out_chan, in_chan - in_chan_start, k_y, k_x]];
                                }
                            }
                        }
                    }
                }

                if let Some(bias) = bias {
                    for c in 0..out_c {
                        output[[n, c, out_y, out_x]] += bias[[c]]
                    }
                }
            }
        }
    }
    output
}

pub struct Conv2d {
    pub padding: (usize, usize),
    pub groups: usize,
}

impl Operator for Conv2d {
    /// Run `conv_2d` operator with `[input, weight, bias?]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = inputs[0];
        let weight = inputs[1];
        let bias = if inputs.len() > 2 {
            Some(inputs[2])
        } else {
            None
        };
        conv_2d(input, weight, bias, self.padding, self.groups)
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

    let mut output = zero_tensor::<f32>(vec![batch, out_c, out_h, out_w]);

    for n in 0..batch {
        for in_y in 0..in_h {
            for in_x in 0..in_w {
                for in_chan in 0..in_c {
                    for k_y in 0..k_h {
                        for k_x in 0..k_w {
                            let out_y = in_y * stride + k_y;
                            let out_x = in_x * stride + k_x;

                            for out_chan in 0..out_c {
                                output[[n, out_chan, out_y, out_x]] += input
                                    [[n, in_chan, in_y, in_x]]
                                    * kernel[[in_chan, out_chan, k_y, k_x]];
                            }
                        }
                    }
                }
            }
        }

        if let Some(bias) = bias {
            for c in 0..out_c {
                for h in 0..out_h {
                    for w in 0..out_w {
                        output[[n, c, h, w]] += bias[[c]];
                    }
                }
            }
        }
    }

    output
}

pub struct ConvTranspose2d {
    pub stride: usize,
}

impl Operator for ConvTranspose2d {
    /// Run `conv_2d` operator with `[input, weight]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        let weight = &inputs[1];
        let bias = if inputs.len() > 2 {
            Some(inputs[2])
        } else {
            None
        };
        conv_transpose_2d(input, weight, bias, self.stride)
    }
}

pub fn max_pool_2d(input: &Tensor, kernel_size: usize) -> Tensor {
    let [batch, in_c, in_h, in_w] = input.dims();
    let out_h = in_h / kernel_size;
    let out_w = in_w / kernel_size;
    let mut output = zero_tensor::<f32>(vec![batch, in_c, out_h, out_w]);

    for n in 0..batch {
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                for chan in 0..in_c {
                    let mut max_val = input[[n, chan, out_y * kernel_size, out_x * kernel_size]];
                    for k_y in 0..kernel_size {
                        for k_x in 0..kernel_size {
                            let val = input[[
                                n,
                                chan,
                                out_y * kernel_size + k_y,
                                out_x * kernel_size + k_x,
                            ]];
                            max_val = max_val.max(val);
                        }
                    }
                    output[[n, chan, out_y, out_x]] = max_val;
                }
            }
        }
    }

    output
}

pub struct MaxPool2d {
    pub kernel_size: usize,
}

impl Operator for MaxPool2d {
    /// Run `sigmoid` operator with `[input]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        max_pool_2d(input, self.kernel_size)
    }
}

pub fn relu(x: &Tensor) -> Tensor {
    x.map(|e| e.max(0f32))
}

pub struct ReLU {}
impl Operator for ReLU {
    /// Run `relu` operator with `[input]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        relu(input)
    }
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|e| 1. / (1. + (-e).exp()))
}

pub struct Sigmoid {}
impl Operator for Sigmoid {
    /// Run `sigmoid` operator with `[input]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        sigmoid(input)
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

    let mut out_shape: Vec<_> = a_shape.into();
    out_shape[dim] += b_shape[dim];

    let mut output = zero_tensor::<f32>(out_shape);

    let a_stride = if dim == 0 { a.len() } else { a.stride(dim - 1) };
    let b_stride = if dim == 0 { b.len() } else { b.stride(dim - 1) };

    let mut a_pos = 0;
    let mut b_pos = 0;
    let mut out_pos = 0;

    let a_data = a.data();
    let b_data = b.data();

    while a_pos < a_data.len() && b_pos < b_data.len() {
        for _ in 0..a_stride {
            output.data_mut()[out_pos] = a_data[a_pos];
            out_pos += 1;
            a_pos += 1;
        }
        for _ in 0..b_stride {
            output.data_mut()[out_pos] = b_data[b_pos];
            out_pos += 1;
            b_pos += 1;
        }
    }

    output
}

pub struct Concat {
    pub dim: usize,
}

impl Operator for Concat {
    /// Run `concat` operator with `[a, b]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let a = &inputs[0];
        let b = &inputs[1];
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
    let mut output = zero_tensor::<f32>(vec![batch, in_c, out_h, out_w]);

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

pub struct Pad2d {
    pub padding: [usize; 4],
}

impl Operator for Pad2d {
    /// Run `pad` operator with `[input]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        pad_2d(input, self.padding)
    }
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice(input: &Tensor, dim: usize, start: usize, end: usize) -> Tensor {
    let mut out_shape: Vec<_> = input.shape().into();
    out_shape[dim] = end - start;

    let out_len = out_shape.iter().fold(0, |sum, x| sum + x);
    let mut out_data = Vec::with_capacity(out_len);

    let dim_stride = input.stride(dim);
    let steps = if dim == 0 {
        1
    } else {
        input.shape()[0..dim].iter().fold(1, |steps, x| steps * x)
    };
    let parent_dim_stride = if dim == 0 {
        input.len()
    } else {
        input.stride(dim - 1)
    };

    for i in 0..steps {
        let offset = i * parent_dim_stride + start * dim_stride;
        let len = (end - start) * dim_stride;
        out_data.extend_from_slice(&input.data()[offset..offset + len]);
    }

    from_data(out_shape, out_data)
}

pub struct Slice {
    pub dim: usize,
    pub start: usize,
    pub end: usize,
}

impl Operator for Slice {
    /// Run `slice` operator with `[input]` inputs.
    fn run(&self, inputs: &[&Tensor]) -> Tensor {
        let input = &inputs[0];
        slice(input, self.dim, self.start, self.end)
    }
}

// Expectated values of operations in tests should be computed from the
// corresponding operations in PyTorch, since that is the framework being used
// to train the models that will initially be executed with this library.
#[cfg(test)]
mod tests {
    use crate::ops::{
        concat, conv_2d, conv_transpose_2d, max_pool_2d, pad_2d, relu, sigmoid, slice,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, random_tensor, zero_tensor, Tensor};

    /// Check that the shapes of two tensors are equal and that their contents
    /// are approximately equal.
    fn expect_equal(x: &Tensor, y: &Tensor) -> Result<(), String> {
        if x.shape() != y.shape() {
            return Err(format!(
                "Tensors have different shapes. {:?} vs. {:?}",
                x.shape(),
                y.shape()
            ));
        }

        let eps = 0.001;
        for i in 0..x.len() {
            let xi = x.data()[i];
            let yi = y.data()[i];

            if (xi - yi).abs() > eps {
                return Err(format!(
                    "Tensor values differ at index {}: {} vs {}",
                    i, xi, yi
                ));
            }
        }

        return Ok(());
    }

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
        expect_equal(&result, &expected_with_same_padding)?;

        let expected_with_no_padding = from_data(vec![1, 1, 1, 1], vec![2.6358]);

        let result = conv_2d(&input, &kernel, None, (0, 0), 1 /* groups */);
        expect_equal(&result, &expected_with_no_padding)?;

        let expected_with_bias = from_data(vec![1, 1, 1, 1], vec![3.6358]);
        let bias = from_data(vec![1], vec![1.0]);
        let result = conv_2d(&input, &kernel, Some(&bias), (0, 0), 1 /* groups */);
        expect_equal(&result, &expected_with_bias)
    }

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

        let result = conv_2d(&input, &kernel, None, (0, 0), 3 /* groups */);

        expect_equal(&result, &expected)
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
    fn test_max_pool_2d() -> Result<(), String> {
        let height = 4;
        let width = 8;
        let mut input = zero_tensor(vec![1, 1, height, width]);

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
        let input = random_tensor(vec![2, 2, 5, 3], &mut rng);

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
        let input = random_tensor(vec![5, 2, 5, 3], &mut rng);

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
        let input = random_tensor(vec![5, 2, 5, 3], &mut rng);

        for dim in 0..input.shape().len() {
            let sliced = slice(&input, dim, 0, input.shape()[dim]);
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }
}

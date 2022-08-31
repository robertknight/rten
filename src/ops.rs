use crate::tensor::{dims3, dims4, zero_tensor, Tensor};

/// Perform a 2D convolution of `input` with `kernel`.
///
/// `input` has dimensions HWC and kernel has dimensions HWOC where `O` is
/// the number of output channels.
///
/// This is a reference implementation which uses a naive direct convolution
/// algorithm.
pub fn conv_2d(input: &Tensor, kernel: &Tensor, padding: (usize, usize)) -> Tensor {
    let (in_h, in_w, in_c) = dims3(input);
    let (k_h, k_w, out_c, k_in_c) = dims4(kernel);

    if in_c != k_in_c {
        panic!(
            "Input channels {} does not match kernel input channels {}",
            in_c, k_in_c
        )
    }

    let (pad_h, pad_w) = padding;
    let out_h = in_h - k_h + 1 + 2 * pad_h;
    let out_w = in_w - k_w + 1 + 2 * pad_w;

    let mut output = zero_tensor(vec![out_h, out_w, out_c]);
    for out_y in 0..out_h {
        for out_x in 0..out_w {
            for out_chan in 0..out_c {
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

                        for in_chan in 0..in_c {
                            output[[out_y, out_x, out_chan]] += input[[in_y, in_x, in_chan]]
                                * kernel[[k_y, k_x, out_chan, in_chan]];
                        }
                    }
                }
            }
        }
    }
    output
}

/// Perform a transposed 2D convolution of a tensor by a kernel.
///
/// `input` has dimensions HWC and kernel has dimensions HWOC where `O` is
/// the number of output channels.
pub fn conv_transpose_2d(input: &Tensor, kernel: &Tensor, stride: usize) -> Tensor {
    let (in_h, in_w, in_c) = dims3(input);
    let (k_h, k_w, out_c, k_in_c) = dims4(kernel);

    if in_c != k_in_c {
        panic!(
            "Input channels {} does not match kernel input channels {}",
            in_c, k_in_c
        )
    }

    let out_h = (in_h - 1) * stride + k_h;
    let out_w = (in_w - 1) * stride + k_w;

    let mut output = zero_tensor(vec![out_h, out_w, out_c]);

    for in_y in 0..in_h {
        for in_x in 0..in_w {
            for in_chan in 0..in_c {
                for k_y in 0..k_h {
                    for k_x in 0..k_w {
                        let out_y = in_y * stride + k_y;
                        let out_x = in_x * stride + k_x;

                        for out_chan in 0..out_c {
                            output[[out_y, out_x, out_chan]] += input[[in_y, in_x, in_chan]]
                                * kernel[[k_y, k_x, out_chan, in_chan]];
                        }
                    }
                }
            }
        }
    }

    output
}

pub fn max_pool_2d(input: &Tensor, kernel_size: usize) -> Tensor {
    let (in_h, in_w, in_c) = dims3(input);
    let out_h = in_h / kernel_size;
    let out_w = in_w / kernel_size;
    let mut output = zero_tensor(vec![out_h, out_w, in_c]);
    for out_y in 0..out_h {
        for out_x in 0..out_w {
            for chan in 0..in_c {
                let mut max_val = input[[out_y, out_x, chan]];
                for k_y in 0..kernel_size {
                    for k_x in 0..kernel_size {
                        let val =
                            input[[out_y * kernel_size + k_y, out_x * kernel_size + k_x, chan]];
                        max_val = max_val.max(val);
                    }
                }
                output[[out_y, out_x, chan]] = max_val;
            }
        }
    }
    output
}

pub fn relu(x: &Tensor) -> Tensor {
    x.map(|e| e.max(0f32))
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|e| 1. / (1. + (-e).exp()))
}

pub fn concat(a: &Tensor, b: &Tensor, dim: usize) -> Tensor {
    let a_shape = &a.shape;
    let b_shape = &b.shape;

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

    let mut out_shape = a_shape.clone();
    out_shape[dim] += b_shape[dim];

    let mut output = zero_tensor(out_shape);

    let a_stride = a.stride(dim);
    let b_stride = b.stride(dim);

    let mut a_pos = 0;
    let mut b_pos = 0;
    let mut out_pos = 0;

    while a_pos < a.data.len() && b_pos < b.data.len() {
        for i in 0..a_stride {
            output.data[out_pos] = a.data[a_pos];
            out_pos += 1;
            a_pos += 1;
        }
        for i in 0..b_stride {
            output.data[out_pos] = b.data[b_pos];
            out_pos += 1;
            b_pos += 1;
        }
    }

    output
}

/// Pad an HWC tensor in the height and width dimensions.
///
/// `padding` specifies the amount of left, top, right and bottom padding to add.
pub fn pad_2d(input: &Tensor, padding: [usize; 4]) -> Tensor {
    let (in_h, in_w, in_c) = dims3(input);

    let pad_left = padding[0];
    let pad_top = padding[1];
    let pad_right = padding[2];
    let pad_bottom = padding[3];

    let out_h = in_h + pad_top + pad_bottom;
    let out_w = in_w + pad_left + pad_right;
    let mut output = zero_tensor(vec![out_h, out_w, in_c]);

    for y in pad_top..(out_h - pad_bottom) {
        for x in pad_left..(out_w - pad_right) {
            for c in 0..in_c {
                output[[y, x, c]] = input[[y - pad_top, x - pad_left, c]];
            }
        }
    }

    output
}

use crate::tensor::{dims3, dims4, zero_tensor, Tensor};

/// Perform a 2D convolution of `input` with `kernel`.
///
/// `input` has dimensions HWC and kernel has dimensions HWOC where `O` is
/// the number of output channels.
///
/// This is a reference implementation which uses a naive direct convolution
/// algorithm.
pub fn conv2d_direct(input: &Tensor, kernel: &Tensor, padding: (usize, usize)) -> Tensor {
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
                        if in_y <= pad_h || in_y > in_h - pad_h {
                            continue;
                        }
                        if in_x <= pad_w || in_x > in_w - pad_w {
                            continue;
                        }
                        let in_y = in_y - pad_h;
                        let in_x = in_x - pad_w;

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

use crate::ops::{Input, Operator, Output};
use crate::tensor::{zero_tensor, Tensor};

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

#[cfg(test)]
mod tests {
    use crate::ops::{global_average_pool, max_pool_2d};
    use crate::tensor::{from_data, zero_tensor};
    use crate::test_util::expect_equal;

    #[test]
    fn test_global_average_pool() -> Result<(), String> {
        let input = from_data(vec![1, 2, 2, 2], vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let expected = from_data(vec![1, 2, 1, 1], vec![2.5, 25.]);
        let result = global_average_pool(&input);
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
}

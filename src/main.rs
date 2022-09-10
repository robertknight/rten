use std::time::SystemTime;

mod ops;
mod rng;
mod tensor;

use ops::{concat, conv_2d, max_pool_2d, pad_2d, sigmoid};
use rng::XorShiftRNG;
use tensor::random_tensor;

fn main() {
    let mut rng = XorShiftRNG::new(123u64);
    let input = random_tensor(vec![800, 600, 1], &mut rng);

    let kernel = random_tensor(vec![3, 3, 1, 1], &mut rng);
    let point_kernel = random_tensor(vec![1, 1, 8, 1], &mut rng);

    let start = SystemTime::now();
    let output = conv_2d(&input, &kernel, (1, 1), 1);
    let output = conv_2d(&input, &point_kernel, (0, 0), 1);
    let end = SystemTime::now();

    let pooled = max_pool_2d(&output, 2);
    let result = sigmoid(&output);

    println!("Output shape {:?}", &output.shape);
    println!("Pooled shape {:?}", &pooled.shape);
    println!("Output val {}", output[[500, 500, 0]]);
    println!("Elapsed {:?}", end.duration_since(start));

    let a = random_tensor(vec![10, 20, 1], &mut rng);
    let b = pad_2d(&a, [1, 2, 3, 4]);
    println!("in shape {:?} padded shape {:?}", &a.shape, &b.shape);
}

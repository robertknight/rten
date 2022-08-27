use std::time::SystemTime;

mod ops;
mod rng;
mod tensor;

use ops::{conv2d_direct, sigmoid};
use rng::XorShiftRNG;
use tensor::random_tensor;

fn main() {
    let mut rng = XorShiftRNG::new(123u64);
    let input = random_tensor(vec![800, 600, 1], &mut rng);
    // let input = random_tensor(vec![16, 16, 1], &mut rng);

    let kernel = random_tensor(vec![3, 3, 1, 1], &mut rng);
    let point_kernel = random_tensor(vec![1, 1, 8, 1], &mut rng);

    let start = SystemTime::now();
    let output = conv2d_direct(&input, &kernel, (1, 1));
    let output = conv2d_direct(&input, &point_kernel, (0, 0));
    let end = SystemTime::now();

    let result = sigmoid(&output);

    println!("Output shape {:?}", &output.shape);
    println!("Output val {}", output[[500, 500, 0]]);
    println!("Elapsed {:?}", end.duration_since(start));
}

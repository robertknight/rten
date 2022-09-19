use std::fs;

mod graph;
mod model;
mod ops;
mod rng;
mod tensor;

#[allow(dead_code, unused_imports)]
mod schema_generated;

#[cfg(test)]
mod model_builder;

use model::load_model;
use rng::XorShiftRNG;
use tensor::random_tensor;

fn main() {
    let model_bytes = fs::read("output.model").unwrap();
    let model = load_model(&model_bytes).unwrap();
    let input_id = model.find_node("input.1").unwrap();
    let output_id = model.find_node("380").unwrap();

    let mut rng = XorShiftRNG::new(1234);
    let img_data = random_tensor(vec![1, 1, 800, 600], &mut rng);
    let text_mask = model.run(&[(input_id, &img_data)], &[output_id]);

    println!("Mask shape {:?}", text_mask[0].shape);
}

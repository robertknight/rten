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
    let model_bytes = [0u8, 1u8, 2u8, 3u8];
    let mut rng = XorShiftRNG::new(1234);

    let model = load_model(&model_bytes).unwrap();
    let input_id = model.find_node("input").unwrap();
    let output_id = model.find_node("output").unwrap();
    let img_data = random_tensor(vec![800, 600, 1], &mut rng);
    let _text_mask = model.run(&[(input_id, &img_data)], &[output_id]);
}

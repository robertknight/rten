extern crate png;
extern crate wasnn;

use std::fs;

use wasnn::ops::arg_max;
use wasnn::{Model, RunOptions, Tensor, TensorLayout};

/// Convert an 8-bit grayscale image to an NCHW float tensor with values in
/// the range [-0.5, 0.5].
fn tensor_from_image(width: usize, height: usize, data: &[u8]) -> Tensor {
    let mut img_tensor = Tensor::zeros(&[1, 1, height, width]);
    for y in 0..img_tensor.shape()[2] {
        for x in 0..img_tensor.shape()[3] {
            let b = y * width + x;
            img_tensor[[0, 0, y, x]] = (data[b] as f32 / 255.0) - 0.5;
        }
    }
    img_tensor
}

fn main() {
    let model_bytes = fs::read("models/ocr-rec.model").unwrap();
    let model = Model::load(&model_bytes).unwrap();

    let input_id = model
        .input_ids()
        .get(0)
        .copied()
        .expect("model has no inputs");
    let output_id = model
        .output_ids()
        .get(0)
        .copied()
        .expect("model has no outputs");

    let input_img = fs::File::open("test-data/ocr-input-gray.png").unwrap();
    let decoder = png::Decoder::new(input_img);

    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let frame_info = reader.next_frame(&mut buf).unwrap();
    let img_data = &buf[..frame_info.buffer_size()];
    let img_tensor = tensor_from_image(1088, 64, img_data);

    let result = model.run(
        &[(input_id, (&img_tensor).into())],
        &[output_id],
        Some(RunOptions {
            timing: true,
            verbose: false,
        }),
    );
    if let Err(err) = result {
        panic!("Model run failed: {:?}", err);
    }

    let outputs = result.unwrap();

    let char_probs = &outputs[0].as_float_ref().unwrap();

    // Character classes. Index 0 is a special "blank" character and is never
    // emitted in the output.
    let char_map = [
        'X', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', '\'',
        '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
        '^', '_', '`', '{', '|', '}', '~', ' ', '€', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];

    // Track if we have encountered a blank token since the last char was emmitted.
    let mut seen_blank = true;
    let chars = arg_max(char_probs, 2, false /* keep_dims */).expect("failed to get char scores");

    let mut text = String::new();
    for pos in 0..char_probs.shape()[1] {
        let char = chars[[0, pos]];

        if char > 0 {
            if seen_blank {
                text.push(char_map[char as usize]);
            }
            seen_blank = false;
        } else {
            seen_blank = true;
        }
    }

    println!("Text: {:?}", text);
}

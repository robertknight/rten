extern crate png;

use std::fs;
use std::io::BufWriter;

mod gemm;
mod graph;
mod model;
mod ops;
mod rng;
mod tensor;
mod timer;

#[allow(dead_code, unused_imports)]
mod schema_generated;

#[cfg(test)]
mod model_builder;

#[cfg(test)]
mod test_util;

use graph::RunOptions;
use model::load_model;
use tensor::{zero_tensor, Tensor};

/// Convert an 8-bit grayscale image to an NCHW float tensor with values in
/// the range [-0.5, 0.5].
fn tensor_from_image(width: usize, height: usize, data: &[u8]) -> Tensor {
    let mut img_tensor = zero_tensor(&[1, 1, height, width]);
    for y in 0..img_tensor.shape()[2] {
        for x in 0..img_tensor.shape()[3] {
            let b = y * width + x;
            img_tensor[[0, 0, y, x]] = (data[b] as f32 / 255.0) - 0.5;
        }
    }
    img_tensor
}

/// Convert an NCHW float tensor with values in the range [0, 1] to an
/// 8-bit grayscale image.
fn image_from_tensor(tensor: &Tensor) -> Vec<u8> {
    let mut buf = Vec::new();
    for y in 0..tensor.shape()[2] {
        for x in 0..tensor.shape()[3] {
            let byte = ((tensor[[0, 0, y, x]] + 0.5) * 255.0) as u8;
            buf.push(byte);
        }
    }
    buf
}

/// Convert an NCHW float tensor with values in the range [0, 1] to an
/// 8-bit grayscale image.
fn image_from_prob_tensor(tensor: &Tensor) -> Vec<u8> {
    let mut buf = Vec::new();
    for y in 0..tensor.shape()[2] {
        for x in 0..tensor.shape()[3] {
            let byte = (tensor[[0, 0, y, x]] * 255.0) as u8;
            buf.push(byte);
        }
    }
    buf
}

fn main() {
    let model_bytes = fs::read("output.model").unwrap();
    let model = load_model(&model_bytes).unwrap();

    let input_id = model.find_node("input.1").unwrap();
    let output_id = model.find_node("380").unwrap();

    let input_img = fs::File::open("test-image-800x600.png").unwrap();
    let decoder = png::Decoder::new(input_img);

    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let frame_info = reader.next_frame(&mut buf).unwrap();
    let img_data = &buf[..frame_info.buffer_size()];
    let img_tensor = tensor_from_image(600, 800, img_data);

    let outputs = model.run(
        &[(input_id, &img_tensor)],
        &[output_id],
        Some(RunOptions { timing: true }),
    );
    let text_mask = &outputs[0];
    let text_img = image_from_prob_tensor(&text_mask);

    let file = fs::File::create("output.png").unwrap();
    let writer = BufWriter::new(file);

    let encoder = png::Encoder::new(writer, 600, 800);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&text_img).unwrap();
}

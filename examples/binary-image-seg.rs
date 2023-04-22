extern crate png;
extern crate wasnn;

use std::error::Error;
use std::fs;
use std::io::BufWriter;
use std::iter::zip;

use wasnn::geometry::{draw_polygon, min_area_rect, Point, Polygon, Rect};
use wasnn::ops::{resize, CoordTransformMode, NearestMode, ResizeMode, ResizeTarget};
use wasnn::page_layout::{find_connected_component_rects, find_text_lines, line_polygon};
use wasnn::{tensor, Dimension, Model, RunOptions, Tensor, TensorLayout, TensorView};

/// Read a PNG image from `path` into a CHW tensor.
fn read_image(path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = fs::File::open(path)?;
    let decoder = png::Decoder::new(input_img);
    let mut reader = decoder.read_info()?;

    let in_chans = match reader.output_color_type() {
        (png::ColorType::Rgb, png::BitDepth::Eight) => 3,
        (png::ColorType::Rgba, png::BitDepth::Eight) => 4,
        (png::ColorType::Grayscale, png::BitDepth::Eight) => 1,
        _ => return Err("Unsupported input image format".into()),
    };

    let info = reader.info();
    let width = info.width as usize;
    let height = info.height as usize;

    let mut img_data = vec![0; reader.output_buffer_size()];
    let frame_info = reader.next_frame(&mut img_data).unwrap();
    img_data.truncate(frame_info.buffer_size());

    let img = TensorView::from_data(&[height, width, in_chans], img_data.as_slice());
    let mut float_img = Tensor::zeros(&[in_chans, height, width]);
    for c in 0..in_chans {
        for y in 0..height {
            for x in 0..width {
                float_img[[c, y, x]] = img[[y, x, c]] as f32 / 255.0
            }
        }
    }
    Ok(float_img)
}

/// Convert a CHW image into a greyscale image.
///
/// This function is intended to approximately match torchvision's RGB =>
/// greyscale conversion when using `torchvision.io.read_image(path,
/// ImageReadMode.GRAY)`, which is used when training models with greyscale
/// inputs. torchvision internally uses libpng's `png_set_rgb_to_gray`.
///
/// `normalize_pixel` is a function applied to each greyscale pixel value before
/// it is written into the output tensor.
fn greyscale_image<F: Fn(f32) -> f32>(img: TensorView<f32>, normalize_pixel: F) -> Tensor<f32> {
    let [chans, height, width]: [usize; 3] = img.shape().try_into().expect("expected 3 dim input");
    assert!(
        chans == 1 || chans == 3 || chans == 4,
        "expected greyscale, RGB or RGBA input image"
    );

    let mut output = Tensor::zeros(&[1, height, width]);

    let used_chans = chans.min(3); // For RGBA images, only RGB channels are used
    let chan_weights: &[f32] = if chans == 1 {
        &[1.]
    } else {
        // ITU BT.601 weights for RGB => luminance conversion. These match what
        // torchvision uses. See also https://stackoverflow.com/a/596241/434243.
        &[0.299, 0.587, 0.114]
    };

    for y in 0..height {
        for x in 0..width {
            let mut pixel = 0.;
            for c in 0..used_chans {
                pixel += img[[c, y, x]] * chan_weights[c];
            }
            output[[0, y, x]] = normalize_pixel(pixel);
        }
    }
    output
}

/// Write a CHW image to a PNG file in `path`.
fn write_image(path: &str, img: TensorView<f32>) -> Result<(), Box<dyn Error>> {
    if img.ndim() != 3 {
        return Err("Expected CHW input".into());
    }

    let img_width = img.shape()[img.ndim() - 1];
    let img_height = img.shape()[img.ndim() - 2];
    let color_type = match img.shape()[img.ndim() - 3] {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return Err("Unsupported channel count".into()),
    };

    let mut hwc_img = img.to_owned();
    hwc_img.permute(&[1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}

/// Convert an NCHW float tensor with values in the range [0, 1] to Vec<u8>
/// with values scaled to [0, 255].
fn image_from_tensor(tensor: TensorView<f32>) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

/// Compute color of composited pixel using source-over compositing.
///
/// See https://en.wikipedia.org/wiki/Alpha_compositing#Description.
fn composite_pixel(src_color: f32, src_alpha: f32, dest_color: f32, dest_alpha: f32) -> f32 {
    let alpha_out = src_alpha + dest_alpha * (1. - src_alpha);
    let color_out =
        (src_color * src_alpha + (dest_color * dest_alpha) * (1. - src_alpha)) / alpha_out;
    color_out
}

/// This example loads a PNG image and uses a binary image segmentation model
/// to classify pixels, for tasks such as detecting text, faces or
/// foreground/background separation. The output is the result of multiplying
/// input image pixels by the corresponding probabilities from the segmentation
/// mask.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() <= 1 {
        println!(
            "Usage: {} <model> <image> <output>",
            args.get(0).map(|s| s.as_str()).unwrap_or("")
        );
        // Exit with non-zero status, but also don't show an error.
        std::process::exit(1);
    }

    let model_name = args.get(1).ok_or("model name not specified")?;
    let image_path = args.get(2).ok_or("image path not specified")?;
    let output_path = args.get(3).ok_or("output path not specified")?;

    let model_bytes = fs::read(model_name)?;
    let model = Model::load(&model_bytes)?;

    let input_id = model
        .input_ids()
        .get(0)
        .copied()
        .expect("model has no inputs");
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;
    let output_id = model
        .output_ids()
        .get(0)
        .copied()
        .expect("model has no outputs");

    // Read image into CHW tensor.
    let mut color_img = read_image(image_path).expect("failed to read input image");
    let normalize_pixel = |pixel| pixel - 0.5;

    // Convert color CHW tensor to fixed-size greyscale NCHW input expected by model.
    let mut grey_img = greyscale_image(color_img.view(), normalize_pixel);
    let [_, img_height, img_width] = grey_img.dims();
    grey_img.insert_dim(0); // Add batch dimension

    let bilinear_resize = |img, height, width| {
        resize(
            img,
            ResizeTarget::Sizes(&tensor!([1, 1, height, width])),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
    };

    let (in_height, in_width) = match input_shape[..] {
        [_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),
        _ => {
            return Err("failed to get model dims".into());
        }
    };

    let resized_grey_img = bilinear_resize(&grey_img, in_height as i32, in_width as i32)?;

    // Run text detection model to compute a probability mask indicating whether
    // each pixel is part of a text word or not.
    let outputs = model.run(
        &[(input_id, (&resized_grey_img).into())],
        &[output_id],
        Some(RunOptions {
            timing: true,
            verbose: false,
        }),
    )?;

    // Resize probability mask to original input size and apply threshold to get a
    // binary text/not-text mask.
    let text_mask = &outputs[0].as_float_ref().unwrap();
    let text_mask = bilinear_resize(text_mask, img_height as i32, img_width as i32)?;
    let threshold = 0.2;
    let binary_mask = text_mask.map(|prob| if prob > threshold { 1i32 } else { 0 });

    // Highlight object pixels in image.
    // let mut combined_img_mask = Tensor::<f32>::zeros(text_mask.shape());
    // for (out, (img, prob)) in zip(
    //     combined_img_mask.iter_mut(),
    //     zip(img.iter(), text_mask.iter()),
    // ) {
    //     // Convert normalized image pixel back to greyscale value in [0, 1]
    //     let in_grey = img + 0.5;
    //     let mask_pixel = if prob > threshold { 1. } else { 0. };
    //     *out = composite_pixel(mask_pixel, 0.2, in_grey, 1.);
    // }

    // Distance to expand bounding boxes by. This is useful when the model is
    // trained to assign a positive label to pixels in a smaller area than the
    // ground truth, which may be done to create separation between adjacent
    // objects.
    let expand_dist = 3.;

    // Perform layout analysis to group words into lines.
    let word_rects = find_connected_component_rects(binary_mask.nd_slice([0, 0]), expand_dist);
    let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
    let line_rects = find_text_lines(&word_rects, page_rect);

    // Annotate input image with detected line outlines.
    for (line_index, word_rects) in line_rects.iter().enumerate() {
        let line_poly = Polygon::new(line_polygon(word_rects));
        let grey_chan = grey_img.slice(&[0.into(), 0.into()]);

        draw_polygon(color_img.nd_slice_mut([0]), line_poly.vertices(), 0.9); // Red
        draw_polygon(color_img.nd_slice_mut([1]), line_poly.vertices(), 0.); // Green
        draw_polygon(color_img.nd_slice_mut([2]), line_poly.vertices(), 0.); // Blue

        // Extract line image
        let line_rect = line_poly.bounding_rect();
        let mut out_img =
            Tensor::zeros(&[1, line_rect.height() as usize, line_rect.width() as usize]);

        // Page rect adjusted to only contain coordinates that are valid for
        // indexing into the input image.
        let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

        for in_p in line_poly.fill_iter() {
            let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());

            if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
                continue;
            }

            let normalized_pixel = grey_chan[[in_p.y as usize, in_p.x as usize]];
            let pixel = normalized_pixel + 0.5;
            out_img[[0, out_p.y as usize, out_p.x as usize]] = pixel;
        }
        let line_img_path = format!("line_{}.png", line_index);
        write_image(&line_img_path, out_img.view())?;
    }

    println!(
        "Found {} words, {} lines in image of size {}x{}",
        word_rects.len(),
        line_rects.len(),
        img_width,
        img_height
    );

    // Write out the annotated input image.
    write_image(output_path, color_img.view())?;

    Ok(())
}

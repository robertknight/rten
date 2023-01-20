use std::fs::File;
use std::io::BufReader;
use std::iter::zip;

use serde_json::Value;

use crate::tensor::Tensor;

/// Trait that tests whether two values are approximately equal.
///
/// Here "approximately" means "a value that is reasonable for this crate's
/// tests".
pub trait ApproxEq {
    fn approx_eq(self, other: Self) -> bool;
}

impl ApproxEq for f32 {
    fn approx_eq(self, other: f32) -> bool {
        let eps = 0.001;
        (self - other).abs() < eps
    }
}

impl ApproxEq for i32 {
    fn approx_eq(self, other: i32) -> bool {
        self == other
    }
}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
pub fn expect_equal<T: ApproxEq + Copy>(x: &Tensor<T>, y: &Tensor<T>) -> Result<(), String> {
    if x.shape() != y.shape() {
        return Err(format!(
            "Tensors have different shapes. {:?} vs. {:?}",
            x.shape(),
            y.shape()
        ));
    }

    let mut mismatches = 0;
    for (xi, yi) in zip(x.iter(), y.iter()) {
        if !xi.approx_eq(yi) {
            mismatches += 1;
        }
    }

    if mismatches > 0 {
        Err(format!(
            "Tensor values differ at {} of {} indexes",
            mismatches,
            x.len()
        ))
    } else {
        Ok(())
    }
}

/// Read a float tensor from a JSON value.
///
/// The JSON value is expected to be of the form `[shape, data]` where
/// `shape` is an int array and `data` is a float array.
pub fn read_tensor(val: &Value) -> Result<Tensor<f32>, &'static str> {
    let vec = match val {
        Value::Array(vec) => vec,
        _ => return Err("Expected array"),
    };

    let (shape, data) = match vec.as_slice() {
        [Value::Array(shape), Value::Array(data)] => (shape, data),
        _ => return Err("Expected [shape, data] array"),
    };

    let shape = shape
        .iter()
        .map(|v| v.as_i64().map(|v| v as usize).ok_or("Expected int array"))
        .collect::<Result<Vec<usize>, _>>()?;

    let data = data
        .iter()
        .map(|v| v.as_f64().map(|v| v as f32).ok_or("Expected float array"))
        .collect::<Result<Vec<f32>, _>>()?;

    Ok(Tensor::from_data(shape, data))
}

pub fn read_json_file(path: &str) -> Value {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

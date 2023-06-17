use std::fmt::Debug;
use std::fs::File;
use std::io::BufReader;
use std::iter::zip;

use serde_json::Value;

use crate::tensor::{Tensor, TensorBase, TensorLayout};

/// Trait that tests whether two values are approximately equal.
///
/// Here "approximately" means "a value that is reasonable for this crate's
/// tests".
pub trait ApproxEq {
    fn approx_eq(self, other: Self) -> bool;
}

impl ApproxEq for f32 {
    fn approx_eq(self, other: f32) -> bool {
        let eps = 1e-4;
        (self - other).abs() < eps
    }
}

impl ApproxEq for i32 {
    fn approx_eq(self, other: i32) -> bool {
        self == other
    }
}

/// Return the N-dimensional index in a tensor with a given `shape` that
/// corresponds to a linear index (ie. the index if the tensor was flattened to
/// 1D).
fn index_from_linear_index(shape: &[usize], lin_index: usize) -> Vec<usize> {
    assert!(
        lin_index < shape.iter().product(),
        "Linear index {} is out of bounds for shape {:?}",
        lin_index,
        shape,
    );
    (0..shape.len())
        .map(|dim| {
            let elts_per_index: usize = shape[dim + 1..].iter().product();
            let lin_index_for_dim = lin_index % (shape[dim] * elts_per_index);
            lin_index_for_dim / elts_per_index
        })
        .collect()
}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
///
/// If there are mismatches, this returns an `Err` with a message indicating
/// the count of mismatches and details of the first N cases.
pub fn expect_equal<T: ApproxEq + Copy + Debug, S: AsRef<[T]>>(
    x: &TensorBase<T, S>,
    y: &TensorBase<T, S>,
) -> Result<(), String> {
    if x.shape() != y.shape() {
        return Err(format!(
            "Tensors have different shapes. {:?} vs. {:?}",
            x.shape(),
            y.shape()
        ));
    }

    let mismatches: Vec<_> = zip(x.iter(), y.iter())
        .enumerate()
        .filter_map(|(i, (xi, yi))| {
            if !xi.approx_eq(yi) {
                Some((index_from_linear_index(x.shape(), i), xi, yi))
            } else {
                None
            }
        })
        .collect();

    if mismatches.len() > 0 {
        let max_examples = 16;
        Err(format!(
            "Tensor values differ at {} of {} indexes: {:?}{}",
            mismatches.len(),
            x.len(),
            &mismatches[..mismatches.len().min(max_examples)],
            if mismatches.len() > max_examples {
                "..."
            } else {
                ""
            }
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

    Ok(Tensor::from_data(&shape, data))
}

pub fn read_json_file(path: &str) -> Value {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

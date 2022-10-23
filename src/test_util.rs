use std::iter::zip;

use crate::tensor::Tensor;

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
pub fn expect_equal(x: &Tensor, y: &Tensor) -> Result<(), String> {
    if x.shape() != y.shape() {
        return Err(format!(
            "Tensors have different shapes. {:?} vs. {:?}",
            x.shape(),
            y.shape()
        ));
    }

    let mut mismatches = 0;
    let eps = 0.001;

    for (xi, yi) in zip(x.elements(), y.elements()) {
        if (xi - yi).abs() > eps {
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

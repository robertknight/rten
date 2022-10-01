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

    let eps = 0.001;
    for i in 0..x.len() {
        let xi = x.data()[i];
        let yi = y.data()[i];

        if (xi - yi).abs() > eps {
            return Err(format!(
                "Tensor values differ at index {}: {} vs {}",
                i, xi, yi
            ));
        }
    }

    Ok(())
}

//! Activations related to the ReLU activation.
//!
//! Vanilla ReLU doesn't really need an explicitly vectorized kernel because it
//! is just `x.max(0)` which is easy for compilers to auto-vectorize. Variants
//! such as leaky ReLU however do benefit.

use rten_simd::ops::NumOps;
use rten_simd::{Isa, SimdUnaryOp};

/// Computes the leaky ReLU activation function.
///
/// This evaluates `if x < 0. { alpha * x } else { x }`.
pub struct LeakyRelu {
    pub alpha: f32,
}

impl SimdUnaryOp<f32> for LeakyRelu {
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let alpha = ops.splat(self.alpha);
        let x_neg = ops.lt(x, ops.zero());
        let x_mul_alpha = ops.mul(x, alpha);
        ops.select(x_mul_alpha, x, x_neg)
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{Tolerance, UnaryOpTester};

    use super::LeakyRelu;

    fn reference_leaky_relu(x: f32, alpha: f32) -> f32 {
        if x < 0. { alpha * x } else { x }
    }

    #[test]
    fn test_leaky_relu() {
        let alpha = 0.5;
        let test = UnaryOpTester {
            reference: |x: f32| reference_leaky_relu(x, alpha),
            simd: LeakyRelu { alpha },
            range: [-2., -1., 0., 1., 2.].iter().copied(),
            tolerance: Tolerance::Ulp(1.0),
        };
        test.run();
    }
}

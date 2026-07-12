//! Vectorized thresholding operations.

use rten_simd::ops::{BitOps, FloatOps, NumOps};
use rten_simd::{Isa, SimdUnaryOp};

/// Computes the Shrink operation.
///
/// This evaluates `x + bias` if `x < -lambd`, `x - bias` if `x > lambd` and
/// zero otherwise.
pub struct Shrink {
    pub bias: f32,
    pub lambd: f32,
}

impl SimdUnaryOp<f32> for Shrink {
    #[inline(always)]
    fn eval<I: Isa>(&self, isa: I, x: I::F32) -> I::F32 {
        let ops = isa.f32();
        let bias = ops.splat(self.bias);
        let lambd = ops.splat(self.lambd);

        let below = ops.lt(x, ops.neg(lambd));
        let above = ops.gt(x, lambd);

        let result = ops.select(ops.sub(x, bias), ops.zero(), above);
        ops.select(ops.add(x, bias), result, below)
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{Tolerance, UnaryOpTester, arange};

    use super::Shrink;

    fn reference_shrink(x: f32, bias: f32, lambd: f32) -> f32 {
        if x < -lambd {
            x + bias
        } else if x > lambd {
            x - bias
        } else {
            0.
        }
    }

    #[test]
    fn test_shrink() {
        let bias = 1.5;
        let lambd = 1.0;
        let test = UnaryOpTester {
            reference: |x: f32| reference_shrink(x, bias, lambd),
            simd: Shrink { bias, lambd },
            range: arange(-3., 3., 0.1),
            tolerance: Tolerance::Ulp(0.0),
        };
        test.run();
    }
}

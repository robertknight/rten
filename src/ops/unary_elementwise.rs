use rayon::prelude::*;

use std::fmt::Debug;
use std::mem::MaybeUninit;

use rten_base::num::AsBool;
use rten_simd::SimdUnaryOp;
use rten_simd::ops::{GetNumOps, GetSimd};
use rten_tensor::prelude::*;
use rten_tensor::{AssumeInit, Tensor, TensorView, TensorViewMut};
use rten_vecmath as vecmath;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::ops::binary_elementwise::binary_op;
use crate::ops::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, Value, ValueView, map_value,
    map_value_view,
};

trait UnaryKernel<T> {
    /// Apply the unary operation to elements of `src`, writing to `dst`.
    fn map<'dst>(&self, src: &[T], dst: &'dst mut [MaybeUninit<T>]) -> &'dst mut [T];

    /// Apply the unary operation in-place to elements of `src`.
    fn map_mut(&self, src: &mut [T]);
}

impl<T: Copy, Op: Fn(T) -> T> UnaryKernel<T> for Op {
    fn map<'dst>(&self, src: &[T], dst: &'dst mut [MaybeUninit<T>]) -> &'dst mut [T] {
        src.iter().zip(dst.iter_mut()).for_each(|(x, y)| {
            y.write(self(*x));
        });
        unsafe { dst.assume_init() }
    }

    fn map_mut(&self, src: &mut [T]) {
        for x in src {
            *x = self(*x);
        }
    }
}

/// A vectorized unary operator kernel wrapping a [`SimdUnaryOp`].
struct SimdKernel<Op>(Op);

impl<T: GetNumOps + GetSimd, Op: SimdUnaryOp<T>> UnaryKernel<T> for SimdKernel<Op> {
    fn map<'dst>(&self, src: &[T], dst: &'dst mut [MaybeUninit<T>]) -> &'dst mut [T] {
        self.0.map(src, dst)
    }

    fn map_mut(&self, src: &mut [T]) {
        self.0.map_mut(src)
    }
}

/// Get the unary op kernel for a given element type.
trait GetKernel<T> {
    fn get_kernel(&self) -> impl UnaryKernel<T> + Send + Sync;
}

/// Impl the [`GetKernel`] trait for a given operator and element type.
macro_rules! impl_get_kernel {
    ($op:ty, $elem_ty:ty, $kernel:expr) => {
        impl GetKernel<$elem_ty> for $op {
            fn get_kernel(&self) -> impl UnaryKernel<$elem_ty> + Send + Sync {
                $kernel
            }
        }
    };
}

/// Minimum number of elements in a chunk that is processed on a single thread.
const CHUNK_SIZE: usize = 32 * 1024;

/// Apply a unary operation in parallel to contiguous slices of `input`.
fn unary_op<T: Clone + Send + Sync>(
    pool: &BufferPool,
    input: TensorView<T>,
    op: &(dyn UnaryKernel<T> + Send + Sync),
) -> Tensor<T> {
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let mut output = Tensor::uninit_in(pool, input.shape());

    let in_chunks = input.data().unwrap().par_chunks(CHUNK_SIZE);
    let out_chunks = output.data_mut().unwrap().par_chunks_mut(CHUNK_SIZE);
    in_chunks.zip(out_chunks).for_each(|(in_chunk, out_chunk)| {
        op.map(in_chunk, out_chunk);
    });

    // Safety: `op` initialized each chunk of `out_chunks`.
    unsafe { output.assume_init() }
}

/// Apply a unary operation in parallel to contiguous slices of `input`,
/// writing the results in-place.
fn unary_op_in_place<T: Clone + Send + Sync>(
    pool: &BufferPool,
    mut input: Tensor<T>,
    op: &(dyn UnaryKernel<T> + Send + Sync),
) -> Tensor<T> {
    if let Some(data) = input.data_mut() {
        data.par_chunks_mut(CHUNK_SIZE)
            .for_each(|chunk| op.map_mut(chunk));
        input
    } else {
        unary_op(pool, input.view(), op)
    }
}

/// Create a struct for a unary operator with no parameters.
macro_rules! declare_operator {
    ($op_name:ident) => {
        #[derive(Debug)]
        pub struct $op_name {}
    };
}

/// Impl [`Operator`] for a unary operator type.
macro_rules! impl_operator {
    ($op_name:ident, $types:tt) => {
        impl Operator for $op_name {
            fn name(&self) -> &str {
                stringify!($op_name)
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
                let input = ctx.inputs().require(0)?;
                map_value_view!(input, input, $types, {
                    let kernel = self.get_kernel();
                    unary_op(ctx.pool(), input, &kernel).into_op_result()
                })
            }

            fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
                map_value!(input, input, $types, {
                    let kernel = self.get_kernel();
                    let result = unary_op_in_place(ctx.pool(), input, &kernel);
                    Ok(result.into())
                })
            }
        }
    };
}

/// Define a function that runs a unary operator.
macro_rules! impl_operator_fn {
    ($op_name:ident, $func_name:ident) => {
        pub fn $func_name(pool: &BufferPool, input: TensorView) -> Tensor {
            let op = $op_name {};
            let kernel = op.get_kernel();
            unary_op(pool, input, &kernel)
        }
    };

    ($op_name:ident, $func_name:ident, cfg_test) => {
        #[cfg(test)]
        pub fn $func_name(pool: &BufferPool, input: TensorView) -> Tensor {
            let op = $op_name {};
            let kernel = op.get_kernel();
            unary_op(pool, input, &kernel)
        }
    };
}

pub trait AbsValue {
    fn abs(&self) -> Self;
}

impl AbsValue for f32 {
    fn abs(&self) -> f32 {
        (*self).abs()
    }
}

impl AbsValue for i32 {
    fn abs(&self) -> i32 {
        (*self).abs()
    }
}

declare_operator!(Abs);
impl_operator!(Abs, [FloatTensor, Int32Tensor]);

impl<T: AbsValue + Copy> GetKernel<T> for Abs {
    fn get_kernel(&self) -> impl UnaryKernel<T> + Send + Sync {
        |val: T| val.abs()
    }
}

declare_operator!(Acos);
impl_operator!(Acos, [FloatTensor]);
impl_get_kernel!(Acos, f32, |val: f32| val.acos());

declare_operator!(Asin);
impl_operator!(Asin, [FloatTensor]);
impl_get_kernel!(Asin, f32, |val: f32| val.asin());

declare_operator!(Atan);
impl_operator!(Atan, [FloatTensor]);
impl_get_kernel!(Atan, f32, |val: f32| val.atan());

declare_operator!(Ceil);
impl_operator!(Ceil, [FloatTensor]);
impl_operator_fn!(Ceil, ceil, cfg_test);
impl_get_kernel!(Ceil, f32, |val: f32| val.ceil());

/// Numeric value with a finite minimum and maximum and operations to clamp
/// values.
pub trait Clamp: Copy + PartialOrd {
    /// Return the minimum possible finite value for this type.
    fn min_val() -> Self;

    /// Return the maximum possible finite value for this type.
    fn max_val() -> Self;

    /// Return the minimum of `self` and `val`.
    fn min(&self, val: Self) -> Self {
        if *self < val { *self } else { val }
    }

    /// Return the maximum of `self` and `val`.
    fn max(&self, val: Self) -> Self {
        if *self > val { *self } else { val }
    }

    /// Return self constrained to the range `[min, max]`.
    fn clamp(&self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl Clamp for i32 {
    fn min_val() -> Self {
        i32::MIN
    }

    fn max_val() -> Self {
        i32::MAX
    }
}

impl Clamp for f32 {
    fn min_val() -> Self {
        f32::MIN
    }

    fn max_val() -> Self {
        f32::MAX
    }
}

pub fn clip<T: Copy + Clamp>(
    pool: &BufferPool,
    input: TensorView<T>,
    min: Option<T>,
    max: Option<T>,
) -> Tensor<T> {
    let min = min.unwrap_or(T::min_val());
    let max = max.unwrap_or(T::max_val());
    input.map_in(pool, |x| x.clamp(min, max))
}

pub fn clip_in_place<T: Copy + Clamp>(input: &mut Tensor<T>, min: Option<T>, max: Option<T>) {
    let min = min.unwrap_or(T::min_val());
    let max = max.unwrap_or(T::max_val());
    input.apply(|x| x.clamp(min, max))
}

// TODO - Move `Clip` operator into another module since it is no longer a
// unary op (it used to take `min` and `max` as attributes).

#[derive(Debug)]
pub struct Clip {}

impl Operator for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        map_value_view!(input, input, [FloatTensor, Int32Tensor], {
            let min = inputs.get_as(1)?;
            let max = inputs.get_as(2)?;
            clip(ctx.pool(), input, min, max).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        map_value!(input, input, [FloatTensor, Int32Tensor], {
            let min = ctx.inputs().get_as(0)?;
            let max = ctx.inputs().get_as(1)?;
            clip_in_place(&mut input, min, max);
            Ok(input.into())
        })
    }
}

declare_operator!(Cos);
impl_operator!(Cos, [FloatTensor]);
impl_get_kernel!(Cos, f32, SimdKernel(vecmath::Cos::new()));

#[derive(Debug)]
pub struct Elu {
    pub alpha: f32,
}

impl_operator!(Elu, [FloatTensor]);

impl GetKernel<f32> for Elu {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Sync + Send {
        |val: f32| {
            // The ONNX spec and the original paper [1] define Elu in slightly
            // different, but equivalent ways:
            //
            // Original: `f(x) = x if x > 0 else alpha * (exp(x) - 1)`
            // ONNX: `f(x) = x if x >= 0 else alpha * (exp(x) - 1)`
            //
            // [1] https://arxiv.org/pdf/1511.07289

            if val >= 0. {
                val
            } else {
                self.alpha * (val.exp() - 1.)
            }
        }
    }
}

declare_operator!(Erf);
impl_operator!(Erf, [FloatTensor]);
impl_operator_fn!(Erf, erf, cfg_test);
impl_get_kernel!(Erf, f32, SimdKernel(vecmath::Erf {}));

declare_operator!(Exp);
impl_operator!(Exp, [FloatTensor]);
impl_get_kernel!(Exp, f32, SimdKernel(vecmath::Exp {}));

declare_operator!(Floor);
impl_operator!(Floor, [FloatTensor]);
impl_operator_fn!(Floor, floor, cfg_test);
impl_get_kernel!(Floor, f32, |val: f32| val.floor());

#[derive(Debug)]
pub struct Gelu {
    pub approximate: bool,
}

impl_operator!(Gelu, [FloatTensor]);

impl GetKernel<f32> for Gelu {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Send + Sync {
        if self.approximate {
            GeluKernel::Approximate(SimdKernel(vecmath::ApproxGelu {}))
        } else {
            GeluKernel::Standard(SimdKernel(vecmath::Gelu {}))
        }
    }
}

enum GeluKernel {
    Approximate(SimdKernel<vecmath::ApproxGelu>),
    Standard(SimdKernel<vecmath::Gelu>),
}

impl UnaryKernel<f32> for GeluKernel {
    fn map<'dst>(&self, src: &[f32], dst: &'dst mut [MaybeUninit<f32>]) -> &'dst mut [f32] {
        match self {
            Self::Approximate(kern) => kern.map(src, dst),
            Self::Standard(kern) => kern.map(src, dst),
        }
    }

    fn map_mut(&self, src: &mut [f32]) {
        match self {
            Self::Approximate(kern) => kern.map_mut(src),
            Self::Standard(kern) => kern.map_mut(src),
        }
    }
}

#[derive(Debug)]
pub struct HardSigmoid {
    pub alpha: f32,
    pub beta: f32,
}

impl_operator!(HardSigmoid, [FloatTensor]);

impl GetKernel<f32> for HardSigmoid {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Send + Sync {
        move |val: f32| (self.alpha * val + self.beta).clamp(0., 1.)
    }
}

#[cfg(test)]
pub fn hard_sigmoid(pool: &BufferPool, input: TensorView, alpha: f32, beta: f32) -> Tensor {
    let op = HardSigmoid { alpha, beta };
    unary_op(pool, input, &op.get_kernel())
}

#[derive(Debug)]
pub struct HardSwish {}

impl_operator!(HardSwish, [FloatTensor]);

impl GetKernel<f32> for HardSwish {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Send + Sync {
        |val: f32| {
            let alpha = 1. / 6.;
            let beta = 0.5;
            val * (alpha * val + beta).clamp(0., 1.)
        }
    }
}

impl_operator_fn!(HardSwish, hard_swish, cfg_test);

#[derive(Debug)]
pub struct IsInf {}

impl Operator for IsInf {
    fn name(&self) -> &str {
        "IsInf"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        let output = input.map_in(ctx.pool(), |x| i32::from(x.is_infinite()));
        output.into_op_result()
    }
}

#[derive(Debug)]
pub struct IsNaN {}

impl Operator for IsNaN {
    fn name(&self) -> &str {
        "IsNaN"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        let output = input.map_in(ctx.pool(), |x| i32::from(x.is_nan()));
        output.into_op_result()
    }
}

#[derive(Debug)]
pub struct LeakyRelu {
    pub alpha: f32,
}

impl_operator!(LeakyRelu, [FloatTensor]);

impl GetKernel<f32> for LeakyRelu {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Send + Sync {
        |val: f32| {
            if val < 0.0 { self.alpha * val } else { val }
        }
    }
}

#[cfg(test)]
pub fn leaky_relu(pool: &BufferPool, input: TensorView, alpha: f32) -> Tensor {
    let op = LeakyRelu { alpha };
    unary_op(pool, input, &op.get_kernel())
}

declare_operator!(Log);
impl_operator!(Log, [FloatTensor]);
impl_get_kernel!(Log, f32, |val: f32| val.ln());

declare_operator!(Neg);
impl_operator!(Neg, [FloatTensor, Int32Tensor]);

impl<T: Copy + std::ops::Neg<Output = T>> GetKernel<T> for Neg {
    fn get_kernel(&self) -> impl UnaryKernel<T> + Send + Sync {
        |val: T| val.neg()
    }
}

pub fn not<T: AsBool + PartialEq>(pool: &BufferPool, input: TensorView<T>) -> Tensor<i32> {
    input.map_in(pool, |x| i32::from(!x.as_bool()))
}

pub fn not_in_place(mut input: TensorViewMut<i32>) {
    input.apply(|x| i32::from(!x.as_bool()));
}

#[derive(Debug)]
pub struct Not {}

impl Operator for Not {
    fn name(&self) -> &str {
        "Not"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<i32> = ctx.inputs().require_as(0)?;
        not(ctx.pool(), input).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, _ctx: &OpRunContext) -> Result<Value, OpError> {
        let mut output: Tensor<i32> = input.try_into()?;
        not_in_place(output.view_mut());
        Ok(output.into())
    }
}

declare_operator!(Reciprocal);
impl_operator!(Reciprocal, [FloatTensor]);
impl_get_kernel!(Reciprocal, f32, |val: f32| 1. / val);

declare_operator!(Relu);
impl_operator!(Relu, [FloatTensor]);
impl_get_kernel!(Relu, f32, |val: f32| val.max(0.));

/// Round float values to the nearest integer. Values with a fractional part
/// of 0.5 are rounded to the nearest even number, like `round` in Python and
/// unlike [`f32::round`] in Rust.
#[derive(Debug)]
pub struct Round {}
impl_operator!(Round, [FloatTensor]);
impl_get_kernel!(Round, f32, |val: f32| val.round_ties_even());
impl_operator_fn!(Round, round, cfg_test);

fn prelu<T: Copy + Default + PartialOrd + std::ops::Mul<Output = T>>(
    pool: &BufferPool,
    x: TensorView<T>,
    slope: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    if !slope.can_broadcast_to(x.shape()) {
        return Err(OpError::IncompatibleInputShapes(
            "Slope is not broadcastable to input shape",
        ));
    }

    // Even though PRelu is technically a binary operation in ONNX, it is
    // usually described as elementwise because the slope parameter is normally
    // a constant scalar or vector.
    binary_op(pool, x, slope, &|x, alpha| {
        if x < T::default() { alpha * x } else { x }
    })
}

#[derive(Debug)]
pub struct PRelu {}

impl Operator for PRelu {
    fn name(&self) -> &str {
        "PRelu"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;

        map_value_view!(input, input, [FloatTensor], {
            let slope = ctx.inputs().require_as(1)?;
            prelu(ctx.pool(), input, slope).into_op_result()
        })
    }
}

declare_operator!(Sigmoid);
impl_operator!(Sigmoid, [FloatTensor]);
impl_operator_fn!(Sigmoid, sigmoid);
impl_get_kernel!(Sigmoid, f32, SimdKernel(vecmath::Sigmoid {}));

// Sigmoid Linear Unit (SiLU) function.
//
// This is a special case of the Swish function
// (<https://en.wikipedia.org/wiki/Swish_function>).
//
// Not an official ONNX operator, but used in popular object detection models.
// See https://github.com/onnx/onnx/issues/4854.
declare_operator!(Silu);
impl_operator!(Silu, [FloatTensor]);
impl_get_kernel!(Silu, f32, SimdKernel(vecmath::Silu {}));

/// Swish function (<https://en.wikipedia.org/wiki/Swish_function>).
///
/// This computes `x * sigmoid(beta * x)`. The special case where beta = 1 is
/// known as [`Silu`].
#[derive(Debug)]
pub struct Swish {
    pub beta: f32,
}

impl_operator!(Swish, [FloatTensor]);

impl GetKernel<f32> for Swish {
    fn get_kernel(&self) -> impl UnaryKernel<f32> + Send + Sync {
        SimdKernel(vecmath::Swish { beta: self.beta })
    }
}

declare_operator!(Sin);
impl_operator!(Sin, [FloatTensor]);
impl_get_kernel!(Sin, f32, SimdKernel(vecmath::Sin::new()));

/// Trait for obtaining the sign of a number (-1, 0 or 1) as a value of the
/// same type.
pub trait Signum: Copy {
    /// Return -1, 0 or 1 if the value is negative, zero or positive
    /// respectively.
    fn signum(self) -> Self;
}

macro_rules! impl_signum {
    ($type:ident) => {
        impl Signum for $type {
            fn signum(self) -> Self {
                $type::signum(self)
            }
        }
    };
}
impl_signum!(i32);
impl_signum!(f32);

declare_operator!(Sign);
impl_operator!(Sign, [FloatTensor, Int32Tensor]);

impl<T: Copy + Signum> GetKernel<T> for Sign {
    fn get_kernel(&self) -> impl UnaryKernel<T> + Send + Sync {
        |val: T| val.signum()
    }
}

declare_operator!(Sqrt);
impl_operator!(Sqrt, [FloatTensor]);
impl_get_kernel!(Sqrt, f32, |val: f32| val.sqrt());

declare_operator!(Softplus);
impl_operator!(Softplus, [FloatTensor]);
impl_get_kernel!(Softplus, f32, |val: f32| val.exp().ln_1p());

declare_operator!(Tan);
impl_operator!(Tan, [FloatTensor]);
impl_get_kernel!(Tan, f32, |val: f32| val.tan());

declare_operator!(Tanh);
impl_operator!(Tanh, [FloatTensor]);
impl_operator_fn!(Tanh, tanh);
impl_get_kernel!(Tanh, f32, SimdKernel(vecmath::Tanh {}));

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{eq_with_nans, expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{RandomSource, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{
        Abs, Acos, Asin, Atan, Cos, Elu, Exp, Gelu, IsInf, IsNaN, Log, Neg, Not, PRelu, Reciprocal,
        Relu, Sigmoid, Sign, Silu, Sin, Softplus, Sqrt, Swish, Tan, Tanh, ceil, clip,
        clip_in_place, erf, floor, hard_sigmoid, hard_swish, leaky_relu, round,
    };
    use crate::ops::tests::new_pool;
    use crate::ops::{CastError, OpError, Operator, OperatorExt, Value, ValueView};
    use rten_tensor::test_util::ApproxEq;

    // Test a unary operator's in-place and non-in-place implementations.
    fn test_unary_op_both<T: Clone + std::fmt::Debug + ApproxEq>(
        op: impl Operator,
        reference_op: impl Fn(&T) -> T,
        input: Tensor<T>,
    ) -> Result<(), Box<dyn Error>>
    where
        for<'a> TensorView<'a, T>: Into<ValueView<'a>>,
        Tensor<T>: Into<Value> + TryFrom<Value, Error = CastError>,
    {
        let expected = input.map(reference_op);

        // Test copying variant.
        test_unary_op_not_in_place(&op, input.view(), expected.view())?;

        // Test in-place variant.
        let input_mut = input.clone();
        let result: Tensor<T> = op.run_simple_in_place(input_mut, ()).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    // Test a unary operator's non-in-place implementation.
    fn test_unary_op_not_in_place<T, U: Clone + std::fmt::Debug + ApproxEq>(
        op: &impl Operator,
        input: TensorView<T>,
        expected: TensorView<U>,
    ) -> Result<(), Box<dyn Error>>
    where
        for<'a> TensorView<'a, T>: Into<ValueView<'a>>,
        Tensor<U>: Into<Value> + TryFrom<Value, Error = CastError>,
    {
        let result: Tensor<U> = op.run_simple(input).unwrap();
        expect_equal(&result.view(), &expected.view())?;
        Ok(())
    }

    /// Define a test for a unary operator which applies the function
    /// `$gen_expected` to each input element.
    macro_rules! test_unary_op {
        ($test_name:ident, $op:expr, $gen_expected:expr) => {
            #[test]
            fn $test_name() -> Result<(), Box<dyn Error>> {
                // Test inputs here chosen to be in the domain of inverse trig
                // operators (ie. (-1, 1)).
                let input = Tensor::from([0., 0.1, -0.1, 0.9, -0.9]);
                test_unary_op_both($op, $gen_expected, input)
            }
        };

        ($test_name:ident, $op:expr, $gen_expected:expr, $input:expr) => {
            #[test]
            fn $test_name() -> Result<(), Box<dyn Error>> {
                test_unary_op_both($op, $gen_expected, $input)
            }
        };
    }

    /// Source for `Tensor::rand` that generates values in a given range.
    struct RandomFloat {
        rng: XorShiftRng,
        min: f32,
        max: f32,
    }

    impl RandomFloat {
        /// Create a new generator with a default range of `[0, 1)`.
        fn new(seed: u64) -> RandomFloat {
            RandomFloat {
                rng: XorShiftRng::new(seed),
                min: 0.,
                max: 1.,
            }
        }

        /// Convert `self` into a generator of values in `[min, max)`.
        fn with_range(self, min: f32, max: f32) -> RandomFloat {
            RandomFloat {
                rng: self.rng,
                min,
                max,
            }
        }
    }

    impl RandomSource<f32> for RandomFloat {
        fn next(&mut self) -> f32 {
            let x: f32 = self.rng.next();
            self.min + (self.max - self.min) * x
        }
    }

    #[test]
    fn test_abs() {
        test_unary_op_both(Abs {}, |x| x.abs(), [1., -1., 0.].into()).unwrap();
        test_unary_op_both(Abs {}, |x| x.abs(), [1, -1, 0].into()).unwrap();
    }

    test_unary_op!(test_acos, Acos {}, |x: &f32| x.acos());
    test_unary_op!(test_asin, Asin {}, |x: &f32| x.asin());
    test_unary_op!(test_atan, Atan {}, |x: &f32| x.atan());

    #[test]
    fn test_ceil() {
        let pool = new_pool();
        let input = Tensor::from([
            1.,
            1.2,
            1.5,
            1.8,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        let expected = Tensor::from([
            1.,
            2.,
            2.,
            2.,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        let result = ceil(&pool, input.view());
        assert!(eq_with_nans(result.view(), expected.view()));
    }

    #[test]
    fn test_clip() {
        #[derive(Debug)]
        struct Case {
            input: Tensor,
            min: Option<f32>,
            max: Option<f32>,
            expected: Tensor,
        }

        let cases = [
            Case {
                input: [[-5., -2.], [3., 20.]].into(),
                min: Some(1.),
                max: Some(5.),
                expected: [[1., 1.], [3., 5.]].into(),
            },
            Case {
                input: [[-5., -2.], [3., 20.]].into(),
                min: Some(1.),
                max: None,
                expected: [[1., 1.], [3., 20.]].into(),
            },
            Case {
                input: [[-5., -2.], [3., 20.]].into(),
                min: None,
                max: Some(5.),
                expected: [[-5., -2.], [3., 5.]].into(),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let result = clip(&pool, case.input.view(), case.min, case.max);
            expect_equal(&result, &case.expected).unwrap();

            let mut input = case.input.clone();
            clip_in_place(&mut input, case.min, case.max);
            expect_equal(&input, &case.expected).unwrap();
        })
    }

    test_unary_op!(test_cos, Cos {}, |x: &f32| x.cos());

    #[test]
    fn test_elu() {
        #[derive(Debug)]
        struct Case {
            alpha: f32,
        }

        let cases = [Case { alpha: 1.0 }, Case { alpha: 0.5 }];

        cases.test_each(|Case { alpha }| {
            let input = Tensor::from([-5., -2., -1., -0.5, 0., 0.5, 1., 2., 5.]);
            let reference_op = |&x: &f32| if x >= 0. { x } else { *alpha * (x.exp() - 1.) };
            test_unary_op_both(Elu { alpha: *alpha }, reference_op, input).unwrap();
        })
    }

    #[test]
    fn test_erf() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from([-2.0, -0.5, 0.5, 2.0]);
        let expected = Tensor::from([
            -0.9953222650189527,
            -0.5204998778130465,
            0.5204998778130465,
            0.9953222650189527,
        ]);
        let result = erf(&pool, input.view());
        expect_equal(&result, &expected)?;

        // Since we use a custom implementation of erf, do a test against the
        // standard library version with a lower tolerance than `expect_equal`s
        // default.
        //
        // `libm::erff` agrees with the higher precision `libm::erf` to a a
        // tolerance of ~1e-7. This implementation however agrees only to a
        // higher tolerance of ~1e-6. You should increase `samples` to a much
        // larger value if testing accuracy changes.
        let mut rng = RandomFloat::new(3456).with_range(-5., 5.);
        let samples = 1000;
        let input = Tensor::rand(&[samples], &mut rng);

        let expected = input.map(|x| libm::erff(*x));
        let result = erf(&pool, input.view());
        expect_equal_with_tolerance(&result, &expected, 1e-6, 0.)?;

        // Special values.
        let input = Tensor::from([f32::NAN, 0., f32::INFINITY, -f32::INFINITY]);
        let expected = Tensor::from([f32::NAN, 0., 1., -1.]);
        let result = erf(&pool, input.view());
        assert!(eq_with_nans(result.view(), expected.view()));

        Ok(())
    }

    test_unary_op!(
        test_exp,
        Exp {},
        |x| x.exp(),
        Tensor::from([-2., -0.5, 0.5, 2.0])
    );

    #[test]
    fn test_floor() {
        let pool = new_pool();
        let input = Tensor::from([
            1.,
            1.2,
            1.5,
            1.8,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        let expected = Tensor::from([
            1.,
            1.,
            1.,
            1.,
            0.,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        let result = floor(&pool, input.view());
        assert!(eq_with_nans(result.view(), expected.view()));
    }

    fn reference_gelu(x: f32) -> f32 {
        0.5 * x * (1. + libm::erff(x / (2.0f32).sqrt()))
    }

    fn reference_approx_gelu(x: f32) -> f32 {
        let x_cubed = x * x * x;
        let approx_erf = ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x_cubed)).tanh();
        0.5 * x * (1. + approx_erf)
    }

    test_unary_op!(test_gelu, Gelu { approximate: false }, |x| reference_gelu(
        *x
    ));
    test_unary_op!(test_approx_gelu, Gelu { approximate: true }, |x| {
        reference_approx_gelu(*x)
    });

    #[test]
    fn test_hard_sigmoid() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from([-4., -3., -1., 0., 1., 3., 4.]);
        let alpha = 0.2;
        let beta = 0.5;
        let pool = new_pool();
        let result = hard_sigmoid(&pool, input.view(), alpha, beta);
        let expected = Tensor::from([0., 0., -1. / 5. + 0.5, 0.5, 1. / 5. + 0.5, 1., 1.]);
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_hard_swish() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from([-4., -3., -1., 0., 1., 3., 4.]);
        let result = hard_swish(&pool, input.view());
        let expected = Tensor::from([0., 0., -1. / 3., 0., 2. / 3., 3., 4.]);
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_is_inf() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from([f32::NEG_INFINITY, 0., 1., f32::INFINITY]);
        let expected = Tensor::from([1i32, 0, 0, 1i32]);
        test_unary_op_not_in_place(&IsInf {}, input.view(), expected.view())
    }

    #[test]
    fn test_is_nan() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from([f32::NEG_INFINITY, 0., f32::NAN, 1., f32::INFINITY]);
        let expected = Tensor::from([0i32, 0, 1, 0, 0]);
        test_unary_op_not_in_place(&IsNaN {}, input.view(), expected.view())
    }

    #[test]
    fn test_leaky_relu() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = Tensor::from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        let result = leaky_relu(&pool, input.view(), alpha);
        expect_equal(&result, &expected)?;
        Ok(())
    }

    test_unary_op!(
        test_log,
        Log {},
        |x| x.ln(),
        Tensor::from([0.1, 0.5, 1., 10.])
    );

    test_unary_op!(test_neg, Neg {}, |x| -x, Tensor::from([0, 1, -1, 2]));

    test_unary_op!(
        test_not,
        Not {},
        |x| i32::from(*x == 0),
        Tensor::from([0, 1, 1, 0])
    );

    test_unary_op!(
        test_reciprocal,
        Reciprocal {},
        |x| 1. / x,
        Tensor::from([1., 2., 0.5, 0.])
    );

    test_unary_op!(
        test_relu,
        Relu {},
        |x| x.max(0.),
        Tensor::from([-0.5, 0.5, 3.0, -5.5])
    );

    #[test]
    fn test_round() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Example from ONNX spec.
        let input = Tensor::from([0.9, 2.5, 2.3, 1.5, -4.5]);
        let expected = Tensor::from([1., 2., 2., 2., -4.]);
        let result = round(&pool, input.view());
        expect_equal(&result, &expected)?;

        // Per spec, integral, zero, NaN and infinities are unchanged.
        let input = Tensor::from([1., 0., -0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let result = round(&pool, input.view());
        assert!(eq_with_nans(input.view(), result.view()));

        Ok(())
    }

    #[test]
    fn test_prelu() {
        let op = PRelu {};
        let x = Tensor::from([0., 0.5, 1.0, -0.5, -1.0]);
        let slope = Tensor::from([2.]);
        let expected = Tensor::from([0., 0.5, 1.0, -1.0, -2.0]);
        let result: Tensor = op.run_simple((x.view(), slope.view())).unwrap();
        assert_eq!(result, expected);

        let slope_2d = Tensor::from([[1.], [2.]]);
        let result: Result<Tensor, _> = op.run_simple((x.view(), slope_2d.view()));
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Slope is not broadcastable to input shape"
            ))
        );
    }

    test_unary_op!(test_sign, Sign {}, |x: &f32| x.signum());

    fn reference_sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    test_unary_op!(
        test_sigmoid,
        Sigmoid {},
        |x| reference_sigmoid(*x),
        Tensor::from([-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0])
    );
    test_unary_op!(test_silu, Silu {}, |x: &f32| x * reference_sigmoid(*x));
    test_unary_op!(test_sin, Sin {}, |x: &f32| x.sin());
    test_unary_op!(test_softplus, Softplus {}, |x: &f32| { x.exp().ln_1p() });
    test_unary_op!(
        test_sqrt,
        Sqrt {},
        |x: &f32| x.sqrt(),
        Tensor::from([4., 9., 16.])
    );
    test_unary_op!(test_swish, Swish { beta: 0.5 }, |x: &f32| x
        * reference_sigmoid(0.5 * *x));
    test_unary_op!(test_tan, Tan {}, |x: &f32| x.tan());
    test_unary_op!(test_tanh, Tanh {}, |x: &f32| x.tanh());
}

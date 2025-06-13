use rayon::prelude::*;

use std::any::Any;
use std::fmt::Debug;

use rten_simd::ops::GetNumOps;
use rten_simd::{Elem, SimdUnaryOp};
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView, TensorViewMut};
use rten_vecmath as vecmath;

use crate::number::AsBool;
use crate::ops::{
    map_value, map_value_view, IntoOpResult, OpError, OpRunContext, Operator, OutputList, Value,
    ValueView,
};
use crate::tensor_pool::{AutoReturn, TensorPool};

/// Trait for operators which take a single float tensor and apply a function
/// to each element.
pub trait UnaryFloatOp {
    fn name(&self) -> &str;

    /// Apply the operator to a single element.
    fn map_element(&self, val: f32) -> f32;

    /// Apply the operator to all elements in `input`.
    fn map(&self, pool: &TensorPool, input: TensorView) -> Tensor {
        input.map_in(pool, |val| self.map_element(*val))
    }

    /// Apply the operator to all elements in `input`.
    fn apply(&self, mut input: TensorViewMut) {
        input.apply(|val| self.map_element(*val))
    }
}

impl<Op: Any + Debug + UnaryFloatOp> Operator for Op {
    fn name(&self) -> &str {
        self.name()
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        self.map(ctx.pool(), input).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, _ctx: &OpRunContext) -> Result<Value, OpError> {
        let mut output: Tensor = input.try_into()?;
        self.apply(output.view_mut());
        Ok(output.into())
    }
}

/// Define a unary operator, with no arguments, which supports all numeric
/// tensor types.
///
/// The operator is defined by a name and generic functions which apply this
/// operator to 1) an immutable view and 2) a mutable tensor.
macro_rules! unary_numeric_op {
    ($name:ident, $view_impl:ident, $mut_impl:ident) => {
        #[derive(Debug)]
        pub struct $name {}

        impl Operator for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
                let input = ctx.inputs().require(0)?;
                map_value_view!(input, input, [FloatTensor, Int32Tensor], {
                    $view_impl(ctx.pool(), input).into_op_result()
                })
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
                map_value!(input, input, [FloatTensor, Int32Tensor], {
                    let result = $mut_impl(ctx.pool(), input);
                    Ok(result.into())
                })
            }
        }
    };
}

macro_rules! unary_float_funcs {
    ($name:ident, $func_name:ident) => {
        pub fn $func_name(pool: &TensorPool, input: TensorView) -> Tensor {
            $name {}.map(pool, input)
        }
    };
}

/// Define a unary operator, with no arguments, which supports all float tensor
/// types.
///
/// The operator is defined by the names for the operator struct and associated
/// functions, and a closure which evaluates the operator for a single function.
macro_rules! unary_float_op {
    ($name:ident, $func_name:ident, $expr:expr) => {
        unary_float_funcs!($name, $func_name);

        #[derive(Debug)]
        pub struct $name {}

        impl UnaryFloatOp for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn map_element(&self, val: f32) -> f32 {
                #[allow(clippy::redundant_closure_call)]
                $expr(val)
            }
        }
    };
}

/// Minimum number of elements in a chunk that is processed on a single thread.
const CHUNK_SIZE: usize = 32 * 1024;

/// Apply a unary operation in parallel to contiguous slices of `input`.
fn par_unary_op<T: GetNumOps + Elem + Send + Sync>(
    pool: &TensorPool,
    input: TensorView<T>,
    op: impl SimdUnaryOp<T> + Send + Sync,
) -> Tensor<T> {
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let mut output = Tensor::uninit_in(pool, input.shape());

    let in_chunks = input.data().unwrap().par_chunks(CHUNK_SIZE);
    let out_chunks = output.data_mut().unwrap().par_chunks_mut(CHUNK_SIZE);
    in_chunks
        .zip(out_chunks)
        .for_each(|(in_chunk, out_chunk)| op.map(in_chunk, out_chunk));

    // Safety: `op` initialized each chunk of `out_chunks`.
    unsafe { output.assume_init() }
}

/// Apply a unary operation in parallel to contiguous slices of `input`,
/// writing the results in-place.
fn par_unary_op_in_place<T: GetNumOps + Elem + Send + Sync>(
    pool: &TensorPool,
    mut input: Tensor<T>,
    op: impl SimdUnaryOp<T> + Send + Sync,
) -> Tensor<T> {
    if let Some(data) = input.data_mut() {
        data.par_chunks_mut(CHUNK_SIZE)
            .for_each(|chunk| op.map_mut(chunk));
        input
    } else {
        par_unary_op(pool, input.view(), op)
    }
}

/// Define an operator which supports float tensors and is optimized using SIMD
/// and multithreading.
macro_rules! parallel_unary_float_op {
    ($op_name:ident, $func_name:ident, $simd_kernel:expr) => {
        #[derive(Debug)]
        pub struct $op_name {}

        impl Operator for $op_name {
            fn name(&self) -> &str {
                stringify!($op_name)
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
                $func_name(ctx.pool(), ctx.inputs().require_as(0)?).into_op_result()
            }

            fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
                let tensor: Tensor = input.try_into()?;
                let kernel = $simd_kernel;
                let result = par_unary_op_in_place(ctx.pool(), tensor, kernel);
                Ok(result.into())
            }
        }

        pub fn $func_name(pool: &TensorPool, input: TensorView) -> Tensor {
            let kernel = $simd_kernel;
            par_unary_op(pool, input, kernel)
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

pub fn abs<T: AbsValue>(pool: &TensorPool, input: TensorView<T>) -> Tensor<T> {
    input.map_in(pool, |x| x.abs())
}

pub fn abs_in_place<T: AbsValue>(_pool: &TensorPool, mut input: Tensor<T>) -> Tensor<T> {
    input.apply(|x| x.abs());
    input
}

unary_numeric_op!(Abs, abs, abs_in_place);
unary_float_op!(Acos, acos, |val: f32| val.acos());
unary_float_op!(Asin, asin, |val: f32| val.asin());
unary_float_op!(Atan, atan, |val: f32| val.atan());
unary_float_op!(Ceil, ceil, |val: f32| val.ceil());

/// Numeric value with a finite minimum and maximum and operations to clamp
/// values.
pub trait Clamp: Copy + PartialOrd {
    /// Return the minimum possible finite value for this type.
    fn min_val() -> Self;

    /// Return the maximum possible finite value for this type.
    fn max_val() -> Self;

    /// Return the minimum of `self` and `val`.
    fn min(&self, val: Self) -> Self {
        if *self < val {
            *self
        } else {
            val
        }
    }

    /// Return the maximum of `self` and `val`.
    fn max(&self, val: Self) -> Self {
        if *self > val {
            *self
        } else {
            val
        }
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
    pool: &TensorPool,
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

unary_float_op!(Cos, cos, |val: f32| val.cos());

#[derive(Debug)]
pub struct Elu {
    pub alpha: f32,
}

impl UnaryFloatOp for Elu {
    fn name(&self) -> &str {
        "Elu"
    }

    fn map_element(&self, val: f32) -> f32 {
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

pub fn elu(pool: &TensorPool, input: TensorView, alpha: f32) -> Tensor {
    Elu { alpha }.map(pool, input)
}

parallel_unary_float_op!(Erf, erf, vecmath::Erf {});
parallel_unary_float_op!(Exp, exp, vecmath::Exp {});
unary_float_op!(Floor, floor, |val: f32| val.floor());

#[derive(Debug)]
pub struct Gelu {
    pub approximate: bool,
}

impl Operator for Gelu {
    fn name(&self) -> &str {
        "Gelu"
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        gelu(ctx.pool(), ctx.inputs().require_as(0)?, self.approximate).into_op_result()
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let tensor: Tensor = input.try_into()?;
        let result = if self.approximate {
            par_unary_op_in_place(ctx.pool(), tensor, vecmath::ApproxGelu {})
        } else {
            par_unary_op_in_place(ctx.pool(), tensor, vecmath::Gelu {})
        };
        Ok(result.into())
    }
}

pub fn gelu(pool: &TensorPool, input: TensorView, approximate: bool) -> Tensor {
    if approximate {
        par_unary_op(pool, input, vecmath::ApproxGelu {})
    } else {
        par_unary_op(pool, input, vecmath::Gelu {})
    }
}

#[derive(Debug)]
pub struct HardSigmoid {
    pub alpha: f32,
    pub beta: f32,
}

impl UnaryFloatOp for HardSigmoid {
    fn name(&self) -> &str {
        "HardSigmoid"
    }

    fn map_element(&self, val: f32) -> f32 {
        (self.alpha * val + self.beta).clamp(0., 1.)
    }
}

pub fn hard_sigmoid(pool: &TensorPool, input: TensorView, alpha: f32, beta: f32) -> Tensor {
    HardSigmoid { alpha, beta }.map(pool, input)
}

#[derive(Debug)]
pub struct HardSwish {}

impl UnaryFloatOp for HardSwish {
    fn name(&self) -> &str {
        "HardSwish"
    }

    fn map_element(&self, val: f32) -> f32 {
        let alpha = 1. / 6.;
        let beta = 0.5;
        val * HardSigmoid { alpha, beta }.map_element(val)
    }
}

unary_float_funcs!(HardSwish, hard_swish);

pub fn leaky_relu(pool: &TensorPool, input: TensorView, alpha: f32) -> Tensor {
    LeakyRelu { alpha }.map(pool, input)
}

#[derive(Debug)]
pub struct LeakyRelu {
    pub alpha: f32,
}

impl UnaryFloatOp for LeakyRelu {
    fn name(&self) -> &str {
        "LeakyRelu"
    }

    fn map_element(&self, val: f32) -> f32 {
        if val < 0.0 {
            self.alpha * val
        } else {
            val
        }
    }
}

unary_float_op!(Log, log, |val: f32| val.ln());

pub fn neg<T: Copy + std::ops::Neg<Output = T>>(
    pool: &TensorPool,
    input: TensorView<T>,
) -> Tensor<T> {
    input.map_in(pool, |x| x.neg())
}

pub fn neg_in_place<T: Copy + std::ops::Neg<Output = T>>(
    _pool: &TensorPool,
    mut input: Tensor<T>,
) -> Tensor<T> {
    input.apply(|x| x.neg());
    input
}

unary_numeric_op!(Neg, neg, neg_in_place);

pub fn not<T: AsBool + PartialEq>(pool: &TensorPool, input: TensorView<T>) -> Tensor<i32> {
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

unary_float_op!(Reciprocal, reciprocal, |val: f32| 1. / val);
unary_float_op!(Relu, relu, |val: f32| val.max(0.));

/// Round float values to the nearest integer. Values with a fractional part
/// of 0.5 are rounded to the nearest even number, like `round` in Python and
/// unlike [`f32::round`] in Rust.
#[derive(Debug)]
pub struct Round {}
impl UnaryFloatOp for Round {
    fn name(&self) -> &str {
        "Round"
    }

    fn map_element(&self, val: f32) -> f32 {
        val.round_ties_even()
    }
}

pub fn round(pool: &TensorPool, x: TensorView) -> Tensor {
    Round {}.map(pool, x)
}

parallel_unary_float_op!(Sigmoid, sigmoid, vecmath::Sigmoid {});

// Sigmoid Linear Unit (SiLU) function.
//
// This is a special case of the Swish function
// (<https://en.wikipedia.org/wiki/Swish_function>).
//
// Not an official ONNX operator, but used in popular object detection models.
// See https://github.com/onnx/onnx/issues/4854.
parallel_unary_float_op!(Silu, silu, vecmath::Silu {});

/// Swish function (<https://en.wikipedia.org/wiki/Swish_function>).
///
/// This computes `x * sigmoid(beta * x)`. The special case where beta = 1 is
/// known as [`Silu`].
#[derive(Debug)]
pub struct Swish {
    pub beta: f32,
}

impl Operator for Swish {
    fn name(&self) -> &str {
        "Swish"
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        swish(ctx.pool(), ctx.inputs().require_as(0)?, self.beta).into_op_result()
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let tensor: Tensor = input.try_into()?;
        let output = swish_in_place(ctx.pool(), tensor, self.beta);
        Ok(output.into())
    }
}

pub fn swish(pool: &TensorPool, input: TensorView, beta: f32) -> Tensor {
    par_unary_op(pool, input, vecmath::Swish { beta })
}

pub fn swish_in_place(pool: &TensorPool, input: Tensor, beta: f32) -> Tensor {
    par_unary_op_in_place(pool, input, vecmath::Swish { beta })
}

unary_float_op!(Sin, sin, |val: f32| val.sin());

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

pub fn sign<T: Signum>(pool: &TensorPool, input: TensorView<T>) -> Tensor<T> {
    input.map_in(pool, |x| x.signum())
}

pub fn sign_in_place<T: Signum>(_pool: &TensorPool, mut input: Tensor<T>) -> Tensor<T> {
    input.apply(|x| x.signum());
    input
}

unary_numeric_op!(Sign, sign, sign_in_place);
unary_float_op!(Sqrt, sqrt, |val: f32| val.sqrt());
unary_float_op!(Softplus, softplus, |val: f32| { val.exp().ln_1p() });
unary_float_op!(Tan, tan, |val: f32| val.tan());
parallel_unary_float_op!(Tanh, tanh, vecmath::Tanh {});

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{eq_with_nans, expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{RandomSource, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{
        ceil, clip, clip_in_place, erf, floor, hard_sigmoid, hard_swish, leaky_relu, round, Abs,
        Acos, Asin, Atan, Cos, Elu, Exp, Gelu, Log, Neg, Not, Reciprocal, Relu, Sigmoid, Sign,
        Silu, Sin, Softplus, Sqrt, Swish, Tan, Tanh,
    };
    use crate::ops::tests::new_pool;
    use crate::ops::{CastError, Operator, OperatorExt, Value, ValueView};
    use rten_tensor::test_util::ApproxEq;

    fn test_unary_op_impl<T: Clone + std::fmt::Debug + ApproxEq>(
        op: impl Operator,
        reference_op: impl Fn(&T) -> T,
        input: Tensor<T>,
    ) -> Result<(), Box<dyn Error>>
    where
        for<'a> TensorView<'a, T>: Into<ValueView<'a>>,
        Tensor<T>: Into<Value> + TryFrom<Value, Error = CastError>,
    {
        // Test copying variant.
        let expected = input.map(reference_op);
        let result: Tensor<T> = op.run_simple(input.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Test in-place variant.
        let input_mut = input.clone();
        let result: Tensor<T> = op.run_simple_in_place(input_mut, ()).unwrap();
        expect_equal(&result, &expected)?;

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
                test_unary_op_impl($op, $gen_expected, input)
            }
        };

        ($test_name:ident, $op:expr, $gen_expected:expr, $input:expr) => {
            #[test]
            fn $test_name() -> Result<(), Box<dyn Error>> {
                test_unary_op_impl($op, $gen_expected, $input)
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
        test_unary_op_impl(Abs {}, |x| x.abs(), [1., -1., 0.].into()).unwrap();
        test_unary_op_impl(Abs {}, |x| x.abs(), [1, -1, 0].into()).unwrap();
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
            test_unary_op_impl(Elu { alpha: *alpha }, reference_op, input).unwrap();
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

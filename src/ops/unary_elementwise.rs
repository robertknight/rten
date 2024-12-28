use rayon::prelude::*;

use std::any::Any;
use std::fmt::Debug;
use std::mem::MaybeUninit;

use rten_simd::dispatch::SimdUnaryOp;
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView, TensorViewMut};
use rten_vecmath as vecmath;

use crate::number::AsBool;
use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output, OutputList};
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        self.map(pool, input).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        let mut output = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
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

            fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
                let input = inputs.require(0)?;
                match input {
                    Input::FloatTensor(input) => $view_impl(pool, input).into_op_result(),
                    Input::Int32Tensor(input) => $view_impl(pool, input).into_op_result(),
                    _ => Err(OpError::UnsupportedType),
                }
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run_in_place(
                &self,
                _pool: &TensorPool,
                input: Output,
                _: InputList,
            ) -> Result<Output, OpError> {
                match input {
                    Output::FloatTensor(mut input) => {
                        $mut_impl(input.view_mut());
                        Ok(input.into())
                    }
                    Output::Int32Tensor(mut input) => {
                        $mut_impl(input.view_mut());
                        Ok(input.into())
                    }
                    _ => Err(OpError::UnsupportedType),
                }
            }
        }
    };
}

macro_rules! unary_float_funcs {
    ($name:ident, $func_name:ident, $in_place_func_name:ident) => {
        pub fn $func_name(pool: &TensorPool, input: TensorView) -> Tensor {
            $name {}.map(pool, input)
        }

        pub fn $in_place_func_name(input: TensorViewMut) {
            $name {}.apply(input)
        }
    };
}

/// Define a unary operator, with no arguments, which supports all float tensor
/// types.
///
/// The operator is defined by the names for the operator struct and associated
/// functions, and a closure which evaluates the operator for a single function.
macro_rules! unary_float_op {
    ($name:ident, $func_name:ident, $in_place_func_name:ident, $expr:expr) => {
        unary_float_funcs!($name, $func_name, $in_place_func_name);

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
fn par_unary_op<
    T: Copy + Default + Send + Sync,
    F: Fn(&[T], &mut [MaybeUninit<T>]) + Send + Sync,
>(
    pool: &TensorPool,
    input: TensorView<T>,
    op: F,
) -> Tensor<T> {
    let input = input.to_contiguous_in(pool).auto_return(pool);
    let mut output = Tensor::uninit_in(pool, input.shape());

    let in_chunks = input.data().unwrap().par_chunks(CHUNK_SIZE);
    let out_chunks = output.data_mut().unwrap().par_chunks_mut(CHUNK_SIZE);
    in_chunks
        .zip(out_chunks)
        .for_each(|(in_chunk, out_chunk)| op(in_chunk, out_chunk));

    // Safety: `op` initialized each chunk of `out_chunks`.
    unsafe { output.assume_init() }
}

/// Apply a unary operation in parallel to contiguous slices of `input`,
/// writing the results in-place.
fn par_unary_op_in_place<T: Copy + Send, VF: Fn(&mut [T]) + Send + Sync, SF: Fn(T) -> T>(
    mut input: TensorViewMut<T>,
    vec_op: VF,
    scalar_op: SF,
) {
    if let Some(data) = input.data_mut() {
        data.par_chunks_mut(CHUNK_SIZE).for_each(vec_op);
    } else {
        input.apply(|x| scalar_op(*x));
    }
}

/// Define an operator which supports float tensors and is optimize using SIMD
/// and multithreading.
macro_rules! parallel_unary_float_op {
    ($op_name:ident, $func_name:ident, $in_place_func_name:ident, $impl_func:expr, $impl_in_place_func:expr, $impl_scalar:expr) => {
        #[derive(Debug)]
        pub struct $op_name {}

        impl Operator for $op_name {
            fn name(&self) -> &str {
                stringify!($op_name)
            }

            fn can_run_in_place(&self) -> bool {
                true
            }

            fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
                $func_name(pool, inputs.require_as(0)?).into_op_result()
            }

            fn run_in_place(
                &self,
                _pool: &TensorPool,
                input: Output,
                _: InputList,
            ) -> Result<Output, OpError> {
                let mut tensor = input
                    .into_tensor::<f32>()
                    .ok_or(OpError::IncorrectInputType)?;
                $in_place_func_name(tensor.view_mut());
                Ok(tensor.into())
            }
        }

        pub fn $func_name(pool: &TensorPool, input: TensorView) -> Tensor {
            par_unary_op(pool, input, $impl_func)
        }

        pub fn $in_place_func_name(input: TensorViewMut) {
            par_unary_op_in_place(input, $impl_in_place_func, $impl_scalar);
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

pub fn abs_in_place<T: AbsValue>(mut input: TensorViewMut<T>) {
    input.apply(|x| x.abs())
}

unary_numeric_op!(Abs, abs, abs_in_place);
unary_float_op!(Acos, acos, acos_in_place, |val: f32| val.acos());
unary_float_op!(Asin, asin, asin_in_place, |val: f32| val.asin());
unary_float_op!(Atan, atan, atan_in_place, |val: f32| val.atan());
unary_float_op!(Ceil, ceil, ceil_in_place, |val: f32| val.ceil());

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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        match input {
            Input::FloatTensor(input) => {
                let min = inputs.get_as_scalar(1)?;
                let max = inputs.get_as_scalar(2)?;
                clip(pool, input, min, max).into_op_result()
            }
            Input::Int32Tensor(input) => {
                let min = inputs.get_as_scalar(1)?;
                let max = inputs.get_as_scalar(2)?;
                clip(pool, input, min, max).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        match input {
            Output::FloatTensor(mut input) => {
                let min = other.get_as_scalar(0)?;
                let max = other.get_as_scalar(1)?;
                clip_in_place(&mut input, min, max);
                Ok(input.into())
            }
            Output::Int32Tensor(mut input) => {
                let min = other.get_as_scalar(0)?;
                let max = other.get_as_scalar(1)?;
                clip_in_place(&mut input, min, max);
                Ok(input.into())
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

unary_float_op!(Cos, cos, cos_in_place, |val: f32| val.cos());

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

pub fn elu_in_place(input: TensorViewMut, alpha: f32) {
    Elu { alpha }.apply(input)
}

parallel_unary_float_op!(
    Erf,
    erf,
    erf_in_place,
    |src, dest| vecmath::Erf {}.map(src, dest),
    |src| vecmath::Erf {}.map_mut(src),
    |x| vecmath::Erf {}.scalar_eval(x)
);
parallel_unary_float_op!(
    Exp,
    exp,
    exp_in_place,
    |src, dest| vecmath::Exp {}.map(src, dest),
    |src| vecmath::Exp {}.map_mut(src),
    |x| vecmath::Exp {}.scalar_eval(x)
);
unary_float_op!(Floor, floor, floor_in_place, |val: f32| val.floor());

parallel_unary_float_op!(
    Gelu,
    gelu,
    gelu_in_place,
    |src, dest| vecmath::Gelu {}.map(src, dest),
    |src| vecmath::Gelu {}.map_mut(src),
    |x| vecmath::Gelu {}.scalar_eval(x)
);

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

pub fn hard_sigmoid_in_place(input: TensorViewMut, alpha: f32, beta: f32) {
    HardSigmoid { alpha, beta }.apply(input)
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

unary_float_funcs!(HardSwish, hard_swish, hard_swish_in_place);

pub fn leaky_relu(pool: &TensorPool, input: TensorView, alpha: f32) -> Tensor {
    LeakyRelu { alpha }.map(pool, input)
}

pub fn leaky_relu_in_place(input: TensorViewMut, alpha: f32) {
    LeakyRelu { alpha }.apply(input)
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

unary_float_op!(Log, log, log_in_place, |val: f32| val.ln());

pub fn neg<T: Copy + std::ops::Neg<Output = T>>(
    pool: &TensorPool,
    input: TensorView<T>,
) -> Tensor<T> {
    input.map_in(pool, |x| x.neg())
}

pub fn neg_in_place<T: Copy + std::ops::Neg<Output = T>>(mut input: TensorViewMut<T>) {
    input.apply(|x| x.neg())
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as::<i32>(0)?;
        not(pool, input).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        let mut output = input
            .into_tensor::<i32>()
            .ok_or(OpError::IncorrectInputType)?;
        not_in_place(output.view_mut());
        Ok(output.into())
    }
}

unary_float_op!(Reciprocal, reciprocal, reciprocal_in_place, |val: f32| 1.
    / val);
unary_float_op!(Relu, relu, relu_in_place, |val: f32| val.max(0.));

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

pub fn round_in_place(x: TensorViewMut) {
    Round {}.apply(x)
}

parallel_unary_float_op!(
    Sigmoid,
    sigmoid,
    sigmoid_in_place,
    |src, dest| vecmath::Sigmoid {}.map(src, dest),
    |src| vecmath::Sigmoid {}.map_mut(src),
    |x| vecmath::Sigmoid {}.scalar_eval(x)
);

// Sigmoid Linear Unit (SiLU) function.
//
// This is a special case of the Swish function
// (<https://en.wikipedia.org/wiki/Swish_function>).
//
// Not an official ONNX operator, but used in popular object detection models.
// See https://github.com/onnx/onnx/issues/4854.
parallel_unary_float_op!(
    Silu,
    silu,
    silu_in_place,
    |src, dest| vecmath::Silu {}.map(src, dest),
    |src| vecmath::Silu {}.map_mut(src),
    |x| vecmath::Silu {}.scalar_eval(x)
);

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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        swish(pool, inputs.require_as(0)?, self.beta).into_op_result()
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        let mut tensor = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
        swish_in_place(tensor.view_mut(), self.beta);
        Ok(tensor.into())
    }
}

pub fn swish(pool: &TensorPool, input: TensorView, beta: f32) -> Tensor {
    let swish = vecmath::Swish { beta };
    par_unary_op(pool, input, |src, dest| swish.map(src, dest))
}

pub fn swish_in_place(input: TensorViewMut, beta: f32) {
    let swish = vecmath::Swish { beta };
    par_unary_op_in_place(input, |src| swish.map_mut(src), |x| swish.scalar_eval(x));
}

unary_float_op!(Sin, sin, sin_in_place, |val: f32| val.sin());

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

pub fn sign_in_place<T: Signum>(mut input: TensorViewMut<T>) {
    input.apply(|x| x.signum())
}

unary_numeric_op!(Sign, sign, sign_in_place);
unary_float_op!(Sqrt, sqrt, sqrt_in_place, |val: f32| val.sqrt());
unary_float_op!(Softplus, softplus, softplus_in_place, |val: f32| {
    val.exp().ln_1p()
});
unary_float_op!(Tan, tan, tan_in_place, |val: f32| val.tan());
parallel_unary_float_op!(
    Tanh,
    tanh,
    tanh_in_place,
    |src, dest| vecmath::Tanh {}.map(src, dest),
    |src| vecmath::Tanh {}.map_mut(src),
    |x| vecmath::Tanh {}.scalar_eval(x)
);

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{eq_with_nans, expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{RandomSource, Tensor};

    use crate::ops::tests::new_pool;
    use crate::ops::{
        abs, acos, acos_in_place, asin, asin_in_place, atan, atan_in_place, ceil, clip,
        clip_in_place, cos, cos_in_place, elu, elu_in_place, erf, erf_in_place, exp, exp_in_place,
        floor, gelu, gelu_in_place, hard_sigmoid, hard_swish, leaky_relu, leaky_relu_in_place, log,
        log_in_place, neg, neg_in_place, not, not_in_place, reciprocal, relu, relu_in_place, round,
        round_in_place, sigmoid, sigmoid_in_place, sign, sign_in_place, silu, silu_in_place, sin,
        sin_in_place, softplus, softplus_in_place, sqrt, sqrt_in_place, swish, swish_in_place, tan,
        tan_in_place, tanh, tanh_in_place,
    };

    /// Define a test for a simple unary operator which applies the function
    /// `$gen_expected` to each input element.
    macro_rules! test_unary_op {
        ($test_name:ident, $op:expr, $in_place_op:expr, $gen_expected:expr) => {
            #[test]
            fn $test_name() -> Result<(), Box<dyn Error>> {
                let pool = new_pool();

                // Test inputs here chosen to be in the domain of inverse trig
                // operators (ie. (-1, 1)).
                let input = Tensor::from([0., 0.1, -0.1, 0.9, -0.9]);
                let expected = input.map($gen_expected);
                let result = $op(&pool, input.view());
                expect_equal(&result, &expected)?;

                let mut input = input.clone();
                $in_place_op(input.view_mut());
                expect_equal(&input, &expected)?;

                Ok(())
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
        let pool = new_pool();

        // Float tensor
        let x: Tensor<f32> = Tensor::from([1., -1., 0.]);
        let result = abs(&pool, x.view());
        assert_eq!(result, Tensor::from([1., 1., 0.]));

        // Int tensor
        let x: Tensor<i32> = Tensor::from([1, -1, 0]);
        let result = abs(&pool, x.view());
        assert_eq!(result, Tensor::from([1, 1, 0]));
    }

    test_unary_op!(test_acos, acos, acos_in_place, |x: &f32| x.acos());
    test_unary_op!(test_asin, asin, asin_in_place, |x: &f32| x.asin());
    test_unary_op!(test_atan, atan, atan_in_place, |x: &f32| x.atan());

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
    fn test_clip() -> Result<(), Box<dyn Error>> {
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

        let pool = new_pool();
        for case in cases {
            let result = clip(&pool, case.input.view(), case.min, case.max);
            expect_equal(&result, &case.expected)?;

            let mut input = case.input.clone();
            clip_in_place(&mut input, case.min, case.max);
            expect_equal(&input, &case.expected)?;
        }

        Ok(())
    }

    // TODO: Eliminate the duplication for tests that apply the operator
    // in-place vs returning a new tensor.

    test_unary_op!(test_cos, cos, cos_in_place, |x: &f32| x.cos());

    #[test]
    fn test_elu() -> Result<(), Box<dyn Error>> {
        struct Case {
            alpha: f32,
        }

        let cases = [Case { alpha: 1.0 }, Case { alpha: 0.5 }];

        let pool = new_pool();
        for Case { alpha } in cases {
            let input = Tensor::from([-5., -2., -1., -0.5, 0., 0.5, 1., 2., 5.]);
            let expected = input.map(|&x: &f32| if x >= 0. { x } else { alpha * (x.exp() - 1.) });

            let actual = elu(&pool, input.view(), alpha);
            expect_equal(&actual, &expected)?;

            let mut input = input.clone();
            elu_in_place(input.view_mut(), alpha);
            expect_equal(&input, &expected)?;
        }

        Ok(())
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

    #[test]
    fn test_erf_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from([-2.0, -0.5, 0.5, 2.0]);
        let expected = Tensor::from([
            -0.9953222650189527,
            -0.5204998778130465,
            0.5204998778130465,
            0.9953222650189527,
        ]);
        erf_in_place(input.view_mut());
        expect_equal(&input, &expected)?;
        Ok(())
    }

    #[test]
    fn test_exp() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from([-2.0, -0.5, 0.5, 2.0]);
        let expected = Tensor::from([
            0.1353352832366127,
            0.6065306597126334,
            1.6487212707001282,
            7.38905609893065,
        ]);
        let result = exp(&pool, input.view());
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_exp_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from([-2.0, -0.5, 0.5, 2.0]);
        let expected = Tensor::from([
            0.1353352832366127,
            0.6065306597126334,
            1.6487212707001282,
            7.38905609893065,
        ]);
        exp_in_place(input.view_mut());
        expect_equal(&input, &expected)?;
        Ok(())
    }

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
    test_unary_op!(test_gelu, gelu, gelu_in_place, |x| reference_gelu(*x));

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

    #[test]
    fn test_leaky_relu_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from_data(&[2, 2], vec![-5., -2., 3., 20.]);
        let alpha = 0.1;
        let expected = Tensor::from_data(&[2, 2], vec![-5. * alpha, -2. * alpha, 3., 20.]);
        leaky_relu_in_place(input.view_mut(), alpha);
        expect_equal(&input, &expected)?;
        Ok(())
    }

    #[test]
    fn test_log() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from([0.1, 0.5, 1., 10.]);
        let expected = Tensor::from([
            -2.3025850929940455,
            -0.6931471805599453,
            0.,
            2.302585092994046,
        ]);
        let result = log(&pool, input.view());
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_log_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from([0.1, 0.5, 1., 10.]);
        let expected = Tensor::from([
            -2.3025850929940455,
            -0.6931471805599453,
            0.,
            2.302585092994046,
        ]);
        log_in_place(input.view_mut());
        expect_equal(&input, &expected)?;
        Ok(())
    }

    #[test]
    fn test_neg() {
        let pool = new_pool();
        let input = Tensor::from([0, 1, -1, 2]);
        let expected = Tensor::from([0, -1, 1, -2]);
        let result = neg(&pool, input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_neg_in_place() {
        let mut input = Tensor::from([0, 1, -1, 2]);
        let expected = Tensor::from([0, -1, 1, -2]);
        neg_in_place(input.view_mut());
        assert_eq!(input, expected);
    }

    #[test]
    fn test_not() {
        let pool = new_pool();
        let input = Tensor::from([0, 1, 1, 0]);
        let expected = Tensor::from([1, 0, 0, 1]);
        let result = not(&pool, input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_not_in_place() {
        let mut input = Tensor::from([0, 1, 1, 0]);
        let expected = Tensor::from([1, 0, 0, 1]);
        not_in_place(input.view_mut());
        assert_eq!(input, expected);
    }

    #[test]
    fn test_reciprocal() {
        let pool = new_pool();
        let input = Tensor::from([1., 2., 0.5, 0.]);
        let expected = input.map(|x| 1. / x);
        let result = reciprocal(&pool, input.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_relu() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2, 1], vec![-0.5, 0.5, 3.0, -5.5]);
        let expected = Tensor::from_data(&[2, 2, 1], vec![0.0, 0.5, 3.0, 0.0]);

        let result = relu(&pool, input.view());
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        relu_in_place(result.view_mut());
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_round() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Example from ONNX spec.
        let input = Tensor::from([0.9, 2.5, 2.3, 1.5, -4.5]);
        let expected = Tensor::from([1., 2., 2., 2., -4.]);
        let result = round(&pool, input.view());
        expect_equal(&result, &expected)?;

        let mut input = input.clone();
        round_in_place(input.view_mut());
        expect_equal(&input, &expected)?;

        // Per spec, integral, zero, NaN and infinities are unchanged.
        let input = Tensor::from([1., 0., -0., f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let result = round(&pool, input.view());
        assert!(eq_with_nans(input.view(), result.view()));

        Ok(())
    }

    fn reference_sigmoid(x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    #[test]
    fn test_sigmoid() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input: Tensor<f32> =
            Tensor::from([-500.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 500.0]);
        let expected = input.map(|x| reference_sigmoid(*x));

        let result = sigmoid(&pool, input.view());
        expect_equal(&result, &expected)?;

        let mut result = input.clone();
        sigmoid_in_place(result.view_mut());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    test_unary_op!(test_silu, silu, silu_in_place, |x: &f32| x
        * reference_sigmoid(*x));
    test_unary_op!(
        test_swish,
        |pool, input| swish(pool, input, 0.5),
        |input| swish_in_place(input, 0.5),
        |x: &f32| x * reference_sigmoid(0.5 * *x)
    );
    test_unary_op!(test_sign, sign, sign_in_place, |x: &f32| x.signum());
    test_unary_op!(test_sin, sin, sin_in_place, |x: &f32| x.sin());
    test_unary_op!(test_softplus, softplus, softplus_in_place, |x: &f32| {
        x.exp().ln_1p()
    });

    #[test]
    fn test_sqrt() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from([4., 9., 16.]);
        let expected = Tensor::from([2., 3., 4.]);
        let result = sqrt(&pool, input.view());
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_sqrt_in_place() -> Result<(), Box<dyn Error>> {
        let mut input = Tensor::from([4., 9., 16.]);
        let expected = Tensor::from([2., 3., 4.]);
        sqrt_in_place(input.view_mut());
        expect_equal(&input, &expected)?;
        Ok(())
    }

    test_unary_op!(test_tan, tan, tan_in_place, |x: &f32| x.tan());
    test_unary_op!(test_tanh, tanh, tanh_in_place, |x: &f32| x.tanh());
}

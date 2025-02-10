use smallvec::SmallVec;
use std::fmt::Debug;
use std::iter::repeat;

use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView, TensorViewMut};

use crate::number::{AsBool, Identities, IsInt};
use crate::ops::{
    map_input, map_output, Input, InputList, IntoOpResult, OpError, Operator, Output, OutputList,
};
use crate::tensor_pool::TensorPool;

/// Given the shapes of two inputs to a binary operation, return the shape
/// that will result from broadcasting them following NumPy rules or `None`
/// if the shapes are not compatible.
///
/// Broadcasting works by left-padding the input shapes with 1s so they are
/// the same length, then matching dimensions starting from the right. For
/// each dimension, the values are compatible if they are the same or one of
/// them is 1. The larger of the two values is the size of that dimension in
/// the output shape.
///
/// See https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<SmallVec<[usize; 4]>> {
    let a_pad = b.len().saturating_sub(a.len());
    let b_pad = a.len().saturating_sub(b.len());

    let a_iter = a.iter().copied().rev().chain(repeat(1).take(a_pad));
    let b_iter = b.iter().copied().rev().chain(repeat(1).take(b_pad));

    let mut result = SmallVec::with_capacity(a.len().max(b.len()));
    for (a, b) in a_iter.zip(b_iter) {
        if a == b {
            result.push(a);
        } else if a == 1 {
            result.push(b);
        } else if b == 1 {
            result.push(a);
        } else {
            return None;
        }
    }
    result.reverse();

    Some(result)
}

/// Return true if an elementwise binary operation can be performed in-place
/// on `a` given `b` as the other argument.
fn can_run_binary_op_in_place<L1: Layout, L2: Layout>(a: &L1, b: &L2) -> bool {
    b.can_broadcast_to(a.shape().as_ref())
}

/// Check whether a tensor of shape `from_shape` can be broadcast to `to_shape`
/// using fast broadcasting. Fast broadcasting is possible when:
///
///  - The view being broadcast has a contiguous layout (checked outside this function)
///  - All the dimensions being broadcast are leading and/or trailing.
///
/// In this case broadcasting can be done by repeating and cycling through
/// elements. Each element is repeated to broadcast trailing dims, and the
/// whole sequence is cycled to broadcast leading dims.
///
/// Returns a tuple of `(cycles, repeats)` indicating the number of element
/// repeats and sequence cycles needed.
pub fn fast_broadcast_cycles_repeats(
    from_shape: &[usize],
    to_shape: &[usize],
) -> Option<(usize, usize)> {
    if from_shape == to_shape {
        return Some((1, 1));
    }

    // When there is only one item, we have a choice of whether to return
    // `Some(cycles, 1)` or `Some(1, repeats)`. The calling code is optimized
    // to prefer the latter.
    if from_shape.iter().product::<usize>() == 1 {
        return Some((1, to_shape.iter().product()));
    }

    assert!(to_shape.len() >= from_shape.len());

    // Implicitly left-pad `from_shape` with 1s to match length of `to_shape`.
    let from_pad = to_shape.len() - from_shape.len();
    let from_size = |dim| {
        if dim < from_pad {
            1
        } else {
            from_shape[dim - from_pad]
        }
    };

    let mut leading_1s = 0; // Common leading 1s in both shapes
    let mut leading_bcast = 0; // Leading dims to broadcast
    let mut cycles = 1;
    for (i, to) in to_shape.iter().copied().enumerate() {
        let from = from_size(i);
        if from == 1 && to == 1 {
            leading_1s += 1;
        } else if from == 1 && to > 1 {
            leading_bcast += 1;
            cycles *= to;
        } else {
            break;
        }
    }

    let mut trailing_1s = 0; // Common trailing 1s in both shapes
    let mut trailing_bcast = 0; // Trailing dims to broadcast
    let mut repeats = 1;
    for (i, to) in to_shape.iter().copied().enumerate().rev() {
        let from = from_size(i);
        if from == 1 && to == 1 {
            trailing_1s += 1;
        } else if from == 1 && to > 1 {
            trailing_bcast += 1;
            repeats *= to;
        } else {
            break;
        }
    }

    for i in (leading_1s + leading_bcast)..(to_shape.len() - trailing_1s - trailing_bcast) {
        let from = from_size(i);
        let to = to_shape[i];
        if from != to {
            // A middle dimension that is sandwiched between non-broadcasted
            // dims needs to be broadcast. We can't use fast broadcasting :(
            return None;
        }
    }

    Some((cycles, repeats))
}

/// Compute the result of applying the binary operation `op` to corresponding
/// elements of `a` and `b`. The shapes of `a` and `b` are broadcast to a
/// matching shape if necessary.
pub fn binary_op<T: Copy, R, F: Fn(T, T) -> R>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
    op: F,
) -> Result<Tensor<R>, OpError> {
    let out_shape = broadcast_shapes(a.shape(), b.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    // Fast path for when LHS and RHS are contiguous, and fast broadcasting is
    // possible.
    if let (true, Some(a_data), Some(b_data)) =
        (a.shape() == out_shape.as_slice(), a.data(), b.data())
    {
        if let Some((cycles, repeats)) = fast_broadcast_cycles_repeats(b.shape(), a.shape()) {
            assert!(cycles * b_data.len() * repeats == a.len());

            let mut output = Tensor::uninit_in(pool, &out_shape);

            // Unsafe access used to skip bounds checks in inner loop.
            let out_data = output.data_mut().unwrap();
            let a_ptr = a_data.as_ptr();

            let mut i = 0;
            for _ in 0..cycles {
                if repeats == 1 {
                    for b_elt in b_data {
                        // Safety: We checked the total loop count is in `[0,
                        // out_data.len())` above, which is the same as
                        // `a_data.len().
                        let (a_elt, out_elt) =
                            unsafe { (*a_ptr.add(i), out_data.get_unchecked_mut(i)) };
                        out_elt.write(op(a_elt, *b_elt));
                        i += 1;
                    }
                } else {
                    for b_elt in b_data {
                        for _ in 0..repeats {
                            // Safety: We checked the total loop count is in `[0,
                            // out_data.len())` above, which is the same as
                            // `a_data.len().
                            let (a_elt, out_elt) =
                                unsafe { (*a_ptr.add(i), out_data.get_unchecked_mut(i)) };
                            out_elt.write(op(a_elt, *b_elt));
                            i += 1;
                        }
                    }
                }
            }

            // Safety: We initialized all output elements.
            assert!(i == output.len());
            let output = unsafe { output.assume_init() };
            return Ok(output);
        }
    }

    let mut a = a.broadcast(out_shape.as_slice());
    let mut b = b.broadcast(out_shape.as_slice());
    let mut out_data = pool.alloc(a.len());
    let out_uninit = &mut out_data.spare_capacity_mut()[..a.len()];
    let mut out_offset = 0;

    // Loop over a statically known number of inner dims for efficiency.
    while a.ndim() <= 4 {
        a.insert_axis(0);
        b.insert_axis(0);
    }

    a.inner_iter::<4>()
        .zip(b.inner_iter::<4>())
        .for_each(|(a, b)| {
            for i0 in 0..a.size(0) {
                for i1 in 0..a.size(1) {
                    for i2 in 0..a.size(2) {
                        for i3 in 0..a.size(3) {
                            // Safety:
                            // - `a` and `b` have the same shape, and i0..i3 are in `[0, a.size(i))`.
                            // - The length of `out_uninit` is the same as `a.len()`.
                            unsafe {
                                let a_elt = a.get_unchecked([i0, i1, i2, i3]);
                                let b_elt = b.get_unchecked([i0, i1, i2, i3]);
                                out_uninit
                                    .get_unchecked_mut(out_offset)
                                    .write(op(*a_elt, *b_elt));
                                out_offset += 1;
                            }
                        }
                    }
                }
            }
        });

    // Safety: We initialized `out_offset` elements.
    unsafe {
        out_data.set_len(out_offset);
    }
    Ok(Tensor::from_data(&out_shape, out_data))
}

/// Perform an elementwise binary operation in-place.
///
/// This requires that `b` can be broadcast to the shape of `a`.
fn binary_op_in_place<T: Copy + Debug, F: Fn(T, T) -> T>(
    mut a: TensorViewMut<T>,
    b: TensorView<T>,
    op: F,
) {
    // Fast path for when LHS and RHS are contiguous, and fast broadcasting is
    // possible.
    if let (true, Some(b_data)) = (a.is_contiguous(), b.data()) {
        if let Some((cycles, repeats)) = fast_broadcast_cycles_repeats(b.shape(), a.shape()) {
            assert!(cycles * b_data.len() * repeats == a.len());
            let a_data = a.data_mut().unwrap();
            let mut i = 0;
            for _ in 0..cycles {
                if repeats == 1 {
                    for b_elt in b_data {
                        // Safety: We checked the total loop count is in `[0, a.len())` above.
                        let a_elt = unsafe { a_data.get_unchecked_mut(i) };
                        *a_elt = op(*a_elt, *b_elt);
                        i += 1;
                    }
                } else {
                    for b_elt in b_data {
                        for _ in 0..repeats {
                            // Safety: We checked the total loop count is in `[0, a.len())` above.
                            let a_elt = unsafe { a_data.get_unchecked_mut(i) };
                            *a_elt = op(*a_elt, *b_elt);
                            i += 1;
                        }
                    }
                }
            }
            return;
        }
    }

    // Loop over a statically known number of inner dims for efficiency.
    let mut b = b.broadcast(a.shape());
    while a.ndim() <= 4 {
        a.insert_axis(0);
        b.insert_axis(0);
    }

    a.inner_iter_mut::<4>()
        .zip(b.inner_iter::<4>())
        .for_each(|(mut a, b)| {
            for i0 in 0..a.size(0) {
                for i1 in 0..a.size(1) {
                    for i2 in 0..a.size(2) {
                        for i3 in 0..a.size(3) {
                            // Safety: `a` and `b` have the same shape, and
                            // i0..i3 are in `[0, a.size(i))`.
                            unsafe {
                                let a_elt = a.get_unchecked_mut([i0, i1, i2, i3]);
                                let b_elt = b.get_unchecked([i0, i1, i2, i3]);
                                *a_elt = op(*a_elt, *b_elt);
                            }
                        }
                    }
                }
            }
        });
}

/// Perform a commutative elementwise binary operation.
///
/// This is an optimized alternative to `binary_op` for the case where the
/// operands can be swapped without affecting the result. In this case we
/// can make the larger of the two operands the LHS and benefit from
/// optimizations in `binary_op` that assume this.
fn binary_commutative_op<T: Copy + Debug + Default, F: Fn(T, T) -> T>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
    op: F,
) -> Result<Tensor<T>, OpError> {
    if b.len() > a.len() {
        // `a` must be broadcast to `b`s shape. Swap operands so we can take
        // potentially take advantage of fast paths for this.
        binary_op(pool, b, a, op)
    } else {
        binary_op(pool, a, b, op)
    }
}

/// Extract two input operands from `$inputs` and invoke the appropriate
/// instantiation of `$op_func` depending on the tensor type.
macro_rules! run_typed_op {
    ($pool:expr, $inputs:expr, $op_func:ident) => {{
        let a = $inputs.require(0)?;
        map_input!(a, a, [FloatTensor, Int32Tensor], {
            let b = $inputs.require_as(1)?;
            $op_func($pool, a, b).into_op_result()
        })
    }};
    ($inputs:expr, $op_func:ident) => {
        run_typed_op!(&TensorPool::new(), $inputs, $op_func)
    };
}

/// Extract two input operands from `$input` and `$other` and invoke the
/// appropriate instantiations of `$in_place_op_func` or `$op_func` depending
/// on the tensor type.
macro_rules! run_typed_op_in_place {
    ($pool:expr, $input:expr, $other: expr, $in_place_op_func:ident, $op_func:ident) => {{
        map_output!($input, a, [FloatTensor, Int32Tensor], {
            let b = $other.require_as(0)?;
            if can_run_binary_op_in_place(&a, &b) {
                $in_place_op_func(a.view_mut(), b);
                Ok(a.into())
            } else {
                $op_func($pool, a.view(), b.view()).map(|t| t.into())
            }
        })
    }};
}

/// Perform elementwise addition of two tensors.
pub fn add<T: Copy + Debug + Default + std::ops::Add<Output = T>>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(pool, a, b, |x, y| x + y)
}

/// Perform in-place elementwise addition of two tensors.
pub fn add_in_place<T: Copy + Debug + std::ops::Add<Output = T>>(
    a: TensorViewMut<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |x, y| x + y);
}

#[derive(Debug)]
pub struct Add {}

impl Operator for Add {
    fn name(&self) -> &str {
        "Add"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        run_typed_op!(pool, inputs, add)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        run_typed_op_in_place!(pool, input, other, add_in_place, add)
    }
}

/// Define a logical boolean operator.
///
/// These accept two i32 tensors and produce an i32 result.
macro_rules! logical_boolean_op {
    ($op:ident, $op_fn:ident, $expr:expr) => {
        pub fn $op_fn<T: AsBool + Copy + Debug>(
            pool: &TensorPool,
            a: TensorView<T>,
            b: TensorView<T>,
        ) -> Result<Tensor<i32>, OpError> {
            #[allow(clippy::redundant_closure_call)]
            binary_op(pool, a, b, |x, y| $expr(x.as_bool(), y.as_bool()).into())
        }

        #[derive(Debug)]
        pub struct $op {}

        impl Operator for $op {
            fn name(&self) -> &str {
                stringify!($op)
            }

            fn is_commutative(&self) -> bool {
                // These ops are marked as commutative because that is
                // technically true, but this will have no effect until
                // `run_in_place` is implemented.
                true
            }

            fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
                let a: TensorView<i32> = inputs.require_as(0)?;
                let b: TensorView<i32> = inputs.require_as(1)?;
                $op_fn(pool, a, b).into_op_result()
            }
        }
    };
}

logical_boolean_op!(And, and, |x, y| x && y);
logical_boolean_op!(Or, or, |x, y| x || y);
logical_boolean_op!(Xor, xor, |x, y| x ^ y);

/// Perform elementwise division of two tensors.
pub fn div<
    T: Copy
        + Debug
        + Default
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + IsInt
        + Identities,
>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    match (T::is_int(), b.item()) {
        // Optimize division as multiplication-by-reciprocal.
        //
        // This loses some precision, so we might want to revisit this in future.
        (false, Some(scalar)) => mul(pool, a, Tensor::from_scalar(T::one() / *scalar).view()),
        _ => binary_op(pool, a, b, |x, y| x / y),
    }
}

/// Perform in-place elementwise division of two tensors.
pub fn div_in_place<
    T: Copy + Debug + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + IsInt + Identities,
>(
    a: TensorViewMut<T>,
    b: TensorView<T>,
) {
    match (T::is_int(), b.item()) {
        (false, Some(scalar)) => mul_in_place(a, Tensor::from_scalar(T::one() / *scalar).view()),
        _ => binary_op_in_place(a, b, |x, y| x / y),
    }
}

#[derive(Debug)]
pub struct Div {}

impl Operator for Div {
    fn name(&self) -> &str {
        "Div"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        run_typed_op!(pool, inputs, div)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        run_typed_op_in_place!(pool, input, other, div_in_place, div)
    }
}

enum BooleanOp {
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

fn boolean_op<T: Copy + Debug + PartialEq + PartialOrd>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
    op: BooleanOp,
) -> Result<Tensor<i32>, OpError> {
    binary_op(pool, a, b, |x, y| {
        i32::from(match op {
            BooleanOp::Equal => x == y,
            BooleanOp::Less => x < y,
            BooleanOp::LessOrEqual => x <= y,
            BooleanOp::Greater => x > y,
            BooleanOp::GreaterOrEqual => x >= y,
        })
    })
}

/// Define a boolean comparison operator which supports all numeric tensor
/// types.
macro_rules! boolean_cmp_op {
    ($name:ident, $func:ident) => {
        pub fn $func<T: Copy + Debug + PartialEq + PartialOrd>(
            pool: &TensorPool,
            a: TensorView<T>,
            b: TensorView<T>,
        ) -> Result<Tensor<i32>, OpError> {
            boolean_op(pool, a, b, BooleanOp::$name)
        }

        #[derive(Debug)]
        pub struct $name {}

        impl Operator for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn is_commutative(&self) -> bool {
                // `Equal` is marked as commutative, but this will have no
                // effect until an in-place version of the operator is
                // implemented for bool inputs.
                stringify!($name) == "Equal"
            }

            fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
                run_typed_op!(pool, inputs, $func)
            }
        }
    };
}

boolean_cmp_op!(Equal, equal);
boolean_cmp_op!(Greater, greater);
boolean_cmp_op!(GreaterOrEqual, greater_or_equal);
boolean_cmp_op!(Less, less);
boolean_cmp_op!(LessOrEqual, less_or_equal);

/// Calculate the remainder of `x / y` using floored division. See
/// [`DivMode`] for an explanation.
fn rem_floor<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Rem<Output = T>,
>(
    x: T,
    y: T,
) -> T {
    // See https://en.wikipedia.org/wiki/Modulo#Implementing_other_modulo_definitions_using_truncation
    let zero = T::default();
    let mut rem = x % y;
    if rem > zero && y < zero || rem < zero && y > zero {
        rem = rem + y;
    }
    rem
}

/// Division method to use. When both operators to a division or modulus
/// operator are positive, the different methods produce the same results.
///
/// When one or both of the operands is negative however, the different methods
/// produce different results.
///
/// See <https://en.wikipedia.org/wiki/Modulo#Variants_of_the_definition>.
pub enum DivMode {
    /// Use flooring division, like Python's `%` operator and `numpy.mod`.
    FloorDiv,

    /// Use truncated division, like C and Rust's `%` operator and `numpy.fmod`.
    TruncDiv,
}

/// Return the elementwise remainder of dividing `a / b`.
pub fn mod_op<
    T: Copy + Debug + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Rem<Output = T>,
>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
    mode: DivMode,
) -> Result<Tensor<T>, OpError> {
    binary_op(
        pool,
        a,
        b,
        match mode {
            DivMode::FloorDiv => |x, y| rem_floor(x, y),
            DivMode::TruncDiv => |x, y| x % y,
        },
    )
}

#[derive(Debug)]
pub struct Mod {
    /// If true, use truncated division (see [`DivMode::TruncDiv`], otherwise
    /// use flooring division (see [`DivMode::FloorDiv`]).
    pub fmod: bool,
}

impl Operator for Mod {
    fn name(&self) -> &str {
        "Mod"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let a = inputs.require(0)?;
        let mode = if self.fmod {
            DivMode::TruncDiv
        } else {
            DivMode::FloorDiv
        };

        map_input!(a, a, [FloatTensor, Int32Tensor], {
            let b = inputs.require_as(1)?;
            mod_op(pool, a, b, mode).into_op_result()
        })
    }
}

/// Multiply two tensors elementwise.
pub fn mul<T: Copy + Debug + Default + std::ops::Mul<Output = T>>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_commutative_op(pool, a, b, |x, y| x * y)
}

/// Perform in-place elementwise multiplication of two tensors.
pub fn mul_in_place<T: Copy + Debug + std::ops::Mul<Output = T>>(
    a: TensorViewMut<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |a_elt, b_elt| a_elt * b_elt);
}

#[derive(Debug)]
pub struct Mul {}

impl Operator for Mul {
    fn name(&self) -> &str {
        "Mul"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        run_typed_op!(pool, inputs, mul)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        run_typed_op_in_place!(pool, input, other, mul_in_place, mul)
    }
}

/// Like [`f32::powf`] but with fast paths for common values.
fn powf(x: f32, y: f32) -> f32 {
    if y == 2. {
        x * x
    } else if y == 3. {
        x * x * x
    } else {
        x.powf(y)
    }
}

/// Raise elements of `a` to powers of corresponding elements in `b`.
pub fn pow(pool: &TensorPool, a: TensorView, b: TensorView) -> Result<Tensor, OpError> {
    if let Some(&exp) = b.item() {
        Ok(a.map_in(pool, |x| powf(*x, exp)))
    } else {
        binary_op(pool, a, b, powf)
    }
}

/// Perform in-place raise of elements of `a` to power of corresponding elements in `b`.
pub fn pow_in_place(mut a: TensorViewMut, b: TensorView) {
    if let Some(exp) = b.item() {
        a.apply(|x| powf(*x, *exp))
    } else {
        binary_op_in_place(a, b, powf);
    }
}

#[derive(Debug)]
pub struct Pow {}

impl Operator for Pow {
    fn name(&self) -> &str {
        "Pow"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        pow(pool, a, b).into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        let mut a = input
            .into_tensor::<f32>()
            .ok_or(OpError::IncorrectInputType)?;
        let b = other.require_as(0)?;

        if can_run_binary_op_in_place(&a, &b) {
            pow_in_place(a.view_mut(), b);
            Ok(a.into())
        } else {
            pow(pool, a.view(), b).map(|t| t.into())
        }
    }
}

/// Perform elementwise subtraction of two tensors.
pub fn sub<T: Copy + Debug + Default + std::ops::Sub<Output = T>>(
    pool: &TensorPool,
    a: TensorView<T>,
    b: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    binary_op(pool, a, b, |x, y| x - y)
}

/// Perform in-place elementwise subtraction of two tensors.
pub fn sub_in_place<T: Copy + Debug + std::ops::Sub<Output = T>>(
    a: TensorViewMut<T>,
    b: TensorView<T>,
) {
    binary_op_in_place(a, b, |x, y| x - y);
}

#[derive(Debug)]
pub struct Sub {}

impl Operator for Sub {
    fn name(&self) -> &str {
        "Sub"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        run_typed_op!(pool, inputs, sub)
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        run_typed_op_in_place!(pool, input, other, sub_in_place, sub)
    }
}

pub fn where_op<T: Copy>(
    pool: &TensorPool,
    cond: TensorView<i32>,
    x: TensorView<T>,
    y: TensorView<T>,
) -> Result<Tensor<T>, OpError> {
    let broadcast_xy_shape = broadcast_shapes(x.shape(), y.shape())
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;
    let result_shape = broadcast_shapes(cond.shape(), &broadcast_xy_shape)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))?;

    let out_len = result_shape.iter().product();
    let mut out_data = pool.alloc(out_len);

    let mut cond = cond.broadcast(result_shape.as_slice());
    let mut x = x.broadcast(result_shape.as_slice());
    let mut y = y.broadcast(result_shape.as_slice());

    // Loop over a statically known number of inner dims for efficiency.
    while cond.ndim() <= 4 {
        cond.insert_axis(0);
        x.insert_axis(0);
        y.insert_axis(0);
    }

    let out_uninit = &mut out_data.spare_capacity_mut()[..cond.len()];
    let mut out_offset = 0;

    cond.inner_iter::<4>()
        .zip(x.inner_iter::<4>().zip(y.inner_iter::<4>()))
        .for_each(|(cond, (x, y))| {
            for i0 in 0..cond.size(0) {
                for i1 in 0..cond.size(1) {
                    for i2 in 0..cond.size(2) {
                        for i3 in 0..cond.size(3) {
                            // Safety:
                            // - `cond`, `x` and `y` have the same shape, and i0..i3 are in `[0, cond.size(i))`.
                            // - The length of `out_uninit` is the same as `cond.len()`.
                            unsafe {
                                let cond_elt = *cond.get_unchecked([i0, i1, i2, i3]);
                                let x_elt = *x.get_unchecked([i0, i1, i2, i3]);
                                let y_elt = *y.get_unchecked([i0, i1, i2, i3]);
                                let out_elt = if cond_elt != 0 { x_elt } else { y_elt };
                                out_uninit.get_unchecked_mut(out_offset).write(out_elt);
                                out_offset += 1;
                            }
                        }
                    }
                }
            }
        });

    // Safety: We just initialized `cond.len` elements.
    assert!(out_offset == cond.len());
    unsafe {
        out_data.set_len(cond.len());
    }

    Ok(Tensor::from_data(&result_shape, out_data))
}

#[derive(Debug)]
pub struct Where {}

impl Operator for Where {
    fn name(&self) -> &str {
        "Where"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let condition = inputs.require_as::<i32>(0)?;
        let x = inputs.require(1)?;

        map_input!(x, x, [FloatTensor, Int32Tensor], {
            let y = inputs.require_as(2)?;
            where_op(pool, condition, x, y).into_op_result()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use super::fast_broadcast_cycles_repeats;
    use crate::ops::tests::new_pool;
    use crate::ops::{
        add, add_in_place, and, div, div_in_place, equal, greater, greater_or_equal, less,
        less_or_equal, mod_op, mul, mul_in_place, or, pow, pow_in_place, sub, sub_in_place,
        where_op, xor, Add, DivMode, OpError, Operator, Output,
    };

    #[test]
    fn test_fast_broadcast_cycles_repeats() {
        // Scalar
        let params = fast_broadcast_cycles_repeats(&[], &[1, 2, 3]);
        assert_eq!(params, Some((1, 6)));

        // All dims broadcast
        let params = fast_broadcast_cycles_repeats(&[1, 1, 1], &[5, 6, 2]);
        assert_eq!(params, Some((1, 60)));

        // Same from/to shapes.
        let params = fast_broadcast_cycles_repeats(&[3, 4, 5], &[3, 4, 5]);
        assert_eq!(params, Some((1, 1)));

        // Cycle only
        let params = fast_broadcast_cycles_repeats(&[1, 1, 10], &[5, 2, 10]);
        assert_eq!(params, Some((10, 1)));

        // Repeat only
        let params = fast_broadcast_cycles_repeats(&[10, 1, 1], &[10, 5, 6]);
        assert_eq!(params, Some((1, 30)));

        // Cycle + repeat
        let params = fast_broadcast_cycles_repeats(&[1, 10, 1], &[5, 10, 6]);
        assert_eq!(params, Some((5, 6)));

        // Non-fast broadcast
        let params = fast_broadcast_cycles_repeats(&[5, 1, 5], &[5, 6, 5]);
        assert_eq!(params, None);

        let params = fast_broadcast_cycles_repeats(&[1, 5, 1, 5, 1], &[2, 5, 6, 5, 2]);
        assert_eq!(params, None);

        // Implicit padding
        let params = fast_broadcast_cycles_repeats(&[10], &[5, 3, 10]);
        assert_eq!(params, Some((15, 1)));
    }

    #[test]
    #[should_panic]
    fn test_fast_broadcast_cycles_repeats_invalid() {
        fast_broadcast_cycles_repeats(&[1, 2, 3], &[1, 2]);
    }

    #[test]
    fn test_add() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 22., 33., 44.]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![11, 22, 33, 44]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_add_broadcasted() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Simple case where comparing ordering of tensor shapes tells us
        // target shape.
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from([10.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 12., 13., 14.]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Try alternative ordering for inputs.
        let result = add(&pool, b.view(), a.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Case where the length of tensor shapes needs to be compared before
        // the ordering, since ([5] > [1,5]).
        let a = Tensor::from([1., 2., 3., 4., 5.]);
        let b = Tensor::from_data(&[1, 5], vec![1., 2., 3., 4., 5.]);
        let expected = Tensor::from_data(&[1, 5], vec![2., 4., 6., 8., 10.]);

        let result = add(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Case where one of the inputs is a scalar.
        let a = Tensor::from(3.0);
        let b = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        let expected = Tensor::from_data(&[2, 2], vec![4.0, 5.0, 6.0, 7.0]);
        expect_equal(&result, &expected)?;

        // Case where broadcast shape uses dimensions from both inputs.
        let a = Tensor::from_data(&[2, 1], vec![1, 2]);
        let b = Tensor::from_data(&[1, 2], vec![3, 4]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), &[4, 5, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_add_broadcast_first_input() {
        let pool = new_pool();
        let a: Tensor<i32> = Tensor::zeros(&[1, 1, 10]);
        let b = Tensor::zeros(&[1, 5, 10]);
        let result = add(&pool, a.view(), b.view()).unwrap();
        assert_eq!(result.shape(), &[1, 5, 10]);
    }

    #[test]
    fn test_add_in_place() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // In-place addition with float inputs that have the same shape.
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let a_copy = a.clone();
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![11., 22., 33., 44.]);
        add_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // In-place addition with int inputs that have the same shape.
        let mut a_ints = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b_ints = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected_ints = Tensor::from_data(&[2, 2], vec![11, 22, 33, 44]);
        add_in_place(a_ints.view_mut(), b_ints.view());
        assert_eq!(&a_ints, &expected_ints);

        // Run `Add` operator in place with inputs that support in-place addition.
        let op = Add {};
        let result = op
            .run_in_place(&pool, Output::FloatTensor(a_copy), (&b).into())
            .unwrap();
        expect_equal(&result.as_tensor_view().unwrap(), &expected.view())?;

        // Run `Add` operator in-place with inputs that don't support in-place
        // addition. The operator should fall back to creating a new output tensor.
        let scalar = Tensor::from(1.0);
        let expected = Tensor::from_data(&[2, 2], vec![11., 21., 31., 41.]);
        let result = op
            .run_in_place(&pool, Output::FloatTensor(scalar), (&b).into())
            .unwrap();
        expect_equal(&result.as_tensor_view().unwrap(), &expected.view())?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first.
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from([1., 2.]);
        let expected = Tensor::from_data(&[2, 2], vec![2., 4., 4., 6.]);

        add_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // In-place addition where the second input must be broadcast to the
        // shape of the first, and the first has a non-contiguous layout.
        let mut a = Tensor::from_data(&[2, 3], vec![1., 2., 0., 3., 4., 0.]);
        a.clip_dim(1, 0..2);
        assert!(!a.is_contiguous());
        let b = Tensor::from([1., 2.]);
        let expected = Tensor::from_data(&[2, 2], vec![2., 4., 4., 6.]);

        add_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        Ok(())
    }

    #[test]
    fn test_add_invalid_broadcast() {
        let pool = new_pool();
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);

        let op = Add {};
        let result = op.run(&pool, (&a, &b).into());

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }

    #[test]
    fn test_and() {
        let pool = new_pool();
        let a = Tensor::from([0, 1, 0, 1]);
        let b = Tensor::from([0, 0, 1, 1]);
        let expected = Tensor::from([0, 0, 0, 1]);
        let result = and(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_div() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Non-scalar a and b
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 10., 10., 10.]);
        let result = div(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Scalar b
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from(10.);
        let expected = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let result = div(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Non-scalar a and b ints
        let a = Tensor::from([1, 2, 3, 4]);
        let b = Tensor::from([2, 2, 2, 2]);
        let expected = Tensor::from([0, 1, 1, 2]);
        let result = div(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Scalar b int
        let a = Tensor::from([1, 2, 3, 4]);
        let b = Tensor::from(2);
        let expected = Tensor::from([0, 1, 1, 2]);
        let result = div(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_div_in_place() -> Result<(), Box<dyn Error>> {
        // Non-scalar a and b
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 10., 10., 10.]);
        div_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // Scalar b
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from(10.);
        let expected = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        div_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // Non-scalar a and b ints
        let mut a = Tensor::from([1, 2, 3, 4]);
        let b = Tensor::from([2, 2, 2, 2]);
        let expected = Tensor::from([0, 1, 1, 2]);
        div_in_place(a.view_mut(), b.view());
        assert_eq!(&a, &expected);

        // Scalar b int
        let mut a = Tensor::from([1, 2, 3, 4]);
        let b = Tensor::from(2);
        let expected = Tensor::from([0, 1, 1, 2]);
        div_in_place(a.view_mut(), b.view());
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_equal() {
        let pool = new_pool();

        // Int tensor
        let a = Tensor::from([1, 2]);
        let b = Tensor::from([1, 3]);
        let expected = Tensor::from([1, 0]);
        let result = equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = Tensor::from([1., 2.]);
        let b = Tensor::from([1., 3.]);
        let expected = Tensor::from([1, 0]);
        let result = equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_greater() {
        let pool = new_pool();

        // Int tensor
        let a = Tensor::from([1, 2, 5]);
        let b = Tensor::from([1, 3, 4]);
        let expected = Tensor::from([0, 0, 1]);
        let result = greater(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = Tensor::from([1., 2., 5.]);
        let b = Tensor::from([1., 3., 4.]);
        let expected = Tensor::from([0, 0, 1]);
        let result = greater(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_greater_or_equal() {
        let pool = new_pool();

        // Int tensor
        let a = Tensor::from([1, 2, 5]);
        let b = Tensor::from([1, 3, 4]);
        let expected = Tensor::from([1, 0, 1]);
        let result = greater_or_equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = Tensor::from([1., 2., 5.]);
        let b = Tensor::from([1., 3., 4.]);
        let expected = Tensor::from([1, 0, 1]);
        let result = greater_or_equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_less() {
        let pool = new_pool();

        // Int tensor
        let a = Tensor::from([1, 2]);
        let b = Tensor::from([1, 3]);
        let expected = Tensor::from([0, 1]);
        let result = less(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = Tensor::from([1., 2.]);
        let b = Tensor::from([1., 3.]);
        let expected = Tensor::from([0, 1]);
        let result = less(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_less_or_equal() {
        let pool = new_pool();

        // Int tensor
        let a = Tensor::from([1, 2, 5]);
        let b = Tensor::from([1, 3, 4]);
        let expected = Tensor::from([1, 1, 0]);
        let result = less_or_equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor
        let a = Tensor::from([1., 2., 5.]);
        let b = Tensor::from([1., 3., 4.]);
        let expected = Tensor::from([1, 1, 0]);
        let result = less_or_equal(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_mod_op() {
        let pool = new_pool();

        // Int tensor, floor division (like Python's `%`, `numpy.mod`).
        let a = Tensor::from([10, -10, 10, -10]);
        let b = Tensor::from([3, 3, -3, -3]);
        let expected = Tensor::from([1, 2, -2, -1]);
        let result = mod_op(&pool, a.view(), b.view(), DivMode::FloorDiv).unwrap();
        assert_eq!(&result, &expected);

        // Int tensor, truncated division (like Rust's `%`, `numpy.fmod`).
        let expected = Tensor::from([1, -1, 1, -1]);
        let result = mod_op(&pool, a.view(), b.view(), DivMode::TruncDiv).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor, floor division.
        let af = Tensor::from([3.5, -3.5, 3.5, -3.5]);
        let bf = Tensor::from([2.5, 2.5, -2.5, -2.5]);
        let expected = Tensor::from([1., 1.5, -1.5, -1.]);
        let result = mod_op(&pool, af.view(), bf.view(), DivMode::FloorDiv).unwrap();
        assert_eq!(&result, &expected);

        // Float tensor, truncated division.
        let expected = Tensor::from([1., -1., 1., -1.]);
        let result = mod_op(&pool, af.view(), bf.view(), DivMode::TruncDiv).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_mul() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 40., 90., 160.]);
        let result = mul(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![10, 40, 90, 160]);
        let result = mul(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_mul_in_place() -> Result<(), Box<dyn Error>> {
        // Float tensor
        let mut a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let expected = Tensor::from_data(&[2, 2], vec![10., 40., 90., 160.]);
        mul_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let expected = Tensor::from_data(&[2, 2], vec![10, 40, 90, 160]);
        mul_in_place(a.view_mut(), b.view());
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_or() {
        let pool = new_pool();
        let a = Tensor::from([0, 1, 0, 1]);
        let b = Tensor::from([0, 0, 1, 1]);
        let expected = Tensor::from([0, 1, 1, 1]);
        let result = or(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_pow() -> Result<(), Box<dyn Error>> {
        struct Case {
            a: Tensor<f32>,
            b: Tensor<f32>,
            expected: Tensor<f32>,
        }

        let cases = [
            // Square input
            Case {
                a: [2., 3., 4.].into(),
                b: (2.).into(),
                expected: [4., 9., 16.].into(),
            },
            // Cube input
            Case {
                a: [2., 3., 4.].into(),
                b: (3.).into(),
                expected: [8., 27., 64.].into(),
            },
            // Raise all inputs to scalar
            Case {
                a: [2., 3., 4.].into(),
                b: (0.256).into(),
                expected: [(2f32).powf(0.256), (3f32).powf(0.256), (4f32).powf(0.256)].into(),
            },
            // Raise each input to different powers
            Case {
                a: [2., 3., 4.].into(),
                b: [1., 2., 3.].into(),
                expected: [2., 9., 64.].into(),
            },
        ];

        let pool = new_pool();

        for case in cases {
            // Copying variant
            let result = pow(&pool, case.a.view(), case.b.view()).unwrap();
            expect_equal(&result, &case.expected)?;

            // In-place variant
            let mut a = case.a.clone();
            pow_in_place(a.view_mut(), case.b.view());
            expect_equal(&a, &case.expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Float tensor
        let a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![9., 18., 27., 36.]);
        let result = sub(&pool, a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Int tensor
        let a = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let b = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let expected = Tensor::from_data(&[2, 2], vec![9, 18, 27, 36]);
        let result = sub(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    // Test for a non-commutative binary operator that involves broadcasting
    // both the inner and outer dimensions of the second operand.
    #[test]
    fn test_sub_broadcast() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // [2, 3, 3]
        let a = Tensor::from([
            [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
            [[3, 6, 9], [12, 15, 18], [21, 24, 27]],
        ]);

        // [1, 3, 1]
        let b = Tensor::from([[[1], [2], [3]]]);

        let expected = Tensor::from([
            [[1, 3, 5], [6, 8, 10], [11, 13, 15]],
            [[2, 5, 8], [10, 13, 16], [18, 21, 24]],
        ]);

        let result = sub(&pool, a.view(), b.view()).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_sub_in_place() -> Result<(), Box<dyn Error>> {
        // Float tensor
        let mut a = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let b = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let expected = Tensor::from_data(&[2, 2], vec![9., 18., 27., 36.]);
        sub_in_place(a.view_mut(), b.view());
        expect_equal(&a, &expected)?;

        // Int tensor
        let mut a = Tensor::from_data(&[2, 2], vec![10, 20, 30, 40]);
        let b = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
        let expected = Tensor::from_data(&[2, 2], vec![9, 18, 27, 36]);
        sub_in_place(a.view_mut(), b.view());
        assert_eq!(&a, &expected);

        Ok(())
    }

    #[test]
    fn test_where() {
        let pool = new_pool();

        // Float tensor with exact matching shapes
        let cond = Tensor::from_data(&[2, 2], vec![1, 0, 0, 1]);
        let x = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let y = Tensor::from_data(&[2, 2], vec![10., 20., 30., 40.]);
        let result = where_op(&pool, cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from_data(&[2, 2], vec![1., 20., 30., 4.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `x` and `y`
        let cond = Tensor::from([1, 1, 0, 0]);
        let x = Tensor::from(1.);
        let y = Tensor::from(2.);
        let result = where_op(&pool, cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from([1., 1., 2., 2.]);
        assert_eq!(&result, &expected);

        // Float tensor broadcasting `cond`
        let cond = Tensor::from(1);
        let x = Tensor::from([1., 2.]);
        let y = Tensor::from([3., 4.]);
        let result = where_op(&pool, cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from([1., 2.]);
        assert_eq!(&result, &expected);

        // Int tensor broadcasting `x` and `y`
        let cond = Tensor::from([1, 1, 0, 0]);
        let x = Tensor::from(3);
        let y = Tensor::from(4);
        let result = where_op(&pool, cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from([3, 3, 4, 4]);
        assert_eq!(&result, &expected);

        // Int tensor broadcasting `x` and `y`, and broadcasting involves
        // repeating the last dimension, not just cycling. This exercises a
        // fallback path.
        let cond = Tensor::from([[1, 0], [1, 0]]);
        let x = Tensor::from([[1], [2]]);
        let y = Tensor::from([[3], [4]]);
        let result = where_op(&pool, cond.view(), x.view(), y.view()).unwrap();
        let expected = Tensor::from([[1, 3], [2, 4]]);
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_where_invalid_inputs() {
        let pool = new_pool();

        let cond = Tensor::from([1, 1]);
        let x = Tensor::from([1, 2, 3]);
        let y = Tensor::from([2, 2]);

        // Failure to broadcast `x` to match `cond`
        let result = where_op(&pool, cond.view(), x.view(), y.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );

        // Failure to broadcast `y` to match `cond`
        let result = where_op(&pool, cond.view(), y.view(), x.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes("Cannot broadcast inputs"))
        );
    }

    #[test]
    fn test_xor() {
        let pool = new_pool();
        let a = Tensor::from([0, 1, 0, 1]);
        let b = Tensor::from([0, 0, 1, 1]);
        let expected = Tensor::from([0, 1, 1, 0]);
        let result = xor(&pool, a.view(), b.view()).unwrap();
        assert_eq!(&result, &expected);
    }
}

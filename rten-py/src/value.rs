//! Conversions between numpy arrays and RTen values.

use numpy::ndarray::ArrayD;
use numpy::{
    Element, IxDyn, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use rten::{Value, ValueOrView};
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::RtenError;

/// An input tensor for a model run.
///
/// This is borrowed if possible or owned if the value had to be converted
/// to a supported type.
pub enum InputTensor<'py> {
    Float(PyReadonlyArrayDyn<'py, f32>),
    Int32(PyReadonlyArrayDyn<'py, i32>),
    Int8(PyReadonlyArrayDyn<'py, i8>),
    UInt8(PyReadonlyArrayDyn<'py, u8>),
    Converted(Value),
}

impl InputTensor<'_> {
    pub fn as_value_or_view(&self) -> ValueOrView<'_> {
        fn view<'a, T: Element>(array: &'a PyReadonlyArrayDyn<'_, T>) -> TensorView<'a, T> {
            TensorView::from_data(array.shape(), slice_from_array(array))
        }

        match self {
            Self::Float(array) => view(array).into(),
            Self::Int32(array) => view(array).into(),
            Self::Int8(array) => view(array).into(),
            Self::UInt8(array) => view(array).into(),
            Self::Converted(value) => value.into(),
        }
    }
}

/// Convert a numpy array into an input tensor for the model input `name`.
pub fn input_tensor<'py>(name: &str, value: &Bound<'py, PyAny>) -> PyResult<InputTensor<'py>> {
    let Ok(array) = value.cast::<PyUntypedArray>() else {
        return Err(PyTypeError::new_err(format!(
            "input \"{}\" must be a numpy array, not {}",
            name,
            value.get_type().name()?
        )));
    };

    // Copy arrays that are not aligned and C-contiguous, so that all arrays
    // below can be read as slices.
    let value = if array.is_c_contiguous() && array.is_aligned() {
        value.clone()
    } else {
        value.call_method1("copy", ("C",))?
    };

    // Borrow arrays with element types which RTen supports directly.
    macro_rules! borrow {
        ($elem:ty, $variant:ident) => {
            if let Ok(array) = value.cast::<PyArrayDyn<$elem>>() {
                return Ok(InputTensor::$variant(array.readonly()));
            }
        };
    }
    borrow!(f32, Float);
    borrow!(i32, Int32);
    borrow!(i8, Int8);
    borrow!(u8, UInt8);

    // Convert element types which RTen doesn't support directly to a supported
    // type.
    if let Ok(array) = value.cast::<PyArrayDyn<i64>>() {
        return convert_array(array, |x| {
            i32::try_from(x).map_err(|_| {
                PyOverflowError::new_err(format!(
                    "input \"{}\" contains the value {} which does not fit in an int32. RTen converts int64 inputs to int32.",
                    name, x
                ))
            })
        });
    }
    if let Ok(array) = value.cast::<PyArrayDyn<bool>>() {
        return convert_array(array, |x| Ok(x as i32));
    }
    if let Ok(array) = value.cast::<PyArrayDyn<f64>>() {
        return convert_array(array, |x| Ok(x as f32));
    }

    Err(PyTypeError::new_err(format!(
        "input \"{}\" has unsupported dtype {}. Supported dtypes are float32, int32, int8, uint8, int64, bool and float64.",
        name,
        array.dtype()
    )))
}

/// Return the elements of an array as a slice.
///
/// Panics if the array is not aligned and C-contiguous.
fn slice_from_array<'a, T: Element>(array: &'a PyReadonlyArrayDyn<'_, T>) -> &'a [T] {
    array
        .as_slice()
        .expect("array should be aligned and contiguous")
}

/// Convert the elements of an array to a supported type using `f`.
fn convert_array<'py, T: Element + Copy, U>(
    array: &Bound<'py, PyArrayDyn<T>>,
    f: impl FnMut(T) -> PyResult<U>,
) -> PyResult<InputTensor<'py>>
where
    Value: From<Tensor<U>>,
{
    let array = array.readonly();
    let data = slice_from_array(&array)
        .iter()
        .copied()
        .map(f)
        .collect::<PyResult<Vec<U>>>()?;
    let value = Value::from_shape(array.shape(), data).expect("data length matches shape");
    Ok(InputTensor::Converted(value))
}

/// Convert a model output into a numpy array.
pub fn value_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    fn array<T: Element + Clone>(py: Python<'_>, tensor: Tensor<T>) -> Py<PyAny> {
        let shape = IxDyn(tensor.shape());
        let array =
            ArrayD::from_shape_vec(shape, tensor.into_data()).expect("data length matches shape");
        PyArray::from_owned_array(py, array).into_any().unbind()
    }

    match value {
        Value::FloatTensor(tensor) => Ok(array(py, tensor)),
        Value::Int32Tensor(tensor) => Ok(array(py, tensor)),
        Value::Int8Tensor(tensor) => Ok(array(py, tensor)),
        Value::UInt8Tensor(tensor) => Ok(array(py, tensor)),
        value => Err(RtenError::new_err(format!(
            "outputs of type {} are not supported",
            value.dtype()
        ))),
    }
}

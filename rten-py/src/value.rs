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
            // The borrowed variants are only created for C-contiguous arrays,
            // so `as_slice` always succeeds and the element count always
            // matches the shape.
            TensorView::from_data(
                array.shape(),
                array.as_slice().expect("array should be contiguous"),
            )
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

    // Convert element types which RTen supports directly. This avoids a copy
    // if the array is contiguous.
    macro_rules! borrow_or_copy {
        ($elem:ty, $variant:ident) => {
            if let Ok(array) = value.cast::<PyArrayDyn<$elem>>() {
                let array = array.readonly();
                if array.is_c_contiguous() {
                    return Ok(InputTensor::$variant(array));
                }
                let data: Vec<$elem> = array.as_array().iter().copied().collect();
                return owned(name, array.shape(), data);
            }
        };
    }
    borrow_or_copy!(f32, Float);
    borrow_or_copy!(i32, Int32);
    borrow_or_copy!(i8, Int8);
    borrow_or_copy!(u8, UInt8);

    // Convert element types which RTen doesn't support directly to a supported
    // type.
    if let Ok(array) = value.cast::<PyArrayDyn<i64>>() {
        let array = array.readonly();
        let data = array
            .as_array()
            .iter()
            .map(|&x| {
                i32::try_from(x).map_err(|_| {
                    PyOverflowError::new_err(format!(
                        "input \"{}\" contains the value {} which does not fit in an int32. RTen converts int64 inputs to int32.",
                        name, x
                    ))
                })
            })
            .collect::<PyResult<Vec<i32>>>()?;
        return owned(name, array.shape(), data);
    }
    if let Ok(array) = value.cast::<PyArrayDyn<bool>>() {
        let array = array.readonly();
        let data: Vec<i32> = array.as_array().iter().map(|&x| x as i32).collect();
        return owned(name, array.shape(), data);
    }
    if let Ok(array) = value.cast::<PyArrayDyn<f64>>() {
        let array = array.readonly();
        let data: Vec<f32> = array.as_array().iter().map(|&x| x as f32).collect();
        return owned(name, array.shape(), data);
    }

    Err(PyTypeError::new_err(format!(
        "input \"{}\" has unsupported dtype {}. Supported dtypes are float32, int32, int8, uint8, int64, bool and float64.",
        name,
        array.dtype()
    )))
}

fn owned<'py, T>(name: &str, shape: &[usize], data: Vec<T>) -> PyResult<InputTensor<'py>>
where
    Value: From<Tensor<T>>,
{
    let value = Value::from_shape(shape, data)
        .map_err(|err| PyTypeError::new_err(format!("invalid input \"{}\": {}", name, err)))?;
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

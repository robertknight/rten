use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

use rten_base::byte_cast;
use rten_tensor::{AsView, Layout, Tensor, TensorView};
use safetensors::tensor::TensorView as SafeTensorView;
use safetensors::{Dtype, SafeTensorError, SafeTensors, serialize};

use crate::value::{DataType, Value, View, dispatch_data_type, match_view};

/// A Rust element type that can be (de)serialized in the safetensors format.
trait SafeElement: Sized {
    /// The corresponding safetensors data type.
    const DTYPE: Dtype;

    /// Decode a little-endian byte buffer into a vector of elements.
    fn from_le_bytes(bytes: &[u8]) -> Vec<Self>;

    /// Encode a tensor's elements as a little-endian byte buffer.
    ///
    /// This borrows the data if the view is contiguous or copies it otherwise.
    fn to_le_bytes(view: TensorView<'_, Self>) -> Cow<'_, [u8]>;
}

macro_rules! impl_safe_element {
    ($ty:ty, $dtype:expr) => {
        impl SafeElement for $ty {
            const DTYPE: Dtype = $dtype;

            fn from_le_bytes(bytes: &[u8]) -> Vec<Self> {
                bytes
                    .chunks_exact(size_of::<$ty>())
                    .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            }

            fn to_le_bytes(view: TensorView<'_, Self>) -> Cow<'_, [u8]> {
                match view.data() {
                    Some(data) if cfg!(target_endian = "little") => {
                        Cow::Borrowed(byte_cast::cast_slice(data).unwrap())
                    }
                    _ => {
                        let buf = view.iter().flat_map(|x| x.to_le_bytes()).collect();
                        Cow::Owned(buf)
                    }
                }
            }
        }
    };
}

impl_safe_element!(i8, Dtype::I8);
impl_safe_element!(i16, Dtype::I16);
impl_safe_element!(i32, Dtype::I32);
impl_safe_element!(i64, Dtype::I64);
impl_safe_element!(u8, Dtype::U8);
impl_safe_element!(u16, Dtype::U16);
impl_safe_element!(u32, Dtype::U32);
impl_safe_element!(u64, Dtype::U64);
impl_safe_element!(f32, Dtype::F32);
impl_safe_element!(f64, Dtype::F64);

impl SafeElement for bool {
    const DTYPE: Dtype = Dtype::BOOL;

    fn from_le_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.iter().map(|&b| b != 0).collect()
    }

    fn to_le_bytes(view: TensorView<'_, Self>) -> Cow<'_, [u8]> {
        match view.data() {
            // nb. `cast_slice` to `[u8]` always succeeds.
            Some(data) => Cow::Borrowed(byte_cast::cast_slice(data).unwrap()),
            None => {
                let bool_vec = view.to_vec();
                Cow::Owned(byte_cast::cast_vec(bool_vec).unwrap())
            }
        }
    }
}

/// Map a safetensors data type to the corresponding [`DataType`], or `None` if
/// it has no corresponding Rust primitive type.
fn data_type_from_safetensors(dtype: Dtype) -> Option<DataType> {
    let data_type = match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::I8 => DataType::Int8,
        Dtype::I16 => DataType::Int16,
        Dtype::I32 => DataType::Int32,
        Dtype::I64 => DataType::Int64,
        Dtype::U8 => DataType::UInt8,
        Dtype::U16 => DataType::UInt16,
        Dtype::U32 => DataType::UInt32,
        Dtype::U64 => DataType::UInt64,
        Dtype::F32 => DataType::Float32,
        Dtype::F64 => DataType::Float64,
        _ => return None,
    };
    Some(data_type)
}

fn to_io_error(err: SafeTensorError) -> io::Error {
    // nb. `err` has an `IoError(io::Error)` variant which we could just return
    // directly, but this is not available because we compile safetensors
    // without the std feature to reduce dependencies.
    io::Error::new(io::ErrorKind::InvalidData, err)
}

fn value_from_view(view: &SafeTensorView) -> io::Result<Value> {
    let data_type = data_type_from_safetensors(view.dtype()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported safetensors dtype {:?}", view.dtype()),
        )
    })?;
    let shape = view.shape();
    let bytes = view.data();
    let value = dispatch_data_type!(data_type, T => {
        let data = <T as SafeElement>::from_le_bytes(bytes);
        Value::from(Tensor::<T>::from_data(shape, data))
    });
    Ok(value)
}

/// A maybe-owned tensor encoded as raw little-endian bytes.
///
/// This implements [`safetensors::View`].
struct TensorBytes<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Cow<'a, [u8]>,
}

impl<'a> TensorBytes<'a> {
    /// Convert an RTen tensor view into this type.
    ///
    /// This borrows the data if the view is contiguous or copies it otherwise.
    fn from_rten_tensor<T: SafeElement>(view: TensorView<'a, T>) -> Self {
        TensorBytes {
            dtype: T::DTYPE,
            shape: view.shape().to_vec(),
            data: T::to_le_bytes(view),
        }
    }
}

impl safetensors::View for TensorBytes<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data.as_ref())
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Convert an iterator of named RTen tensor views into an iterator of named
/// views that can be passed to [`safetensors::serialize`].
fn tensor_bytes_iter<'a, I, N, V>(tensors: I) -> impl Iterator<Item = (N, TensorBytes<'a>)>
where
    I: IntoIterator<Item = (N, V)>,
    N: AsRef<str>,
    V: Into<View<'a>>,
{
    tensors.into_iter().map(|(name, tensor)| {
        let bytes = match_view!(tensor.into(), tensor => TensorBytes::from_rten_tensor(tensor));
        (name, bytes)
    })
}

/// Serialize named tensors to a safetensors writer.
pub fn write<'a, I, N, V>(mut writer: impl io::Write, tensors: I) -> io::Result<()>
where
    I: IntoIterator<Item = (N, V)>,
    N: AsRef<str> + Ord + std::fmt::Display,
    V: Into<View<'a>>,
{
    let bytes = serialize(tensor_bytes_iter(tensors), None).map_err(to_io_error)?;
    writer.write_all(&bytes)
}

/// Serialize named tensors to a safetensors file - this will create the file.
pub fn write_to_file<'a, I, N, V>(path: impl AsRef<Path>, tensors: I) -> io::Result<()>
where
    I: IntoIterator<Item = (N, V)>,
    N: AsRef<str> + Ord + std::fmt::Display,
    V: Into<View<'a>>,
{
    let file = io::BufWriter::new(File::create(path)?);
    write(file, tensors)
}

/// Read all tensors from a Safetensors buffer.
fn read_map(data: &[u8]) -> io::Result<HashMap<String, Value>> {
    let tensors = SafeTensors::deserialize(data).map_err(to_io_error)?;
    tensors
        .iter()
        .map(|(name, view)| Ok((name.to_string(), value_from_view(&view)?)))
        .collect()
}

/// Read all tensors from a Safetensors archive.
pub fn read(mut reader: impl io::Read) -> io::Result<HashMap<String, Value>> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    read_map(&data)
}

/// Read all tensors from a Safetensors file.
pub fn read_from_file(path: impl AsRef<Path>) -> io::Result<HashMap<String, Value>> {
    let data = std::fs::read(path)?;
    read_map(&data)
}

/// Read a single named tensor from a Safetensors buffer.
fn read_one(data: &[u8], name: &str) -> io::Result<Value> {
    let tensors = SafeTensors::deserialize(data).map_err(to_io_error)?;
    let view = tensors.tensor(name).map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing safetensors array `{name}`"),
        )
    })?;
    value_from_view(&view)
}

/// Read a single named tensor from a Safetensors archive.
pub fn read_array(mut reader: impl io::Read, name: &str) -> io::Result<Value> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    read_one(&data, name)
}

/// Read a single named tensor from a Safetensors file.
pub fn read_array_from_file(path: impl AsRef<Path>, name: &str) -> io::Result<Value> {
    let data = std::fs::read(path)?;
    read_one(&data, name)
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;

    use super::{
        View, read, read_array, read_array_from_file, read_from_file, write, write_to_file,
    };

    static NEXT_TEST_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rten-safetensors-{}-{id}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn test_round_trip_safetensors_in_memory() {
        let a: Tensor<f32> = [[1., 2., 3.], [4., 5., 6.]].into();
        let b: Tensor<i32> = [7, 8, 9].into();

        let mut buffer = Vec::new();
        // `View::from` on the first entry fixes the element type, so the rest
        // can use `.into()`.
        write(
            &mut buffer,
            [("a", View::from(a.view())), ("b", b.view().into())],
        )
        .unwrap();

        let arrays = read(&buffer[..]).unwrap();

        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays.get("a").unwrap().as_type::<f32>().unwrap(), a);
        assert_eq!(arrays.get("b").unwrap().as_type::<i32>().unwrap(), b);
    }

    #[test]
    fn test_round_trip_safetensors_all_dtypes() {
        let mut buffer = Vec::new();
        write(
            &mut buffer,
            [
                ("bool", View::from(Tensor::from([true, false, true]).view())),
                ("i8", Tensor::from([-1i8, 2, -3]).view().into()),
                ("i16", Tensor::from([-1i16, 2, -3]).view().into()),
                ("i32", Tensor::from([-1i32, 2, -3]).view().into()),
                ("i64", Tensor::from([-1i64, 2, -3]).view().into()),
                ("u8", Tensor::from([1u8, 2, 3]).view().into()),
                ("u16", Tensor::from([1u16, 2, 3]).view().into()),
                ("u32", Tensor::from([1u32, 2, 3]).view().into()),
                ("u64", Tensor::from([1u64, 2, 3]).view().into()),
                ("f32", Tensor::from([1.5f32, -2.5]).view().into()),
                ("f64", Tensor::from([1.5f64, -2.5]).view().into()),
            ],
        )
        .unwrap();

        let arrays = read(&buffer[..]).unwrap();

        assert_eq!(arrays.len(), 11);
        assert_eq!(
            arrays.get("bool").unwrap().as_type::<bool>().unwrap(),
            Tensor::from([true, false, true])
        );
        assert_eq!(
            arrays.get("i64").unwrap().as_type::<i64>().unwrap(),
            Tensor::from([-1i64, 2, -3])
        );
        assert_eq!(
            arrays.get("u32").unwrap().as_type::<u32>().unwrap(),
            Tensor::from([1u32, 2, 3])
        );
        assert_eq!(
            arrays.get("f64").unwrap().as_type::<f64>().unwrap(),
            Tensor::from([1.5f64, -2.5])
        );
    }

    #[test]
    fn test_round_trip_safetensors_to_file() {
        let path = temp_file("round-trip.safetensors");
        let a: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        write_to_file(&path, [("a", a.view())]).unwrap();
        let arrays = read_from_file(&path).unwrap();
        let a_read = read_array_from_file(&path, "a").unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(arrays.get("a").unwrap().as_type::<f32>().unwrap(), a);
        assert_eq!(a_read.into_type::<f32>().unwrap(), a);
    }

    #[test]
    fn test_read_safetensors_array_reports_missing_name() {
        let a: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Vec::new();
        write(&mut buffer, [("a", a.view())]).unwrap();

        let err = read_array(&buffer[..], "missing").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn test_read_safetensors_rejects_invalid_data() {
        let err = read(&b"not a safetensors file"[..]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}

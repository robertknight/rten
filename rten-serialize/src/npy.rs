use npyz::{AutoSerialize, Deserialize, NpyFile, Order, WriterBuilder};
use rten_tensor::{AsView, Layout, Tensor, TensorView};
use std::fs::File;
use std::io;
use std::path::Path;

/// Serialize the given `TensorView` to the writer.
pub fn write_npy<T: Clone + AutoSerialize>(
    writer: impl io::Write,
    array: TensorView<T>,
) -> io::Result<()> {
    let shape = array.shape().iter().map(|x| *x as u64).collect::<Vec<_>>();

    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(writer)
        .begin_nd()?;
    writer.extend(array.iter())?;
    writer.finish()
}

/// Serialize the given `TensorView` to a file - this will create the file.
pub fn write_npy_to_file<T: Clone + AutoSerialize>(
    path: impl AsRef<Path>,
    array: TensorView<T>,
) -> io::Result<()> {
    let file = io::BufWriter::new(File::create(path)?);
    write_npy(file, array)
}

pub fn read_npy<T: Clone + Deserialize>(reader: impl io::Read) -> io::Result<Tensor<T>> {
    let reader = NpyFile::new(reader)?;
    tensor_from_npy_file(reader)
}

pub(crate) fn tensor_from_npy_file<T: Clone + Deserialize, R: io::Read>(
    reader: NpyFile<R>,
) -> io::Result<Tensor<T>> {
    let shape = reader
        .shape()
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<_>>();
    let order = reader.order();
    let data = reader
        .data()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let values: io::Result<Vec<T>> = data.collect();
    let values = values?;
    let values = match order {
        Order::C => values,
        Order::Fortran => fortran_order_to_row_major(values, &shape),
    };

    let mut result = Tensor::from_vec(values);
    result.reshape(&shape);
    Ok(result)
}

fn fortran_order_to_row_major<T: Clone>(values: Vec<T>, shape: &[usize]) -> Vec<T> {
    if shape.len() < 2 {
        return values;
    }

    let reversed_shape = shape.iter().copied().rev().collect::<Vec<_>>();
    let reverse_axes = (0..shape.len()).rev().collect::<Vec<_>>();

    let mut tensor = Tensor::from_vec(values);
    tensor.reshape(&reversed_shape);
    tensor.permuted(&reverse_axes).to_vec()
}

pub fn read_npy_from_file<T: Clone + Deserialize>(path: impl AsRef<Path>) -> io::Result<Tensor<T>> {
    let file = io::BufReader::new(File::open(path)?);
    read_npy(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rten_tensor::AsView;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("rten-npy-{}-{id}-{name}", std::process::id()))
    }

    fn assert_tensor_eq<T: Clone + PartialEq + std::fmt::Debug>(
        actual: &Tensor<T>,
        expected: &Tensor<T>,
    ) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    /// Verifies that data written to an in-memory NPY stream can be read back
    /// with the same shape and values.
    #[test]
    fn round_trip_npy_in_memory() {
        let tensor: Tensor<i32> = [[1, 2, 3], [4, 5, 6]].into();
        let mut buffer = Vec::new();

        write_npy(&mut buffer, tensor.view()).unwrap();
        let read = read_npy::<i32>(&buffer[..]).unwrap();

        assert_tensor_eq(&read, &tensor);
    }

    /// Exercises the file convenience helpers by writing and then reading a
    /// tensor from a real file path.
    #[test]
    fn round_trip_npy_file_helpers() {
        let path = temp_file("round-trip.npy");
        let tensor: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        write_npy_to_file(&path, tensor.view()).unwrap();
        let read = read_npy_from_file::<f32>(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_tensor_eq(&read, &tensor);
    }

    /// Confirms malformed input is surfaced as an I/O error instead of being
    /// accepted as an empty or partially decoded tensor.
    #[test]
    fn read_npy_rejects_invalid_data() {
        let err = read_npy::<i32>(&b"not an npy file"[..]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// Covers the rank-2 boundary in Fortran-to-row-major conversion. The
    /// input is the Fortran flat representation of `[[1, 2, 3], [4, 5, 6]]`.
    #[test]
    fn fortran_order_to_row_major_converts_rank_2_data() {
        let converted = fortran_order_to_row_major(vec![1, 4, 2, 5, 3, 6], &[2, 3]);

        assert_eq!(converted, [1, 2, 3, 4, 5, 6]);
    }

    /// Checks that equivalent NumPy-generated C-order and Fortran-order files
    /// decode to the same row-major tensor.
    #[test]
    fn read_npy_c_and_fortran_order_fixtures_match() {
        let c_order = include_bytes!("../tests/fixtures/order_c_i32.npy");
        let fortran_order = include_bytes!("../tests/fixtures/order_fortran_i32.npy");

        let c_tensor = read_npy::<i32>(&c_order[..]).unwrap();
        let fortran_tensor = read_npy::<i32>(&fortran_order[..]).unwrap();
        let expected = Tensor::from([
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]);

        assert_tensor_eq(&c_tensor, &expected);
        assert_tensor_eq(&fortran_tensor, &expected);
        assert_tensor_eq(&fortran_tensor, &c_tensor);
    }
}

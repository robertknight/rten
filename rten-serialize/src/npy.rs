use npyz::{AutoSerialize, Deserialize, NpyFile, Order, WriterBuilder};
use rten_tensor::{AsView, Layout, Storage, Tensor, TensorBase};
use std::fs::File;
use std::io;
use std::path::Path;

/// Serialize a tensor to a writer.
pub fn write<T: AutoSerialize, S: Storage<Elem = T>, L: Layout + Clone>(
    writer: impl io::Write,
    array: &TensorBase<S, L>,
) -> io::Result<()> {
    let shape = array
        .shape()
        .as_ref()
        .iter()
        .map(|x| *x as u64)
        .collect::<Vec<_>>();

    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(writer)
        .begin_nd()?;
    writer.extend(array.iter())?;
    writer.finish()
}

/// Serialize a tensor to a file.
pub fn write_to_file<T: AutoSerialize, S: Storage<Elem = T>, L: Layout + Clone>(
    path: impl AsRef<Path>,
    array: &TensorBase<S, L>,
) -> io::Result<()> {
    let file = io::BufWriter::new(File::create(path)?);
    write(file, array)
}

/// Read a tensor from a reader.
pub fn read<T: Clone + Deserialize>(reader: impl io::Read) -> io::Result<Tensor<T>> {
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

    let result = Tensor::from_vec(values).into_shape(shape.as_slice());
    Ok(result)
}

fn fortran_order_to_row_major<T: Clone>(values: Vec<T>, shape: &[usize]) -> Vec<T> {
    if shape.len() < 2 {
        return values;
    }

    let reversed_shape = shape.iter().copied().rev().collect::<Vec<_>>();
    let reverse_axes = (0..shape.len()).rev().collect::<Vec<_>>();

    Tensor::from_vec(values)
        .reshaped(reversed_shape.as_slice())
        .permuted(&reverse_axes)
        .to_vec()
}

/// Read a tensor from a file.
pub fn read_from_file<T: Clone + Deserialize>(path: impl AsRef<Path>) -> io::Result<Tensor<T>> {
    let file = io::BufReader::new(File::open(path)?);
    read(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("rten-npy-{}-{id}-{name}", std::process::id()))
    }

    /// Verifies that data written to an in-memory NPY stream can be read back
    /// with the same shape and values.
    #[test]
    fn test_round_trip_npy_in_memory() {
        let tensor: Tensor<i32> = [[1, 2, 3], [4, 5, 6]].into();
        let mut buffer = Vec::new();

        write(&mut buffer, &tensor).unwrap();
        let read = read::<i32>(&buffer[..]).unwrap();

        assert_eq!(&read, &tensor);
    }

    /// Exercises the file convenience helpers by writing and then reading a
    /// tensor from a real file path.
    #[test]
    fn test_round_trip_npy_file_helpers() {
        let path = temp_file("round-trip.npy");
        let tensor: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        write_to_file(&path, &tensor).unwrap();
        let read = read_from_file::<f32>(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(&read, &tensor);
    }

    /// Confirms malformed input is surfaced as an I/O error instead of being
    /// accepted as an empty or partially decoded tensor.
    #[test]
    fn test_read_npy_rejects_invalid_data() {
        let err = read::<i32>(&b"not an npy file"[..]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// Covers the rank-2 boundary in Fortran-to-row-major conversion. The
    /// input is the Fortran flat representation of `[[1, 2, 3], [4, 5, 6]]`.
    #[test]
    fn test_fortran_order_to_row_major_converts_rank_2_data() {
        let converted = fortran_order_to_row_major(vec![1, 4, 2, 5, 3, 6], &[2, 3]);

        assert_eq!(converted, [1, 2, 3, 4, 5, 6]);
    }

    /// Checks that equivalent NumPy-generated C-order and Fortran-order files
    /// decode to the same row-major tensor.
    #[test]
    fn test_read_npy_c_and_fortran_order_fixtures_match() {
        let c_order = include_bytes!("../tests/fixtures/order_c_i32.npy");
        let fortran_order = include_bytes!("../tests/fixtures/order_fortran_i32.npy");

        let c_tensor = read::<i32>(&c_order[..]).unwrap();
        let fortran_tensor = read::<i32>(&fortran_order[..]).unwrap();
        let expected = Tensor::from([
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]);

        assert_eq!(&c_tensor, &expected);
        assert_eq!(&fortran_tensor, &expected);
        assert_eq!(&fortran_tensor, &c_tensor);
    }
}

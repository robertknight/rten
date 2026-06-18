use crate::tensor_from_npy_file;
use npyz::npz::{NpzArchive, NpzWriter};
use npyz::{AutoSerialize, Deserialize, WriterBuilder};
use rten_tensor::{Layout, Tensor, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

/// Serialize named tensors to an NPZ archive.
///
/// Tensor names may be passed with or without the `.npy` suffix.
pub fn write_npz<'a, T, I, N, W>(writer: W, arrays: I) -> io::Result<()>
where
    T: Clone + AutoSerialize + 'a,
    I: IntoIterator<Item = (N, TensorView<'a, T>)>,
    N: AsRef<str>,
    W: io::Write + io::Seek,
{
    let mut archive = NpzWriter::new(writer);

    for (name, array) in arrays {
        let name = npz_array_name(name.as_ref())?;
        let shape = array.shape().iter().map(|x| *x as u64).collect::<Vec<_>>();

        let mut writer = archive
            .array::<T>(&name, Default::default())?
            .default_dtype()
            .shape(&shape)
            .begin_nd()?;
        writer.extend(array.iter().cloned())?;
        writer.finish()?;
    }

    archive.zip_writer().finish()?;
    Ok(())
}

/// Serialize named tensors to an NPZ archive file - this will create the file.
pub fn write_npz_to_file<'a, T, I, N>(path: impl AsRef<Path>, arrays: I) -> io::Result<()>
where
    T: Clone + AutoSerialize + 'a,
    I: IntoIterator<Item = (N, TensorView<'a, T>)>,
    N: AsRef<str>,
{
    let file = io::BufWriter::new(File::create(path)?);
    write_npz(file, arrays)
}

/// Read all `.npy` entries from an NPZ archive.
///
/// Returned names have the `.npy` suffix removed.
pub fn read_npz<T: Clone + Deserialize>(
    reader: impl io::Read + io::Seek,
) -> io::Result<HashMap<String, Tensor<T>>> {
    let mut archive = NpzArchive::new(reader)?;
    let names = archive
        .array_names()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut arrays = HashMap::new();

    for name in names {
        let file = archive.by_name(&name)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing NPZ array `{name}`"),
            )
        })?;
        let array = tensor_from_npy_file(file)?;
        arrays.insert(name, array);
    }

    Ok(arrays)
}

/// Read all `.npy` entries from an NPZ archive file.
pub fn read_npz_from_file<T: Clone + Deserialize>(
    path: impl AsRef<Path>,
) -> io::Result<HashMap<String, Tensor<T>>> {
    let file = io::BufReader::new(File::open(path)?);
    read_npz(file)
}

/// Read a single named tensor from an NPZ archive.
///
/// `name` may be passed with or without the `.npy` suffix.
pub fn read_npz_array<T: Clone + Deserialize>(
    reader: impl io::Read + io::Seek,
    name: &str,
) -> io::Result<Tensor<T>> {
    let mut archive = NpzArchive::new(reader)?;
    let name = npz_array_name(name)?;
    let file = archive.by_name(&name)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing NPZ array `{name}`"),
        )
    })?;
    tensor_from_npy_file(file)
}

/// Read a single named tensor from an NPZ archive file.
pub fn read_npz_array_from_file<T: Clone + Deserialize>(
    path: impl AsRef<Path>,
    name: &str,
) -> io::Result<Tensor<T>> {
    let file = io::BufReader::new(File::open(path)?);
    read_npz_array(file, name)
}

fn npz_array_name(name: &str) -> io::Result<String> {
    if name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "NPZ array name must not be empty",
        ));
    }

    if name.ends_with(".npy") {
        Ok(name.strip_suffix(".npy").unwrap().to_string())
    } else {
        Ok(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rten_tensor::AsView;
    use std::io::{Cursor, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_FILE: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("rten-npy-npz-{}-{id}-{name}", std::process::id()))
    }

    fn assert_tensor_eq<T: Clone + PartialEq + std::fmt::Debug>(
        actual: &Tensor<T>,
        expected: &Tensor<T>,
    ) {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.to_vec(), expected.to_vec());
    }

    /// Verifies that multiple arrays written to an in-memory NPZ archive can be
    /// read back by name with matching shapes and values.
    #[test]
    fn round_trip_npz() {
        let a: Tensor<i32> = [[1, 2, 3], [4, 5, 6]].into();
        let b: Tensor<i32> = [7, 8, 9].into();

        let mut buffer = Cursor::new(Vec::new());
        write_npz(&mut buffer, [("a", a.view()), ("nested/b.npy", b.view())]).unwrap();

        buffer.set_position(0);
        let arrays = read_npz::<i32>(&mut buffer).unwrap();

        assert_eq!(arrays.len(), 2);
        assert_tensor_eq(arrays.get("a").unwrap(), &a);
        assert_tensor_eq(arrays.get("nested/b").unwrap(), &b);
    }

    /// Exercises the single-array reader and confirms callers can use either
    /// the logical array name or its `.npy` archive filename form.
    #[test]
    fn read_npz_array_accepts_suffixless_name() {
        let a: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        let mut buffer = Cursor::new(Vec::new());
        write_npz(&mut buffer, [("a.npy", a.view())]).unwrap();

        buffer.set_position(0);
        let read = read_npz_array::<f32>(&mut buffer, "a").unwrap();

        assert_tensor_eq(&read, &a);
    }

    /// Covers the NPZ file convenience helpers with a real file path and checks
    /// that the resulting archive preserves array data.
    #[test]
    fn round_trip_npz_file_helpers() {
        let path = temp_file("round-trip.npz");
        let a: Tensor<i32> = [[1, 2], [3, 4]].into();
        let b: Tensor<i32> = [5, 6, 7].into();

        write_npz_to_file(&path, [("a", a.view()), ("b", b.view())]).unwrap();
        let arrays = read_npz_from_file::<i32>(&path).unwrap();
        let b_read = read_npz_array_from_file::<i32>(&path, "b.npy").unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(arrays.len(), 2);
        assert_tensor_eq(arrays.get("a").unwrap(), &a);
        assert_tensor_eq(arrays.get("b").unwrap(), &b);
        assert_tensor_eq(&b_read, &b);
    }

    /// Ensures lookup of a missing NPZ array returns a not-found error instead
    /// of fabricating a default tensor or failing with an unrelated error.
    #[test]
    fn read_npz_array_reports_missing_name() {
        let a: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());
        write_npz(&mut buffer, [("a", a.view())]).unwrap();

        buffer.set_position(0);
        let err = read_npz_array::<i32>(&mut buffer, "missing").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// Confirms non-array entries in the underlying zip archive are ignored
    /// when reading all NPZ arrays.
    #[test]
    fn read_npz_ignores_non_npy_members() {
        let a: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut archive = NpzWriter::new(&mut buffer);
            let mut writer = archive
                .array::<i32>("a", Default::default())
                .unwrap()
                .default_dtype()
                .shape(&[3])
                .begin_nd()
                .unwrap();
            writer.extend(a.iter().cloned()).unwrap();
            writer.finish().unwrap();

            archive
                .zip_writer()
                .start_file("metadata.txt", Default::default())
                .unwrap();
            archive.zip_writer().write_all(b"not an array").unwrap();
            archive.zip_writer().finish().unwrap();
        }

        buffer.set_position(0);
        let arrays = read_npz::<i32>(&mut buffer).unwrap();

        assert_eq!(arrays.len(), 1);
        assert_tensor_eq(arrays.get("a").unwrap(), &a);
    }

    /// Checks validation of empty array names so invalid NPZ member names are
    /// rejected before writing an archive entry.
    #[test]
    fn rejects_empty_npz_array_name() {
        assert_eq!(
            npz_array_name("").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    /// Exercises the public write path for empty names rather than only the
    /// private normalizer helper.
    #[test]
    fn write_npz_rejects_empty_array_name() {
        let tensor: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());

        let err = write_npz(&mut buffer, [("", tensor.view())]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}

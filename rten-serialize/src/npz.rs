use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

use npyz::{AutoSerialize, Deserialize};
use rten_tensor::{Layout, Storage, Tensor, TensorBase};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Serialize named tensors to an NPZ archive.
///
/// Tensor names may be passed with or without the `.npy` suffix.
///
/// Arrays are stored uncompressed, matching the output of `numpy.savez`.
pub fn write<'a, T, S, L, I, N, W>(writer: W, arrays: I) -> io::Result<()>
where
    T: AutoSerialize + 'a,
    S: Storage<Elem = T>,
    L: Layout + Clone,
    I: IntoIterator<Item = (N, TensorBase<S, L>)>,
    N: AsRef<str>,
    W: io::Write + io::Seek,
{
    let mut archive = ZipWriter::new(writer);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    for (name, array) in arrays {
        archive.start_file(npz_file_name(name.as_ref())?, options)?;
        crate::npy::write(&mut archive, &array)?;
    }

    archive.finish()?;
    Ok(())
}

/// Serialize named tensors to an NPZ archive file - this will create the file.
pub fn write_to_file<'a, T, S, L, I, N>(path: impl AsRef<Path>, arrays: I) -> io::Result<()>
where
    T: AutoSerialize + 'a,
    S: Storage<Elem = T>,
    L: Layout + Clone,
    I: IntoIterator<Item = (N, TensorBase<S, L>)>,
    N: AsRef<str>,
{
    let file = io::BufWriter::new(File::create(path)?);
    write(file, arrays)
}

/// Read all `.npy` entries from an NPZ archive.
///
/// Returned names have the `.npy` suffix removed.
pub fn read<T: Clone + Deserialize>(
    reader: impl io::Read + io::Seek,
) -> io::Result<HashMap<String, Tensor<T>>> {
    let mut archive = ZipArchive::new(reader)?;
    let names = archive
        .file_names()
        .filter(|name| name.ends_with(".npy"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut arrays = HashMap::new();

    for name in names {
        let file = archive.by_name(&name)?;
        let array = crate::npy::read(file)?;
        let name = name.strip_suffix(".npy").unwrap().to_string();
        arrays.insert(name, array);
    }

    Ok(arrays)
}

/// Read all `.npy` entries from an NPZ archive file.
pub fn read_from_file<T: Clone + Deserialize>(
    path: impl AsRef<Path>,
) -> io::Result<HashMap<String, Tensor<T>>> {
    let file = io::BufReader::new(File::open(path)?);
    read(file)
}

/// Read a single named tensor from an NPZ archive.
///
/// `name` may be passed with or without the `.npy` suffix.
pub fn read_array<T: Clone + Deserialize>(
    reader: impl io::Read + io::Seek,
    name: &str,
) -> io::Result<Tensor<T>> {
    let mut archive = ZipArchive::new(reader)?;
    let file = archive.by_name(&npz_file_name(name)?)?;
    crate::npy::read(file)
}

/// Read a single named tensor from an NPZ archive file.
pub fn read_array_from_file<T: Clone + Deserialize>(
    path: impl AsRef<Path>,
    name: &str,
) -> io::Result<Tensor<T>> {
    let file = io::BufReader::new(File::open(path)?);
    read_array(file, name)
}

/// Map a user-supplied array name, with or without the `.npy` suffix, to the
/// archive entry filename, which always has exactly one `.npy` suffix.
fn npz_file_name(name: &str) -> io::Result<String> {
    let base = name.strip_suffix(".npy").unwrap_or(name);
    if base.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "NPZ array name must not be empty",
        ));
    }
    Ok(format!("{base}.npy"))
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

    /// Verifies that multiple arrays written to an in-memory NPZ archive can be
    /// read back by name with matching shapes and values.
    #[test]
    fn test_round_trip_npz() {
        let a: Tensor<i32> = [[1, 2, 3], [4, 5, 6]].into();
        let b: Tensor<i32> = [7, 8, 9].into();

        let mut buffer = Cursor::new(Vec::new());
        write(&mut buffer, [("a", a.view()), ("nested/b.npy", b.view())]).unwrap();

        buffer.set_position(0);
        let arrays = read::<i32>(&mut buffer).unwrap();

        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays.get("a").unwrap(), &a);
        assert_eq!(arrays.get("nested/b").unwrap(), &b);
    }

    /// Exercises the single-array reader and confirms callers can use either
    /// the logical array name or its `.npy` archive filename form.
    #[test]
    fn test_read_npz_array_accepts_suffixless_name() {
        let a: Tensor<f32> = [[1., 2.], [3., 4.]].into();

        let mut buffer = Cursor::new(Vec::new());
        write(&mut buffer, [("a.npy", a.view())]).unwrap();

        buffer.set_position(0);
        let read = read_array::<f32>(&mut buffer, "a").unwrap();

        assert_eq!(&read, &a);
    }

    /// Covers the NPZ file convenience helpers with a real file path and checks
    /// that the resulting archive preserves array data.
    #[test]
    fn test_round_trip_npz_file_helpers() {
        let path = temp_file("round-trip.npz");
        let a: Tensor<i32> = [[1, 2], [3, 4]].into();
        let b: Tensor<i32> = [5, 6, 7].into();

        write_to_file(&path, [("a", a.view()), ("b", b.view())]).unwrap();
        let arrays = read_from_file::<i32>(&path).unwrap();
        let b_read = read_array_from_file::<i32>(&path, "b.npy").unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays.get("a").unwrap(), &a);
        assert_eq!(arrays.get("b").unwrap(), &b);
        assert_eq!(&b_read, &b);
    }

    /// Ensures lookup of a missing NPZ array returns a not-found error instead
    /// of fabricating a default tensor or failing with an unrelated error.
    #[test]
    fn test_read_npz_array_reports_missing_name() {
        let a: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());
        write(&mut buffer, [("a", a.view())]).unwrap();

        buffer.set_position(0);
        let err = read_array::<i32>(&mut buffer, "missing").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// Confirms non-array entries in the underlying zip archive are ignored
    /// when reading all NPZ arrays.
    #[test]
    fn test_read_npz_ignores_non_npy_members() {
        let a: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());
        {
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            let mut archive = ZipWriter::new(&mut buffer);

            archive.start_file("a.npy", options).unwrap();
            crate::npy::write(&mut archive, &a).unwrap();

            archive.start_file("metadata.txt", options).unwrap();
            archive.write_all(b"not an array").unwrap();
            archive.finish().unwrap();
        }

        buffer.set_position(0);
        let arrays = read::<i32>(&mut buffer).unwrap();

        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays.get("a").unwrap(), &a);
    }

    /// Checks validation of empty array names so invalid NPZ member names are
    /// rejected before writing an archive entry.
    #[test]
    fn test_rejects_empty_npz_array_name() {
        assert_eq!(
            npz_file_name("").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    /// Exercises the public write path for empty names rather than only the
    /// private normalizer helper.
    #[test]
    fn test_write_npz_rejects_empty_array_name() {
        let tensor: Tensor<i32> = [1, 2, 3].into();
        let mut buffer = Cursor::new(Vec::new());

        let err = write(&mut buffer, [("", tensor.view())]).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}

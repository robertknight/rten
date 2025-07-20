use std::marker::PhantomData;

use rten_tensor::prelude::*;
use rten_tensor::{MatrixLayout, MatrixMut, StorageMut};

/// Wrapper around the GEMM output matrix which divides it into a grid of tiles.
/// This can be shared across threads, but each individual tile must only be
/// operated on by one thread at a time.
pub struct OutputTiles<'a, T> {
    data: *mut T,

    // Size and stride of the output matrix.
    rows: usize,
    cols: usize,
    row_stride: usize,

    // Maximum size of each tile.
    tile_rows: usize,
    tile_cols: usize,

    // Precomputed number of tiles along each axis.
    n_row_tiles: usize,
    n_col_tiles: usize,

    _marker: PhantomData<&'a mut [T]>,
}

/// Safety: Caller must ensure they do not operate on overlapping tiles
/// concurrently.
unsafe impl<T> Sync for OutputTiles<'_, T> {}

impl<'a, T> OutputTiles<'a, T> {
    /// Expose `data` as a grid of tiles, each with a maximum size of
    /// `tile_rows` * `tile_cols`.
    pub fn new(
        mut data: MatrixMut<'a, T>,
        tile_rows: usize,
        tile_cols: usize,
    ) -> OutputTiles<'a, T> {
        OutputTiles {
            data: data.storage_mut().as_mut_ptr(),
            rows: data.rows(),
            cols: data.cols(),
            row_stride: data.stride(0),
            tile_rows,
            tile_cols,
            n_row_tiles: data.rows().div_ceil(tile_rows),
            n_col_tiles: data.cols().div_ceil(tile_cols),
            _marker: PhantomData,
        }
    }

    /// Return the output tile with the given coordinates in the grid of
    /// output tiles.
    ///
    /// Safety: The caller must guarantee that every tile is operated on by
    /// only a single thread at a time.
    pub unsafe fn tile(&self, row: usize, col: usize) -> OutputTile<'_, T> {
        assert!(row < self.n_row_tiles && col < self.n_col_tiles);

        let start_row = row * self.tile_rows;
        let start_col = col * self.tile_cols;

        OutputTile {
            ptr: self.data.add(start_row * self.row_stride + start_col),
            row_stride: self.row_stride,
            used_rows: (self.rows - start_row).min(self.tile_rows),
            used_cols: (self.cols - start_col).min(self.tile_cols),
            _marker: PhantomData,
        }
    }
}

/// A single tile of the output matrix.
pub struct OutputTile<'a, T> {
    /// Pointer to first element in this tile.
    pub ptr: *mut T,

    /// Stride between rows of this tile. Note the column stride is always 1.
    pub row_stride: usize,

    /// Number of rows in this tile. Will be <= the [`Kernel`]'s `MR` constant.
    pub used_rows: usize,

    /// Number of columns in this tile. Will be <= the [`Kernel`]'s `NR` constant.
    pub used_cols: usize,

    _marker: PhantomData<&'a mut [T]>,
}

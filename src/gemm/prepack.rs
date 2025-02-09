use std::marker::PhantomData;
use std::ops::Range;

use rten_tensor::{Alloc, Matrix, MatrixLayout};

use super::packing::{PackElem, PackingBuffer};
use super::{depth_block_size, GemmError, Kernel, LhsBlock, RhsBlock};
use crate::iter_util::range_chunks;
use crate::tensor_pool::ExtractBuffer;

/// Common data and logic for a pre-packed A or B matrix.
///
/// The packed matrix is arranged as a series of blocks corresponding to slices
/// along the K (depth) dimension. Each block is subdivided into panels
/// corresponding to slices along the M or N dimensions. Each panel has size
/// `MR x K_block` or `K_block x NR` where `MR x NR` is the kernel's tile size.
/// The layout and data type of elements within each panel depends upon the kernel.
#[derive(Clone)]
struct PackedMatrixBase {
    data: PackingBuffer,

    /// Size of the short side of each panel. This will match the kernel's MR
    /// (for packed A) or NR (for packed B) dimensions.
    panel_size: usize,

    /// Cache-blocking size along the K dimension.
    depth_block: usize,

    /// Stride between each depth block.
    depth_block_stride: usize,

    /// Stride of panels in depth blocks except for the last.
    panel_stride: usize,

    /// Stride of panels in the last depth block. This will be smaller than
    /// `panel_stride` if the size of the K dimension is not a multiple of the
    /// depth block size.
    tail_panel_stride: usize,

    /// Size of the matrix along the N or M dimension.
    nm_size: usize,

    /// Size of the matrix along the K dimension.
    depth_size: usize,

    /// Name of the kernel used to pack the data.
    kernel_name: &'static str,
}

impl PackedMatrixBase {
    /// Retrieve a block from the packed matrix as a `(data, panel_stride)` tuple.
    ///
    /// `nm_range` is the range from the M or N dimensions and `depth_block_idx`
    /// sets the range along the K dimension.
    fn block(&self, nm_range: Range<usize>, depth_block_idx: usize) -> (&[u8], usize) {
        assert_eq!(nm_range.start % self.panel_size, 0);

        let n_blocks = self.depth_size.div_ceil(self.depth_block);
        let panel_stride = if depth_block_idx == n_blocks - 1 {
            self.tail_panel_stride
        } else {
            self.panel_stride
        };
        let depth_block_offset = depth_block_idx * self.depth_block_stride;

        let panel_range = nm_range.start / self.panel_size..nm_range.end.div_ceil(self.panel_size);
        let start = depth_block_offset + panel_range.start * panel_stride;
        let end = depth_block_offset + panel_range.end * panel_stride;
        let data = &self.data.as_bytes()[start..end];

        (data, panel_stride)
    }
}

/// Left-hand or "A" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedAMatrix<T> {
    base: PackedMatrixBase,
    _marker: PhantomData<T>,
}

impl<T> PackedAMatrix<T> {
    /// Return the packed data for a given range along the M and K dimensions.
    pub(super) fn block(&self, rows: Range<usize>, depth_block_idx: usize) -> LhsBlock<T> {
        let (data, panel_stride) = self.base.block(rows, depth_block_idx);
        LhsBlock::Packed { data, panel_stride }
    }

    /// Return number of columns in packed matrix.
    pub fn cols(&self) -> usize {
        self.base.depth_size
    }

    /// Return number of rows in packed matrix.
    pub fn rows(&self) -> usize {
        self.base.nm_size
    }

    /// Check that the kernel and blocking parameters used when packing this
    /// matrix match.
    pub fn validate<LhsT, RhsT, OutT>(
        &self,
        kernel: &dyn Kernel<LhsT, RhsT, OutT>,
        depth_block: usize,
    ) -> Result<(), GemmError> {
        if self.base.panel_size != kernel.mr() || self.base.kernel_name != kernel.name() {
            return Err(GemmError::PackedDataKernelMismatch);
        }
        if self.base.depth_block != depth_block {
            return Err(GemmError::PackedDataBlockingMismatch);
        }
        Ok(())
    }
}

impl<T> ExtractBuffer for PackedAMatrix<T> {
    type Elem = PackElem;

    fn extract_buffer(self) -> Option<Vec<Self::Elem>> {
        Some(self.base.data.into_vec())
    }
}

/// Right-hand or "B" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedBMatrix<T> {
    base: PackedMatrixBase,
    _marker: PhantomData<T>,
}

impl<T> PackedBMatrix<T> {
    /// Return the packed data for a given range along the N and K dimensions.
    pub(super) fn block(&self, cols: Range<usize>, depth_block_idx: usize) -> RhsBlock<T> {
        let (data, panel_stride) = self.base.block(cols, depth_block_idx);
        RhsBlock {
            data,
            panel_stride,
            _marker: PhantomData,
        }
    }

    /// Return number of columns in packed matrix.
    pub fn cols(&self) -> usize {
        self.base.nm_size
    }

    /// Return number of rows in packed matrix.
    pub fn rows(&self) -> usize {
        self.base.depth_size
    }

    /// Check that the kernel and blocking parameters used when packing this
    /// matrix match.
    pub fn validate<LhsT, RhsT, OutT>(
        &self,
        kernel: &dyn Kernel<LhsT, RhsT, OutT>,
        depth_block: usize,
    ) -> Result<(), GemmError> {
        if self.base.panel_size != kernel.nr() || self.base.kernel_name != kernel.name() {
            return Err(GemmError::PackedDataKernelMismatch);
        }
        if self.base.depth_block != depth_block {
            return Err(GemmError::PackedDataBlockingMismatch);
        }
        Ok(())
    }
}

impl<T> ExtractBuffer for PackedBMatrix<T> {
    type Elem = PackElem;

    fn extract_buffer(self) -> Option<Vec<Self::Elem>> {
        Some(self.base.data.into_vec())
    }
}

/// Prepack a GEMM LHS input for use with a given kernel.
pub fn prepack_a<A: Alloc, LhsT, RhsT, OutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    alloc: A,
    a: Matrix<LhsT>,
) -> PackedAMatrix<LhsT> {
    let depth_block = depth_block_size(a.cols());

    let layout = kernel.packed_a_layout(a, a.rows(), depth_block, None);
    let tail_layout = if a.cols() % depth_block != 0 {
        Some(kernel.packed_a_layout(a, a.rows(), a.cols() % depth_block, None))
    } else {
        None
    };

    // Require the size to be a multiple of the alignment. This avoids the
    // need for any gaps between blocks, which would have to be initialized
    // after packing.
    assert_eq!(layout.size() % layout.align(), 0);

    let n_blocks = a.cols() / depth_block;
    let total_size =
        (n_blocks * layout.size()) + tail_layout.as_ref().map(|l| l.size()).unwrap_or(0);

    let mut data = PackingBuffer::new();
    let uninit_data = data.alloc_in(alloc, total_size, layout.align());

    for (col_block, block_data) in
        range_chunks(0..a.cols(), depth_block).zip(uninit_data.chunks_mut(layout.size()))
    {
        kernel.pack_a_block(block_data, a, 0..a.rows(), col_block, None);
    }

    // Safety: We used `pack_a_block` to initialize `total_size` bytes
    unsafe {
        data.set_len(total_size);
    }

    PackedAMatrix {
        base: PackedMatrixBase {
            data,
            nm_size: a.rows(),
            depth_size: a.cols(),
            panel_size: kernel.mr(),
            depth_block,
            panel_stride: layout.panel_stride(),
            tail_panel_stride: tail_layout
                .map(|tl| tl.panel_stride())
                .unwrap_or(layout.panel_stride()),
            depth_block_stride: layout.size(),
            kernel_name: kernel.name(),
        },
        _marker: PhantomData,
    }
}

/// Prepack a GEMM RHS input for use with a given kernel.
pub fn prepack_b<A: Alloc, LhsT, RhsT, OutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    alloc: A,
    b: Matrix<RhsT>,
) -> PackedBMatrix<RhsT> {
    let depth_block = depth_block_size(b.rows());

    let layout = kernel.packed_b_layout(depth_block, b.cols(), None);
    let tail_layout = if b.rows() % depth_block != 0 {
        Some(kernel.packed_b_layout(b.rows() % depth_block, b.cols(), None))
    } else {
        None
    };

    // Require the size to be a multiple of the alignment. This avoids the
    // need for any gaps between blocks, which would have to be initialized
    // after packing.
    assert_eq!(layout.size() % layout.align(), 0);

    let n_blocks = b.rows() / depth_block;
    let total_size =
        (n_blocks * layout.size()) + tail_layout.as_ref().map(|l| l.size()).unwrap_or(0);
    let mut data = PackingBuffer::new();
    let uninit_data = data.alloc_in(alloc, total_size, layout.align());

    for (row_block, block_data) in
        range_chunks(0..b.rows(), depth_block).zip(uninit_data.chunks_mut(layout.size()))
    {
        kernel.pack_b_block(block_data, b, row_block, 0..b.cols(), None);
    }

    // Safety: We used `pack_b_block` to initialize `layout.size` bytes.
    unsafe {
        data.set_len(total_size);
    }

    PackedBMatrix {
        base: PackedMatrixBase {
            data,
            depth_size: b.rows(),
            nm_size: b.cols(),
            panel_size: kernel.nr(),
            depth_block,
            panel_stride: layout.panel_stride(),
            tail_panel_stride: tail_layout
                .map(|tl| tl.panel_stride())
                .unwrap_or(layout.panel_stride()),
            depth_block_stride: layout.size(),
            kernel_name: kernel.name(),
        },
        _marker: PhantomData,
    }
}

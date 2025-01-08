//! Machine-learning oriented matrix multiplication functions.
//!
//! This module provides a subset of BLAS-like functions that are used by neural
//! network operators. The primary functionality is general matrix
//! multiplication (gemm) with ML-oriented additions, but there are also
//! operations like vector-scalar products.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul, Range};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{Alloc, GlobalAlloc, Matrix, MatrixLayout, MatrixMut, NdTensorView, Storage};

use crate::iter_util::{range_chunks, MaybeParIter};
use crate::number::{Identities, Pod};
use crate::tensor_pool::ExtractBuffer;

mod errors;
mod im2col;
mod kernels;
mod packing;

pub use errors::GemmError;
pub use im2col::{ColOffsets, Im2Col, RowOffsets};
use kernels::generic::GenericKernel;
use kernels::Kernel;
use packing::{PackElem, PackingBuffer};

pub type GemmResult = Result<(), GemmError>;

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
    fn block(&self, rows: Range<usize>, depth_block_idx: usize) -> LhsBlock<T> {
        let (data, panel_stride) = self.base.block(rows, depth_block_idx);
        LhsBlock::Packed { data, panel_stride }
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
    fn block(&self, cols: Range<usize>, depth_block_idx: usize) -> RhsBlock<T> {
        let (data, panel_stride) = self.base.block(cols, depth_block_idx);
        RhsBlock {
            data,
            panel_stride,
            _marker: PhantomData,
        }
    }
}

impl<T> ExtractBuffer for PackedBMatrix<T> {
    type Elem = PackElem;

    fn extract_buffer(self) -> Option<Vec<Self::Elem>> {
        Some(self.base.data.into_vec())
    }
}

/// Left-hand or "A" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputA<'a, T> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a, T>),

    /// A matrix which has been pre-packed by [`GemmExecutor::prepack_a`].
    Packed(&'a PackedAMatrix<T>),
    // TODO - Support virtual "A" inputs, like `GemmInputB::Virtual`.
}

impl<T> GemmInputA<'_, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.rows(),
            Self::Packed(pm) => pm.base.nm_size,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.base.depth_size,
        }
    }
}

/// Trait implemented by GEMM input types.
pub trait GemmInT: Copy + Default + Send + Sync + Identities + Pod {}
impl GemmInT for f32 {}

/// Trait implemented by GEMM output types.
pub trait GemmOutT:
    Copy
    + PartialEq
    + Send
    + Sync
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + Identities
    + Pod
{
}
impl GemmOutT for f32 {}

/// Right-hand or "B" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputB<'a, T> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a, T>),

    /// A matrix which has been pre-packed by [`GemmExecutor::prepack_b`].
    Packed(&'a PackedBMatrix<T>),

    /// An image which is transformed into a matrix using an im2col transformation.
    Im2Col(&'a Im2Col<'a, T>),
}

impl<T: Copy + Default> GemmInputB<'_, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.rows(),
            Self::Packed(pm) => pm.base.depth_size,
            Self::Im2Col(im) => im.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.base.nm_size,
            Self::Im2Col(im) => im.cols(),
        }
    }
}

/// A bias to add to the output of a matrix multiplication.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BiasVector<'a, T> {
    /// Slice of values treated as a column vector. The length must match the
    /// number of rows in the LHS / A input.
    Column(&'a [T]),

    /// Slice of values treated as a column vector. The length must match the
    /// number of columns in the RHS / B input.
    Row(&'a [T]),
}

/// Perform a General Matrix Multiplication ("gemm").
///
/// This computes `output = alpha * (a @ b) + beta * output` where `@` is
/// matrix multiplication.
#[allow(unused)]
pub fn gemm<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    out_data: &mut [OutT],
    out_row_stride: usize,
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    alpha: f32,
    beta: OutT,
) -> GemmResult
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    // This heap-allocates a new kernel on each call. That's OK because this
    // is very cheap relative to the large matmuls we expect to be doing, but
    // would be good to avoid for small inputs.
    GemmExecutor::default().gemm(
        out_data,
        out_row_stride,
        GemmInputA::Unpacked(a),
        GemmInputB::Unpacked(b),
        alpha,
        beta,
        None, // bias
    )
}

/// Executes matrix multiplication operations.
///
/// For simple use cases, the standalone [`gemm`] function can be used.
/// GemmExecutor provides a more advanced API that enables features such as
/// performing matrix multiplications with pre-packed inputs.
///
/// ## Prepacking
///
/// Prepacking is useful when an input will be reused in multiple GEMM
/// operations. In this case the work to pack (re-layout) the input for maximum
/// computational efficiency, which is normally does internally on each call,
/// can be done just once for the reused input.
pub struct GemmExecutor<LhsT: GemmInT = f32, RhsT: GemmInT = f32, OutT: GemmOutT = f32> {
    kernel: Box<dyn Kernel<LhsT, RhsT, OutT>>,
    kernel_type: KernelType,
}

/// Arguments for [`GemmExecutor::with_kernel`] specifying which kernel to use.
#[derive(Clone, Copy, Debug)]
pub enum KernelType {
    /// Use the fallback/generic kernel. Always available.
    Generic,

    /// Use the AVX 2 + FMA kernel. Intel x64 only.
    #[cfg(target_arch = "x86_64")]
    Fma,

    /// Use the AVX 512 kernel. Intel x64 only.
    #[cfg(feature = "avx512")]
    Avx512,

    /// Use the ARM NEON kernel. ARM 64 only.
    #[cfg(target_arch = "aarch64")]
    ArmNeon,

    /// Use the WASM SIMD kernel. WASM only.
    #[cfg(target_arch = "wasm32")]
    #[cfg(target_feature = "simd128")]
    Wasm,
}

impl<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT> GemmExecutor<LhsT, RhsT, OutT> {
    /// Return the name of the kernel that this executor is using.
    #[allow(dead_code)]
    pub fn kernel_name(&self) -> &str {
        self.kernel.name()
    }

    /// Return the type of kernel being used.
    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    /// Prepack a matrix for use as the left-hand or "A" input.
    #[allow(unused)]
    pub fn prepack_a(&self, a: Matrix<LhsT>) -> PackedAMatrix<LhsT> {
        self.prepack_a_in(GlobalAlloc::new(), a)
    }

    /// Variant of [`prepack_a`](GemmExecutor::prepack_a) which takes an
    /// allocator.
    pub fn prepack_a_in<A: Alloc>(&self, alloc: A, a: Matrix<LhsT>) -> PackedAMatrix<LhsT> {
        let depth_block = depth_block_size(a.cols());

        let layout = self.kernel.packed_a_layout(a, a.rows(), depth_block);
        let tail_layout = if a.cols() % depth_block != 0 {
            Some(
                self.kernel
                    .packed_a_layout(a, a.rows(), a.cols() % depth_block),
            )
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
            self.kernel
                .pack_a_block(block_data, a, 0..a.rows(), col_block);
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
                panel_size: self.kernel.mr(),
                depth_block,
                panel_stride: layout.panel_stride(),
                tail_panel_stride: tail_layout
                    .map(|tl| tl.panel_stride())
                    .unwrap_or(layout.panel_stride()),
                depth_block_stride: layout.size(),
                kernel_name: self.kernel.name(),
            },
            _marker: PhantomData,
        }
    }

    /// Return column count step for building an [`Im2Col`] input.
    ///
    /// The number of columns in [`ColOffsets`] must be a multiple of this.
    pub fn im2col_col_count_step(&self) -> usize {
        self.kernel.nr()
    }

    /// Prepack a matrix for use as the right-hand or "B" matrix input.
    #[allow(unused)]
    pub fn prepack_b(&self, b: Matrix<RhsT>) -> PackedBMatrix<RhsT> {
        self.prepack_b_in(GlobalAlloc::new(), b)
    }

    /// Variant of [`prepack_b`](GemmExecutor::prepack_b) which takes an
    /// allocator.
    pub fn prepack_b_in<A: Alloc>(&self, alloc: A, b: Matrix<RhsT>) -> PackedBMatrix<RhsT> {
        let depth_block = depth_block_size(b.rows());

        let layout = self.kernel.packed_b_layout(depth_block, b.cols());
        let tail_layout = if b.rows() % depth_block != 0 {
            Some(
                self.kernel
                    .packed_b_layout(b.rows() % depth_block, b.cols()),
            )
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
            self.kernel
                .pack_b_block(block_data, b, row_block, 0..b.cols());
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
                panel_size: self.kernel.nr(),
                depth_block,
                panel_stride: layout.panel_stride(),
                tail_panel_stride: tail_layout
                    .map(|tl| tl.panel_stride())
                    .unwrap_or(layout.panel_stride()),
                depth_block_stride: layout.size(),
                kernel_name: self.kernel.name(),
            },
            _marker: PhantomData,
        }
    }

    /// Perform a General Matrix Multiplication ("gemm").
    ///
    /// This computes `output = alpha * (a @ b) + beta * output` where `@` is
    /// matrix multiplication.
    ///
    /// As a special case, when beta is `0.0`, the computation is simplified to
    /// `output = alpha * (a @ b)`. ie. existing values in `output` are not
    /// used. This matters if the existing values include infinities or NaNs.
    pub fn gemm(
        &self,
        out_data: &mut [OutT],
        out_row_stride: usize,
        a: GemmInputA<LhsT>,
        b: GemmInputB<RhsT>,
        alpha: f32,
        beta: OutT,
        bias: Option<BiasVector<OutT>>,
    ) -> GemmResult {
        gemm_impl(
            &*self.kernel,
            // Safety: `gemm_impl` only writes initialized values to `out_data`.
            unsafe { std::mem::transmute::<&mut [OutT], &mut [MaybeUninit<OutT>]>(out_data) },
            out_row_stride,
            a,
            b,
            alpha,
            beta,
            bias,
        )
    }

    /// Perform a General Matrix Multiplication ("gemm").
    ///
    /// This is the same as [`GemmExecutor::gemm`] but takes an uninitialized
    /// output slice. The `beta` value is implicitly set to zero.
    pub fn gemm_uninit(
        &self,
        out_data: &mut [MaybeUninit<OutT>],
        out_row_stride: usize,
        a: GemmInputA<LhsT>,
        b: GemmInputB<RhsT>,
        alpha: f32,
        bias: Option<BiasVector<OutT>>,
    ) -> GemmResult {
        gemm_impl(
            &*self.kernel,
            out_data,
            out_row_stride,
            a,
            b,
            alpha,
            OutT::zero(),
            bias,
        )
    }
}

impl GemmExecutor<f32, f32, f32> {
    /// Create a [`GemmExecutor`] using the preferred kernel for the current system.
    pub fn new() -> GemmExecutor {
        #[cfg(feature = "avx512")]
        #[cfg(target_arch = "x86_64")]
        if let Some(gemm) = Self::with_kernel(KernelType::Avx512) {
            return gemm;
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(gemm) = Self::with_kernel(KernelType::Fma) {
            return gemm;
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(gemm) = Self::with_kernel(KernelType::ArmNeon) {
            return gemm;
        }
        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        if let Some(gemm) = Self::with_kernel(KernelType::Wasm) {
            return gemm;
        }
        Self::with_generic_kernel()
    }

    /// Create a [`GemmExecutor`] using the given kernel. Returns `None` if the
    /// kernel is not supported.
    #[allow(dead_code)] // Currently only used in tests
    pub fn with_kernel(hint: KernelType) -> Option<Self> {
        fn make_kernel<K: Kernel<f32, f32, f32> + 'static>(
            kernel_type: KernelType,
        ) -> Option<GemmExecutor> {
            K::new().map(|kernel| GemmExecutor {
                kernel: Box::new(kernel),
                kernel_type,
            })
        }

        match hint {
            #[cfg(feature = "avx512")]
            #[cfg(target_arch = "x86_64")]
            KernelType::Avx512 => make_kernel::<kernels::x86_64::Avx512Kernel>(hint),
            #[cfg(target_arch = "x86_64")]
            KernelType::Fma => make_kernel::<kernels::x86_64::FmaKernel>(hint),
            #[cfg(target_arch = "aarch64")]
            KernelType::ArmNeon => make_kernel::<kernels::aarch64::ArmNeonKernel>(hint),
            #[cfg(target_arch = "wasm32")]
            #[cfg(target_feature = "simd128")]
            KernelType::Wasm => make_kernel::<kernels::wasm::WasmKernel>(hint),
            KernelType::Generic => Some(Self::with_generic_kernel()),
        }
    }

    /// Construct a GemmExecutor that uses the generic kernel.
    fn with_generic_kernel() -> Self {
        let kernel = GenericKernel::new().unwrap();
        GemmExecutor {
            kernel: Box::new(kernel),
            kernel_type: KernelType::Generic,
        }
    }
}

impl Default for GemmExecutor<f32, f32, f32> {
    fn default() -> Self {
        Self::new()
    }
}

/// Return the block size for the K / depth dimension of a GEMM operation.
///
/// This is chosen such that a `depth_block_size * nr` panel of B fits in the L1
/// cache, and can be reused in the loop over row tiles within each row block.
/// On AVX2 with f32 GEMM for example, NR=16 so `256 * 16 * 4 = 16KB`.
fn depth_block_size(a_cols: usize) -> usize {
    256.min(a_cols)
}

/// Return the block size for the N / column dimension of a GEMM operation.
///
/// The block size is always a multiple of `nr`, as the block is divided into
/// column tiles of width `nr`.
///
/// The value should be chosen such that a block of B of size
/// `row_block_size * depth_block_size` fits in the L3 cache. This is then
/// reused for each loop over row blocks within the column panel. In the current
/// implementation the column block size is also adjusted dynamically to enable
/// parallelism. In a multi-threaded context, all column blocks that exist
/// concurrently need to fit in the L3 cache.
fn col_block_size(b_cols: usize, nr: usize) -> usize {
    let parallelism = rayon::current_num_threads();
    let lower_bound = 128.min(b_cols);
    let unrounded = (b_cols / parallelism).max(lower_bound).min(1024);
    unrounded.next_multiple_of(nr)
}

/// Return the block size for the M / row dimension of a GEMM operation.
///
/// The block size is always a multiple of `mr`, as the block is divided into
/// row tiles of height `mr`.
///
/// The value should be chosen such that a panel of A of size `row_block_size *
/// depth_block_size` fits in the L2 cache. This is then reused for each
/// column tile that is visited within a block.
fn row_block_size(a_rows: usize, mr: usize) -> usize {
    64.min(a_rows).next_multiple_of(mr)
}

/// A single tile of the output matrix.
struct OutputTile<'a, T> {
    /// Pointer to first element in this tile.
    ptr: *mut T,

    /// Stride between rows of this tile. Note the column stride is always 1.
    row_stride: usize,

    /// Number of rows in this tile. Will be <= the [`Kernel`]'s `MR` constant.
    used_rows: usize,

    /// Number of columns in this tile. Will be <= the [`Kernel`]'s `NR` constant.
    used_cols: usize,

    _marker: PhantomData<&'a mut [T]>,
}

/// Wrapper around the GEMM output matrix which divides it into a grid of tiles.
/// This can be shared across threads, but each individual tile must only be
/// operated on by one thread at a time.
struct OutputTiles<'a, T> {
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
    fn new(mut data: MatrixMut<'a, T>, tile_rows: usize, tile_cols: usize) -> OutputTiles<'a, T> {
        OutputTiles {
            data: data.data_mut().unwrap().as_mut_ptr(),
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
    unsafe fn tile(&self, row: usize, col: usize) -> OutputTile<T> {
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

/// Compute a vector-matrix product.
///
/// This operation is called "gemv" in BLAS APIs.
fn gemv<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    a: NdTensorView<LhsT, 1>,
    b: Matrix<RhsT>,
    mut output_mat: MatrixMut<MaybeUninit<OutT>>,
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
) {
    assert!(output_mat.is_contiguous());

    let a_cols = a.size(0);
    let b_cols = b.cols();
    let out_data = output_mat.data_mut().unwrap();

    let a = a.to_contiguous();
    let a_data = a.data().unwrap();

    // The matrix is partitioned into column blocks that are processed in
    // parallel.
    //
    // Each column block is partitioned into row blocks for calls to the kernel.
    // The kernel internally divides the row blocks into column tiles. The
    // kernel prefers tall row blocks if B has unit row stride, or short row
    // blocks if it has unit column stride.
    let b_block_size = b_cols.div_ceil(rayon::current_num_threads()).max(128);
    let k_block_size = if b.row_stride() == 1 { 512 } else { 8 };

    out_data
        .par_chunks_mut(b_block_size)
        .enumerate()
        .for_each(|(col_block_idx, out_chunk)| {
            let col_block =
                (col_block_idx * b_block_size)..((col_block_idx + 1) * b_block_size).min(b_cols);
            let mut effective_beta = beta;

            for (k_block, a_block) in
                range_chunks(0..a_cols, k_block_size).zip(a_data.chunks(k_block_size))
            {
                let b_block = b.slice((k_block, col_block.clone()));
                kernel.gemv_kernel(out_chunk, a_block, b_block, alpha, effective_beta);

                // Reset `beta` so that subsequent updates for each column
                // accumulate into the first update.
                effective_beta = OutT::one();
            }

            // Safety: Calls to `gemv_kernel` initialized all output elements.
            let out_chunk =
                unsafe { std::mem::transmute::<&mut [MaybeUninit<OutT>], &mut [OutT]>(out_chunk) };
            match bias {
                Some(BiasVector::Column(bias)) => {
                    let bias = bias[0];
                    for x in out_chunk {
                        *x = *x + bias;
                    }
                }
                Some(BiasVector::Row(bias)) => {
                    let bias_block = &bias[col_block.clone()];
                    for (x, bias) in out_chunk.iter_mut().zip(bias_block) {
                        *x = *x + *bias;
                    }
                }
                None => {}
            }
        });
}

/// Perform matrix multiplication with a given kernel.
///
/// # Implementation notes
///
/// The implementation uses the general approach of
/// [BLIS](https://github.com/flame/blis), and was informed by the
/// [matrixmultiply crate](https://github.com/bluss/matrixmultiply). See [^1]
/// for an overview.
///
/// The main ideas of the implementation are 1) to minimize the overhead of
/// transferring data between memory and compute by effectively exploiting the
/// cache hierarchy and 2) to take advantage of available parallelism. The
/// operation is split into three levels, each comprised of several nested
/// loops. See pages 3-5 of [^1] for additional details.
///
///  1. The outer level divides the M/N/K dimensions of the problem into blocks,
///     sized to fit into different cache levels, and packs the corresponding
///     elements of the input matrices into a layout that is efficient for the
///     kernel to operate on.
///  2. The mid level ("macrokernel") divides the M / N dimensions into tiles
///     that are sized to fit into CPU registers.
///  3. The innermost level ("microkernel", or just "kernel" in this
///     implementation) updates a single output tile within the current block.
///
/// [^1]: Low, Tze Meng, et al. "Analytical modeling is enough for
///       high-performance BLIS." ACM Transactions on Mathematical Software (TOMS)
///       43.2 (2016): 1-18. https://dl.acm.org/doi/pdf/10.1145/2925987
fn gemm_impl<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    out_data: &mut [MaybeUninit<OutT>],
    out_row_stride: usize,
    a: GemmInputA<LhsT>,
    b: GemmInputB<RhsT>,
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
) -> GemmResult {
    if a.cols() != b.rows() {
        return Err(GemmError::KSizeMismatch);
    }

    let bias_ok = match bias {
        Some(BiasVector::Row(bias)) => bias.len() == b.cols(),
        Some(BiasVector::Column(bias)) => bias.len() == a.rows(),
        None => true,
    };
    if !bias_ok {
        return Err(GemmError::WrongBiasSize);
    }

    // Handle case where output is empty.
    if a.rows() == 0 || b.cols() == 0 {
        return Ok(());
    }

    // Construct a Matrix from the implied dimensions, to validate the slice length.
    let mut output_mat = MatrixMut::<MaybeUninit<OutT>>::from_data_with_strides(
        [a.rows(), b.cols()],
        out_data,
        [out_row_stride, 1],
    )
    .map_err(|_| GemmError::OutputNotLargeEnough)?;

    // Handle case where depth is zero. We still need to initialize the output
    // in this case.
    if a.cols() == 0 {
        let mut output_mat = if beta == OutT::zero() {
            output_mat.fill(MaybeUninit::new(OutT::zero()));

            // Safety: We just initialized the output.
            unsafe { output_mat.assume_init() }
        } else {
            // Safety: If beta is non-zero we assume the caller initialized the output.
            let mut output_mat = unsafe { output_mat.assume_init() };
            output_mat.apply(|x| *x * beta);
            output_mat
        };

        if let Some(bias) = bias {
            let bias_mat = match bias {
                BiasVector::Column(bias) => {
                    NdTensorView::from_data([a.rows()], bias).broadcast([a.rows(), b.cols()])
                }
                BiasVector::Row(bias) => {
                    NdTensorView::from_data([1, b.cols()], bias).broadcast([a.rows(), b.cols()])
                }
            };
            for r in 0..a.rows() {
                for c in 0..b.cols() {
                    let out_el = &mut output_mat[[r, c]];
                    *out_el = *out_el + bias_mat[[r, c]];
                }
            }
        }
        return Ok(());
    }

    // Use optimized path for vector-matrix products.
    if let (1, GemmInputA::Unpacked(a), GemmInputB::Unpacked(b)) = (a.rows(), a, b) {
        gemv(
            kernel,
            a.slice(0),
            b,
            output_mat.view_mut(),
            alpha,
            beta,
            // nb. We checked above that, if present, the bias length matches
            // `a.rows()` or `b.cols()` as appropriate.
            bias,
        );
        return Ok(());
    }

    let output_tiles = OutputTiles::new(output_mat, kernel.mr(), kernel.nr());

    // Sizes of blocks that the width (nc), depth (kc) and height (mc)
    // dimensions are partitioned into in the outer loops. These are chosen so
    // that blocks can fit in specific cache levels. See
    // https://dl.acm.org/doi/pdf/10.1145/2925987 for notes on choosing the
    // values.
    let nc = col_block_size(b.cols(), kernel.nr());
    let mc = row_block_size(a.rows(), kernel.mr());
    let kc = depth_block_size(a.cols());

    // If using prepacked inputs, make sure they were packed with the same
    // configuration we are using now.
    if let GemmInputA::Packed(packed) = &a {
        if packed.base.panel_size != kernel.mr() || packed.base.kernel_name != kernel.name() {
            return Err(GemmError::PackedDataKernelMismatch);
        }
        if packed.base.depth_block != kc {
            return Err(GemmError::PackedDataBlockingMismatch);
        }
    }
    if let GemmInputB::Packed(packed) = &b {
        if packed.base.panel_size != kernel.nr() || packed.base.kernel_name != kernel.name() {
            return Err(GemmError::PackedDataKernelMismatch);
        }
        if packed.base.depth_block != kc {
            return Err(GemmError::PackedDataBlockingMismatch);
        }
    }

    // Buffers for packed blocks of the matrix.
    thread_local!(static PACKED_A: RefCell<PackingBuffer> = const { RefCell::new(PackingBuffer::new()) });
    thread_local!(static PACKED_B: RefCell<PackingBuffer> = const { RefCell::new(PackingBuffer::new()) });

    let n_col_blocks = b.cols().div_ceil(nc);
    let n_row_blocks = a.rows().div_ceil(mc);

    // In a single-threaded context we get better performance by avoiding Rayon
    // overhead altogether.
    let parallel = rayon::current_num_threads() > 1;

    let (mr, nr) = (kernel.mr(), kernel.nr());

    // Loop over column blocks.
    (0..n_col_blocks)
        .maybe_par_iter(parallel)
        .for_each(|col_idx| {
            let col_start = col_idx * nc;
            let col_end = (col_start + nc).min(b.cols());
            let col_range = col_start..col_end;

            // Loop over depth blocks. This is not parallelized because output
            // tiles are shared across iterations.
            for (depth_block_idx, depth_range) in range_chunks(0..a.cols(), kc).enumerate() {
                // Borrowed packing buffer for current thread. Returned after
                // the GEMM block is computed.
                let mut thread_local_packed_b: Option<PackingBuffer> = None;

                let rhs_block = match b {
                    GemmInputB::Unpacked(_) | GemmInputB::Im2Col(_) => PACKED_B.with(|cell| {
                        let mut packed_b = cell.take();

                        let layout = kernel.packed_b_layout(depth_range.len(), col_end - col_start);
                        let packed_uninit = packed_b.alloc(layout.size(), layout.align());

                        match b {
                            GemmInputB::Unpacked(b) => kernel.pack_b_block(
                                packed_uninit,
                                b,
                                depth_range.clone(),
                                col_start..col_end,
                            ),
                            GemmInputB::Im2Col(im) => kernel.pack_im2col(
                                packed_uninit,
                                im,
                                depth_range.clone(),
                                col_start..col_end,
                            ),
                            GemmInputB::Packed(_) => unreachable!(),
                        }

                        // Safety: `pack_b_block` will have initialized `layout.size()` bytes.
                        unsafe {
                            packed_b.set_len(layout.size());
                        }
                        thread_local_packed_b = Some(packed_b);
                        RhsBlock {
                            data: thread_local_packed_b.as_ref().unwrap().as_bytes(),
                            panel_stride: layout.panel_stride(),
                            _marker: PhantomData,
                        }
                    }),
                    GemmInputB::Packed(pm) => pm.block(col_range.clone(), depth_block_idx),
                };

                // Only use provided `beta` on the first write to this output
                // tile. For subsequent updates accumulate.
                let effective_beta = if depth_range.start == 0 {
                    beta
                } else {
                    OutT::one()
                };

                // Loop over row blocks.
                (0..n_row_blocks)
                    .maybe_par_iter(parallel)
                    .for_each(|row_idx| {
                        let row_start = row_idx * mc;
                        let row_end = (row_start + mc).min(a.rows());
                        let row_range = row_start..row_end;

                        // Borrowed packing buffer for current thread. Returned after
                        // the GEMM block is computed.
                        let mut thread_local_packed_a: Option<PackingBuffer> = None;

                        let lhs_block = match a {
                            GemmInputA::Unpacked(a) => PACKED_A.with(|cell| {
                                let layout = kernel.packed_a_layout(
                                    a,
                                    row_end - row_start,
                                    depth_range.len(),
                                );
                                if !layout.must_pack {
                                    return LhsBlock::Unpacked(a);
                                };

                                let mut packed_a = cell.take();
                                let packed_uninit = packed_a.alloc(layout.size(), layout.align());

                                kernel.pack_a_block(
                                    packed_uninit,
                                    a,
                                    row_start..row_end,
                                    depth_range.clone(),
                                );

                                // Safety: We initialized `layout.size` bytes.
                                unsafe {
                                    packed_a.set_len(layout.size());
                                }
                                thread_local_packed_a = Some(packed_a);
                                LhsBlock::Packed {
                                    data: thread_local_packed_a.as_ref().unwrap().as_bytes(),
                                    panel_stride: layout.panel_stride(),
                                }
                            }),
                            GemmInputA::Packed(pm) => pm.block(row_range.clone(), depth_block_idx),
                        };

                        gemm_block(
                            kernel,
                            &output_tiles,
                            col_start / nr..col_end.div_ceil(nr),
                            row_start / mr..row_end.div_ceil(mr),
                            depth_range.clone(),
                            lhs_block,
                            rhs_block,
                            alpha,
                            effective_beta,
                            bias,
                        );

                        if let Some(packed_a) = thread_local_packed_a {
                            PACKED_A.with(|cell| cell.replace(packed_a));
                        }
                    });

                if let Some(packed_b) = thread_local_packed_b {
                    PACKED_B.with(|cell| cell.replace(packed_b));
                }
            }
        });

    Ok(())
}

/// LHS / A input for a call to [`gemm_block`].
#[derive(Copy, Clone)]
enum LhsBlock<'a, T> {
    /// Packed block of A matrix, arranged as a sequence of row panels.
    Packed {
        data: &'a [u8],

        /// Stride between each row panel.
        panel_stride: usize,
    },

    /// Unpacked A matrix. This must have a column stride of 1.
    Unpacked(Matrix<'a, T>),
}

/// LHS / B input for a call to [`gemm_block`]. Currently always packed as
/// a sequence of column panels.
#[derive(Copy, Clone)]
struct RhsBlock<'a, T> {
    data: &'a [u8],

    /// Stride between each column panel.
    panel_stride: usize,

    _marker: PhantomData<T>,
}

/// Process a single block (ie. a slice along each of the M/N/K dimensions) of a
/// matrix multiplication.
///
/// `col_tiles` and `row_tiles` specifies the range of output tiles to update.
/// `a` and `b` are the inputs. `depth_range` specifies the range along the K
/// dimension.
fn gemm_block<LhsT, RhsT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    output: &OutputTiles<MaybeUninit<OutT>>,
    col_tiles: Range<usize>,
    row_tiles: Range<usize>,
    depth_range: Range<usize>,
    a: LhsBlock<LhsT>,
    b: RhsBlock<RhsT>,
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
) {
    let (mr, nr) = (kernel.mr(), kernel.nr());

    // Sanity check input length here rather than inner loop.
    if let LhsBlock::Unpacked(mat) = &a {
        assert!(mat.rows().div_ceil(mr) >= row_tiles.end);
        assert!(mat.cols() >= depth_range.end);
    }

    // Loop over column tiles.
    //
    // TODO - This should be parallel, but threading overhead needs to be reduced.
    col_tiles
        .enumerate()
        .for_each(|(block_col_tile, col_tile)| {
            let b_panel_offset = block_col_tile * b.panel_stride;
            let b_panel = &b.data[b_panel_offset..b_panel_offset + b.panel_stride];

            // Loop over row tiles.
            for (block_row_tile, row_tile) in row_tiles.clone().enumerate() {
                // Safety:
                //  - The loops in this function and its caller are set up so that
                //    every output tile is processed by one thread at a time.
                let out_tile = unsafe { output.tile(row_tile, col_tile) };

                let kernel_lhs = match a {
                    LhsBlock::Packed {
                        data,
                        panel_stride,
                        // panel_len,
                    } => {
                        let a_panel_offset = block_row_tile * panel_stride;
                        let a_panel = &data[a_panel_offset..a_panel_offset + panel_stride];
                        kernels::Lhs::Packed(a_panel)
                    }
                    LhsBlock::Unpacked(mat) => {
                        let storage = mat.storage();
                        let offset =
                            row_tile * mr * mat.row_stride() + depth_range.start * mat.col_stride();
                        kernels::Lhs::Unpacked {
                            // Safety:
                            //  - `offset` is a valid storage offset within `mat`
                            data: unsafe { storage.as_ptr().add(offset) },
                            len: storage.len().saturating_sub(offset),
                            row_stride: mat.row_stride(),
                        }
                    }
                };

                // Safety:
                //  - Kernel is supported on current system
                //  - Output tile is initialized if beta is non-zero
                unsafe {
                    kernel.kernel(
                        out_tile.ptr as *mut OutT,
                        out_tile.row_stride,
                        kernel_lhs,
                        b_panel,
                        out_tile.used_rows,
                        out_tile.used_cols,
                        depth_range.len(),
                        alpha,
                        beta,
                    );
                }

                // Add bias vector on first write to an output tile.
                if depth_range.start == 0 {
                    // After the kernel is called, all elements of the output
                    // tile are now initialized.
                    let out_ptr = out_tile.ptr as *mut OutT;
                    match bias {
                        Some(BiasVector::Column(bias)) => {
                            for row in 0..out_tile.used_rows {
                                for col in 0..out_tile.used_cols {
                                    // Safety:
                                    //  - Row and column indices are valid for current tile
                                    //  - Bias length was checked at start of `gemm_impl`
                                    unsafe {
                                        let out_el = out_ptr.add(row * out_tile.row_stride + col);
                                        *out_el =
                                            *out_el + *bias.get_unchecked(row_tile * mr + row);
                                    }
                                }
                            }
                        }
                        Some(BiasVector::Row(bias)) => {
                            for row in 0..out_tile.used_rows {
                                for col in 0..out_tile.used_cols {
                                    // Safety:
                                    //  - Row and column indices are valid for current tile
                                    //  - Bias length was checked at start of `gemm_impl`
                                    unsafe {
                                        let out_el = out_ptr.add(row * out_tile.row_stride + col);
                                        *out_el =
                                            *out_el + *bias.get_unchecked(col_tile * nr + col);
                                    }
                                }
                            }
                        }
                        None => {}
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::time::Instant;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor, NdTensorView, Tensor};

    use super::{
        BiasVector, ColOffsets, GemmError, GemmExecutor, GemmInputA, GemmInputB, Im2Col,
        KernelType, RowOffsets,
    };

    fn reference_matmul_alpha_beta(a: &Tensor, b: &Tensor, alpha: f32, beta: f32) -> Tensor {
        let [a_rows, _a_cols]: [usize; 2] = a.shape().try_into().expect("input should be a matrix");
        let [_b_rows, b_cols]: [usize; 2] = b.shape().try_into().expect("input should be a matrix");
        let mut output = Tensor::zeros(&[a_rows, b_cols]);

        reference_gemm(&mut output, a, b, alpha, beta, None);

        output
    }

    fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        reference_matmul_alpha_beta(a, b, 1., 0.)
    }

    // Maximum block sizes that the GEMM implementation uses. Choosing M, N, K
    // inputs larger than this will ensure that multiple blocks are used along
    // that dimension.
    //
    // A better approach would be to make these block sizes configurable and set
    // them to small values in tests, so tests can enforce the use of multiple
    // blocks without needing large inputs that are slow when tests are compiled
    // in debug mode.
    const ROW_BLOCK_SIZE: usize = 64;
    const COL_BLOCK_SIZE: usize = 1024;
    const DEPTH_BLOCK_SIZE: usize = 256;

    /// Run a GEMM operation using the kernel specified by `kernel`, or the
    /// default kernel for the current system if None.
    fn run_gemm(
        output: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        bias: Option<BiasVector<f32>>,
        kernel: Option<KernelType>,
    ) {
        let out_row_stride = output.stride(0);
        let gemm = if let Some(kernel) = kernel {
            GemmExecutor::with_kernel(kernel).expect("kernel not available")
        } else {
            GemmExecutor::new()
        };

        gemm.gemm(
            output.data_mut().expect("expected contiguous input"),
            out_row_stride,
            GemmInputA::Unpacked(a.nd_view()),
            GemmInputB::Unpacked(b.nd_view()),
            alpha,
            beta,
            bias,
        )
        .unwrap();
    }

    /// Very slow but simple reference implementation. This should produce the
    /// same results as the optimized GEMM, but small numerical differences will
    /// appear in problems with a large K dimension, due to the different
    /// ordering of floating-point operations.
    fn reference_gemm(
        output: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        bias: Option<BiasVector<f32>>,
    ) {
        let [a_rows, a_cols]: [usize; 2] = a.shape().try_into().expect("input should be a matrix");
        let [_b_rows, b_cols]: [usize; 2] = b.shape().try_into().expect("input should be a matrix");

        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut accum = 0.0;
                for k in 0..a_cols {
                    accum += a[[r, k]] * b[[k, c]];
                }
                let bias = match bias {
                    Some(BiasVector::Row(b)) => b[c],
                    Some(BiasVector::Column(b)) => b[r],
                    None => 0.,
                };
                output[[r, c]] = alpha * accum + beta * output[[r, c]] + bias;
            }
        }
    }

    // Simplest possible test case for easy debugging.
    #[test]
    fn test_simple_gemm() -> Result<(), Box<dyn Error>> {
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![5., 6., 7., 8.]);
        let expected = reference_matmul(&a, &b);

        let mut result = Tensor::zeros(&[a.size(0), b.size(1)]);
        run_gemm(&mut result, &a, &b, 1., 1., None, None);
        expect_equal(&result, &expected)?;

        let mut result = Tensor::zeros(&[a.size(0), b.size(1)]);
        run_gemm(&mut result, &a, &b, 1., 1., None, Some(KernelType::Generic));
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_input_errors() {
        struct Case {
            a: NdTensor<f32, 2>,
            b: NdTensor<f32, 2>,
            output_len: usize,
            output_row_stride: usize,
            expected: GemmError,
        }

        let cases = [
            Case {
                a: NdTensor::from([[1., 2.], [3., 4.]]),
                b: NdTensor::from([[1., 2.], [3., 4.]]),
                output_len: 2,
                output_row_stride: 2,
                expected: GemmError::OutputNotLargeEnough,
            },
            Case {
                a: NdTensor::from([[1.], [2.]]),
                b: NdTensor::from([[1., 2.], [3., 4.]]),
                output_len: 4,
                output_row_stride: 2,
                expected: GemmError::KSizeMismatch,
            },
        ];

        let gemm = GemmExecutor::default();

        for Case {
            a,
            b,
            output_len,
            output_row_stride,
            expected,
        } in cases
        {
            let mut output = vec![0.; output_len];

            let result = gemm.gemm(
                &mut output,
                output_row_stride,
                GemmInputA::Unpacked(a.view()),
                GemmInputB::Unpacked(b.view()),
                1.,   // alpha
                0.,   // beta
                None, // bias
            );
            assert_eq!(result, Err(expected));
        }
    }

    fn test_gemm_with_kernel(kernel: Option<KernelType>) -> Result<(), Box<dyn Error>> {
        // "Interesting" sizes for the row, column and depth dimensions of the
        // computation. These are chosen to cover cases that are less than,
        // equal to and above the tile/block sizes which the algorithm divides
        // the problem into along each dimension.
        let col_steps = [0, 2, 4, 5, 8, 1024, 1025];
        let depth_steps = [0, 2, 20, 256, 300];
        let row_steps = [0, 2, 8, 10, 16, 64, 80];

        let mut cases = Vec::new();

        // Simple cases where one dimension of the problem is varied to
        // different interesting values and other dimensions are kept small.
        for cs in col_steps {
            cases.push(([2, 2], [2, cs]));
        }
        for ds in depth_steps {
            cases.push(([2, ds], [ds, 2]));
        }
        for rs in row_steps {
            cases.push(([rs, 2], [2, 2]));
        }

        // Some simple square matrix tests of different sizes. This covers all
        // cases below a threshold, and then select sizes after that. This is
        // because larger sizes are slow in debug builds.
        for n in 1..20 {
            cases.push(([n, n], [n, n]));
        }
        for n in [30, 64, 65] {
            cases.push(([n, n], [n, n]));
        }

        for (lhs_size, rhs_size) in cases {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(&lhs_size, &mut rng);
            let b = Tensor::rand(&rhs_size, &mut rng);
            let mut result = Tensor::zeros(&[lhs_size[0], rhs_size[1]]);

            run_gemm(&mut result, &a, &b, 1., 0., None, kernel);

            let expected = reference_matmul(&a, &b);

            if let Err(err) = expect_equal(&result, &expected) {
                println!(
                    "GEMM output for {}x{}x{} did not match reference",
                    lhs_size[0], rhs_size[1], lhs_size[1]
                );
                return Err(err.into());
            }
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemm_with_fma_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(Some(KernelType::Fma))
    }

    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemm_with_avx512_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(Some(KernelType::Avx512))
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_with_arm_neon_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(Some(KernelType::ArmNeon))
    }

    // This duplicates one of the other `test_gemm_with_XXX_kernel` tests
    // depending on what the preferred kernel is. That's OK as long as this
    // test is fast.
    #[test]
    fn test_gemm_with_auto_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(None)
    }

    #[test]
    fn test_gemm_with_generic_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(Some(KernelType::Generic))
    }

    #[test]
    fn test_gemm_transposed() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let mut a = Tensor::rand(&[20, 30], &mut rng);
        let mut b = Tensor::rand(&[10, 20], &mut rng);

        // Transpose the input matrices. This will alter their row and column
        // strides and shapes, but not re-order the data.
        a.permute(&[1, 0]);
        b.permute(&[1, 0]);

        let [a_rows, _]: [usize; 2] = a.shape().try_into().unwrap();
        let [_, b_cols]: [usize; 2] = b.shape().try_into().unwrap();

        let mut result = Tensor::zeros(&[a_rows, b_cols]);
        run_gemm(&mut result, &a, &b, 1., 1., None, None);

        let expected = reference_matmul(&a, &b);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_alpha() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        let a = Tensor::rand(&[10, 5], &mut rng);
        let b = Tensor::rand(&[5, 15], &mut rng);

        for kernel in [None, Some(KernelType::Generic)] {
            for alpha in [0.0, 0.5, 1.0, 2.0] {
                let mut result = Tensor::rand(&[10, 15], &mut rng);
                let mut expected = result.clone();

                run_gemm(&mut result, &a, &b, alpha, 0.0, None, kernel);
                reference_gemm(&mut expected, &a, &b, alpha, 0.0, None);

                expect_equal(&result, &expected)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_beta() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [Case { m: 10, k: 5, n: 15 }, Case { m: 10, k: 0, n: 15 }];

        for Case { m, n, k } in cases {
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);

            for kernel in [None, Some(KernelType::Generic)] {
                for beta in [0.5, 1.0, 2.0] {
                    let mut result = Tensor::rand(&[m, n], &mut rng);
                    let mut expected = result.clone();

                    run_gemm(&mut result, &a, &b, 1., beta, None, kernel);
                    reference_gemm(&mut expected, &a, &b, 1., beta, None);

                    expect_equal(&result, &expected)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_beta_zero() -> Result<(), Box<dyn Error>> {
        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            // Matrix-matrix multiplication
            Case {
                m: 20,
                n: 20,
                k: 20,
            },
            Case { m: 5, n: 5, k: 0 },
            // Vector-matrix multiplication
            Case { m: 1, n: 20, k: 20 },
        ];

        for Case { m, n, k } in cases {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);

            let mut result = Tensor::full(&[m, n], f32::NAN);
            let mut expected = Tensor::zeros(result.shape());

            // Test alpha values for which we may have special cases (0, 1) and
            // the general case.
            for alpha in [0., 0.5, 1.] {
                run_gemm(&mut result, &a, &b, alpha, 0. /* beta */, None, None);
                reference_gemm(&mut expected, &a, &b, alpha, 0. /* beta */, None);
                expect_equal(&result, &expected)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_bias() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            // Matrix-matrix
            Case { m: 10, n: 15, k: 5 },
            // Vector-matrix
            Case { m: 1, n: 15, k: 5 },
            // Vector-matrix, where n > minimum block size
            Case { m: 1, n: 129, k: 1 },
            // Case where k == 0
            Case { m: 5, n: 5, k: 0 },
        ];

        for Case { m, n, k } in cases {
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);

            let mut result = Tensor::zeros(&[m, n]);
            let mut expected = result.clone();

            // Column vector bias
            let bias: Vec<f32> = (0..a.shape()[0]).map(|b| b as f32).collect();
            run_gemm(
                &mut result,
                &a,
                &b,
                1.,
                0.,
                Some(BiasVector::Column(&bias)),
                None,
            );
            reference_gemm(
                &mut expected,
                &a,
                &b,
                1.,
                0.,
                Some(BiasVector::Column(&bias)),
            );

            // Row vector bias
            let bias: Vec<f32> = (0..b.shape()[1]).map(|b| b as f32).collect();
            run_gemm(
                &mut result,
                &a,
                &b,
                1.,
                0.,
                Some(BiasVector::Row(&bias)),
                None,
            );
            reference_gemm(&mut expected, &a, &b, 1., 0., Some(BiasVector::Row(&bias)));

            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_gemm_prepack() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }
        let cases = [
            // Small input that uses one block along each dimension.
            Case { m: 10, n: 15, k: 5 },
            // Inputs with one dimension large enough to require multiple blocks.
            Case {
                m: ROW_BLOCK_SIZE * 2 + ROW_BLOCK_SIZE / 2,
                n: 15,
                k: 5,
            },
            Case {
                m: 10,
                n: COL_BLOCK_SIZE * 2 + COL_BLOCK_SIZE / 2,
                k: 5,
            },
            Case {
                m: 10,
                n: 15,
                k: DEPTH_BLOCK_SIZE * 2 + DEPTH_BLOCK_SIZE / 2,
            },
        ];

        for case in cases {
            let Case { m, n, k } = case;
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);

            let a_mat: Matrix = a.nd_view();
            let b_mat: Matrix = b.nd_view();
            let gemm = GemmExecutor::new();

            let packed_a = gemm.prepack_a(a_mat);
            let packed_b = gemm.prepack_b(b.nd_view());

            let mut result = Tensor::zeros(&[m, n]);
            let result_row_stride = result.stride(0);

            gemm.gemm(
                result.data_mut().unwrap(),
                result_row_stride,
                GemmInputA::Packed(&packed_a),
                GemmInputB::Packed(&packed_b),
                1.,   // alpha
                1.,   // beta
                None, // bias
            )
            .unwrap();

            // Compare the results of pre-packed GEMM to unpacked GEMM rather
            // than reference GEMM because a) it is faster for large inputs
            // and b) in the case where K is large, the accumulated numerical
            // differences will be smaller.
            let mut expected = Tensor::zeros(result.shape());
            let expected_row_stride = expected.stride(0);
            gemm.gemm(
                expected.data_mut().unwrap(),
                expected_row_stride,
                GemmInputA::Unpacked(a_mat),
                GemmInputB::Unpacked(b_mat),
                1.,   // alpha
                1.,   // beta
                None, // bias
            )
            .unwrap();

            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    // Simplified version of the im2col builder used by convolution code.
    //
    // This builds a mapping between elements of an image and a
    // `[chans, height x width]` matrix where `image[c, y, x]` maps to
    // `im2col_matrix[c, y / width, y % width]`.
    fn build_im2col(image: NdTensorView<f32, 3>, col_count_step: usize) -> Im2Col<f32> {
        let [chans, img_h, img_w] = image.shape();
        let [chan_stride, h_stride, w_stride] = image.strides();

        let rows = chans;
        let n_cols = img_w * img_h;
        let n_cols_padded = n_cols.next_multiple_of(col_count_step);

        let row_offsets = RowOffsets {
            chan: (0..rows as i32)
                .map(|chan| chan * chan_stride as i32)
                .collect(),
            y: vec![0; rows],
            x: vec![0; rows],
        };
        let mut col_offsets = ColOffsets {
            y: (0..n_cols)
                .map(|i| i as i32 / img_w as i32)
                .map(|y| y * h_stride as i32)
                .collect(),
            x: (0..n_cols)
                .map(|i| i as i32 % img_w as i32)
                .map(|x| x * w_stride as i32)
                .collect(),
        };
        for _ in n_cols..n_cols_padded {
            col_offsets.y.push(0);
            col_offsets.x.push(0);
        }

        let max_y_offset = (img_h - 1) * h_stride;
        let max_x_offset = (img_w - 1) * w_stride;

        Im2Col {
            image,
            row_offsets,
            col_offsets,
            n_cols,
            max_y_offset: max_y_offset as i32,
            max_x_offset: max_x_offset as i32,
        }
    }

    #[test]
    fn test_gemm_im2col() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let gemm = GemmExecutor::default();

        // nb. If the test fails, debug by setting dimensions to 1.
        let img_h = 2;
        let img_w = 2;
        let img_chans = 2;
        let kernel_chans = 3;

        let img = NdTensor::<f32, 3>::rand([img_chans, img_h, img_w], &mut rng);
        let im2col = build_im2col(img.view(), gemm.im2col_col_count_step());

        let kernel_mat = NdTensor::<f32, 2>::rand([kernel_chans, img_chans], &mut rng);
        let mut output_mat = NdTensor::<f32, 2>::zeros([kernel_chans, img_h * img_w]);
        let out_row_stride = output_mat.row_stride();

        gemm.gemm(
            output_mat.data_mut().unwrap(),
            out_row_stride,
            GemmInputA::Unpacked(kernel_mat.view()),
            GemmInputB::Im2Col(&im2col),
            1.,   // alpha
            0.,   // beta
            None, // bias
        )
        .unwrap();

        let mut expected = NdTensor::<f32, 2>::zeros([kernel_chans, im2col.cols()]);
        for i in 0..expected.rows() {
            for j in 0..expected.cols() {
                let mut acc = 0.;
                for k in 0..kernel_mat.cols() {
                    acc += kernel_mat[[i, k]] * img[[k, j / img_w, j % img_w]];
                }
                expected[[i, j]] = acc;
            }
        }
        expect_equal(&output_mat, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemv() -> Result<(), Box<dyn Error>> {
        enum Strides {
            Contiguous,
            Transposed,
            Other,
        }

        struct Case {
            n: usize,
            k: usize,
            alpha: f32,
            beta: f32,
            bias: Option<f32>,
            b_strides: Strides,
        }

        impl Default for Case {
            fn default() -> Case {
                Case {
                    n: 16,
                    k: 16,
                    alpha: 1.,
                    beta: 0.,
                    bias: None,
                    b_strides: Strides::Contiguous,
                }
            }
        }

        let cases = [
            // Empty inputs
            Case {
                n: 0,
                k: 1,
                ..Default::default()
            },
            Case {
                n: 1,
                k: 0,
                ..Default::default()
            },
            // Smallest possible input
            Case {
                n: 1,
                k: 1,
                ..Default::default()
            },
            // n is a multiple of the tile size (16 for AVX 2 / FMA)
            Case {
                n: 16,
                k: 16,
                ..Default::default()
            },
            // n is not an exact multiple of the tile size
            Case {
                n: 20,
                k: 16,
                ..Default::default()
            },
            // n exceeds column block size
            Case {
                n: 300,
                k: 16,
                ..Default::default()
            },
            // k exceeds depth block size
            Case {
                n: 20,
                k: 300,
                ..Default::default()
            },
            // beta value = 0.
            Case {
                n: 20,
                k: 300,
                beta: 0.,
                ..Default::default()
            },
            // Non-standard beta value
            Case {
                n: 20,
                k: 300,
                beta: 0.5,
                ..Default::default()
            },
            // Non-standard alpha value
            Case {
                n: 20,
                k: 20,
                alpha: 0.5,
                ..Default::default()
            },
            // Test with bias
            Case {
                n: 20,
                k: 20,
                bias: Some(0.5),
                ..Default::default()
            },
            // Transposed matrix. Note both `n` and `k` are chosen to not be
            // an exact multiple of column or depth tile sizes.
            Case {
                n: 21,
                k: 21,
                b_strides: Strides::Transposed,
                ..Default::default()
            },
            // Transposed matrix with beta != 0
            Case {
                n: 21,
                k: 21,
                beta: 1.,
                b_strides: Strides::Transposed,
                ..Default::default()
            },
            // Transposed matrix with alpha != 1
            Case {
                n: 20,
                k: 20,
                alpha: 0.5,
                b_strides: Strides::Transposed,
                ..Default::default()
            },
            // Matrix with non-unit strides
            Case {
                n: 21,
                k: 21,
                b_strides: Strides::Other,
                ..Default::default()
            },
            // Matrix with non-unit strides, beta != 0
            Case {
                n: 21,
                k: 21,
                beta: 0.5,
                b_strides: Strides::Other,
                ..Default::default()
            },
            // Matrix with non-unit strides, alpha != 1
            Case {
                n: 21,
                k: 21,
                alpha: 0.5,
                b_strides: Strides::Other,
                ..Default::default()
            },
        ];

        let mut rng = XorShiftRng::new(1234);

        for Case {
            n,
            k,
            alpha,
            beta,
            bias,
            b_strides,
        } in cases
        {
            let a = Tensor::rand(&[1, k], &mut rng);
            let mut b = Tensor::rand(&[k, n], &mut rng);
            match b_strides {
                Strides::Contiguous => {}
                Strides::Transposed => {
                    b.transpose();
                }
                Strides::Other => {
                    b = Tensor::from_data_with_strides(&[k, n / 2], b.to_vec(), &[b.stride(0), 2])
                        .unwrap();
                }
            }

            let mut result = Tensor::zeros(&[1, b.size(1)]);
            let bias_array = bias.map(|b| [b]);

            run_gemm(
                &mut result,
                &a,
                &b,
                alpha,
                beta,
                bias_array
                    .as_ref()
                    .map(|b| BiasVector::Column(b.as_slice())),
                None,
            );

            let expected =
                reference_matmul_alpha_beta(&a, &b, alpha, beta).map(|x| x + bias.unwrap_or(0.));
            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    struct BenchCase {
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    }

    enum Format {
        Pretty,
        Csv,
    }

    fn run_gemm_bench(cases: &[BenchCase], format: Format) {
        println!("Testing kernel {}", GemmExecutor::new().kernel_name());

        // Print header
        match format {
            Format::Csv => {
                println!("m,n,k,duration_ms,gflops");
            }
            Format::Pretty => {}
        }

        for &BenchCase {
            m,
            n,
            k,
            transpose_b,
        } in cases
        {
            // Adjust number of iterations based on a target amount of work,
            // so that each case takes roughly the same amount of time, assuming
            // equal efficiency.
            let target_iters = 512;
            let target_ops: u64 = 512 * 512 * 512 * target_iters;
            let iters = target_ops / (m * n * k) as u64;

            // Cap the number of iterations, for cases where the equal-efficiency
            // assumption is untrue.
            let iters = iters.min(target_iters);

            let mut rng = XorShiftRng::new(1234);
            let mut result = Tensor::zeros(&[m, n]);
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = if transpose_b {
                let mut b = Tensor::rand(&[n, k], &mut rng);
                b.transpose();
                b
            } else {
                Tensor::rand(&[k, n], &mut rng)
            };

            let start = Instant::now();
            for _i in 0..iters {
                run_gemm(&mut result, &a, &b, 1., 0., None, None);
            }
            let duration = start.elapsed();

            // Calculate throughput. For comparison, the theoretical maximum
            // GFLOPS for a single core (`RAYON_NUM_THREADS=1`) can be computed
            // as:
            //
            //     frequency * simd_width * fma_throughput * fma_units
            //
            // Where:
            //  - `frequency` is the max frequency in Ghz
            //  - `simd_width` is the # of f32 values in a vector register
            //  - `fma_throughput` is the number of ops/cycle
            //  - `fma_units` is the number of execution units
            //
            // On an Intel Skylake CPU for example, `simd_width` will be
            // 8 (256-bit AVX 2 / 32-bit float), `fma_throughput` is 2,
            //   `fma_units` is 2. For a 3.4Ghz CPU this would give a max
            //   theoretical peak of 3.4 * 8 * 2 * 2 = 108.8 GFLOPS.

            let flops = (2 * m * n * k * iters as usize) as f32 / duration.as_secs_f32();
            let gflops = flops / (10f32).powi(9);
            let duration_ms = duration.as_secs_f64() * 1000.0;

            match format {
                Format::Pretty => {
                    println!(
                        "m {} n {} k {} iters {}. Duration {:.3}ms ({:.3}ms/iter). GFLOPS {:.1}",
                        m,
                        n,
                        k,
                        iters,
                        duration_ms,
                        duration_ms / iters as f64,
                        gflops,
                    );
                }
                Format::Csv => {
                    println!("{},{},{},{:.3},{:.1}", m, n, k, duration_ms, gflops);
                }
            }
        }
    }

    // Run with `cargo test --release bench_gemm_mix -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_gemm_mix() {
        type Case = BenchCase;

        let cases = [
            // Square output matrix
            Case {
                m: 512,
                n: 512,
                k: 512,
                transpose_b: false,
            },
            // Larger square output matrix
            Case {
                m: 1024,
                n: 1024,
                k: 1024,
                transpose_b: false,
            },
            // Wide output matrix
            Case {
                m: 128,
                n: 2048,
                k: 512,
                transpose_b: false,
            },
            // Tall output matrix
            Case {
                m: 2048,
                n: 128,
                k: 512,
                transpose_b: false,
            },
            // Vector-matrix. This is common in transformer decoders for example.
            Case {
                m: 1,
                n: 4096,
                k: 512,
                transpose_b: false,
            },
            Case {
                m: 1,
                n: 4096,
                k: 512,
                transpose_b: true,
            },
        ];

        run_gemm_bench(&cases, Format::Pretty);
    }

    #[test]
    #[ignore]
    fn bench_gemm_size_range() {
        let cases: Vec<_> = (1..512)
            .map(|size| BenchCase {
                m: size,
                n: size,
                k: size,
                transpose_b: false,
            })
            .collect();
        run_gemm_bench(&cases, Format::Csv);
    }

    // Like `bench_pack_a`, but this does include allocation costs, so is
    // relevant for ops which prepack inputs (eg. batched matmul).
    #[test]
    #[ignore]
    fn bench_prepack_a() {
        let gemm = GemmExecutor::new();
        let mut rng = XorShiftRng::new(1234);
        let m = 1024;
        let n = 1024;
        let iters = 1000;
        let a = NdTensor::rand([m, n], &mut rng);

        run_bench(
            10,
            Some(&format!("m {} n {} iters {}", m, n, iters)),
            || {
                for _i in 0..iters {
                    gemm.prepack_a(a.view());
                }
            },
        );
    }

    // TODO - Add a set of tests for use with Miri. These should exercise all
    // unsafe code, but be adjusted to run quickly.
}

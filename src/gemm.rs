//! Machine-learning oriented matrix multiplication functions.
//!
//! This module provides a subset of BLAS-like functions that are used by neural
//! network operators. The primary functionality is general matrix
//! multiplication (gemm) with ML-oriented additions, but there are also
//! operations like vector-scalar products.

use std::cell::RefCell;
use std::mem::{transmute, MaybeUninit};
use std::ops::{Add, Mul, Range};

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{Alloc, GlobalAlloc, Matrix, MatrixLayout, MatrixMut, NdTensorView};

use crate::iter_util::{range_chunks, MaybeParIter};
use crate::number::{cast_pod_mut_slice, cast_pod_slice, Identities};
use crate::tensor_pool::ExtractBuffer;

mod kernels;
mod packing;

use kernels::generic::GenericKernel;
use kernels::Kernel;

/// Left-hand or "A" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedAMatrix<T> {
    /// Sequence of packed row panels.
    data: Vec<T>,

    /// Number of elements in each row panel.
    panel_len: usize,

    /// Number of blocks that the matrix was divided into along the M dimension.
    row_blocks: usize,

    /// Number of rows in the unpacked matrix.
    rows: usize,

    /// Number of columns in the unpacked matrix.
    cols: usize,
}

impl<T> PackedAMatrix<T> {
    fn block(&self, row_block_idx: usize, depth_block_idx: usize) -> &[T] {
        let panel_idx = depth_block_idx * self.row_blocks + row_block_idx;
        let offset = panel_idx * self.panel_len;
        &self.data[offset..offset + self.panel_len]
    }
}

impl<T> ExtractBuffer for PackedAMatrix<T> {
    type Elem = T;

    fn extract_buffer(self) -> Option<Vec<T>> {
        Some(self.data)
    }
}

/// Right-hand or "B" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedBMatrix<T> {
    /// Sequence of packed column panels.
    data: Vec<T>,

    /// Number of elements in each column panel.
    panel_len: usize,

    /// Number of blocks that the matrix was divided into along the K dimension.
    depth_blocks: usize,

    /// Number of rows in the unpacked matrix.
    rows: usize,

    /// Number of columns in the unpacked matrix.
    cols: usize,
}

impl<T> PackedBMatrix<T> {
    fn block(&self, col_block_idx: usize, depth_block_idx: usize) -> &[T] {
        let panel_idx = col_block_idx * self.depth_blocks + depth_block_idx;
        let offset = panel_idx * self.panel_len;
        &self.data[offset..offset + self.panel_len]
    }
}

impl<T> ExtractBuffer for PackedBMatrix<T> {
    type Elem = T;

    fn extract_buffer(self) -> Option<Vec<T>> {
        Some(self.data)
    }
}

/// Left-hand or "A" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputA<'a, T> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a, T>),

    /// A matrix which has been pre-packed by [GemmExecutor::prepack_a].
    Packed(&'a PackedAMatrix<T>),
    // TODO - Support virtual "A" inputs, like `GemmInputB::Virtual`.
}

impl<'a, T> GemmInputA<'a, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.rows(),
            Self::Packed(pm) => pm.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.cols,
        }
    }
}

/// Trait implemented by GEMM input types.
pub trait GemmInT: Copy + Send + Sync {}
impl GemmInT for f32 {}

/// Trait implemented by GEMM output types.
pub trait GemmOutT:
    Copy + PartialEq + Send + Sync + Identities + Mul<Self, Output = Self> + Add<Self, Output = Self>
{
}
impl GemmOutT for f32 {}

/// A virtual matrix which has a known size, but may not actually be
/// materialized in memory. The GEMM implementation will call
/// [VirtualMatrix::pack_b] to pack blocks of this matrix into a buffer as it
/// needs them.
///
/// This is useful for operations such as im2col-based convolution, which
/// involve creating potentially large temporary matrices.
///
/// # Safety
///
/// Implementations of [`pack_b`](VirtualMatrix::pack_b) must initialize the
/// entire buffer passed to them.
pub unsafe trait VirtualMatrix<T>: Sync {
    /// Return the number of rows in the virtual matrix.
    fn rows(&self) -> usize;

    /// Return the number of columns in the virtual matrix.
    fn cols(&self) -> usize;

    /// Called by the GEMM implementation to pack a block of the virtual matrix
    /// into temporary buffer where it can be efficiently processed.
    ///
    /// Implementations must copy the subregion of the matrix specified by
    /// `rows` and `cols` into `out` with a specific memory layout:
    ///
    ///  - The columns specified by `cols` are divided into a sequence of
    ///    panels, each wth `panel_width` columns and `rows.len()` rows.
    ///    `panel_width` will depend on the matrix multiplication kernel used, but
    ///    will be a small value like 4 (SSE) or 16 (AVX2).
    ///  - Each panel should be written to `out` sequentially. Within each
    ///    panel, elements should be written out in row-major order.
    ///  - `cols.len()` may not be an even multiple of `panel_width`. In that
    ///    case the final panel should be zero-padded.
    fn pack_b(
        &self,
        out: &mut [MaybeUninit<T>],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    );
}

/// Right-hand or "B" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputB<'a, T> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a, T>),

    /// A matrix which has been pre-packed by [GemmExecutor::prepack_b].
    Packed(&'a PackedBMatrix<T>),

    /// A virtual matrix, blocks of which will be materialized on-demand
    /// during GEMM execution. See [VirtualMatrix].
    Virtual(&'a dyn VirtualMatrix<T>),
}

impl<'a, T> GemmInputB<'a, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.rows(),
            Self::Packed(pm) => pm.rows,
            Self::Virtual(vm) => vm.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.cols,
            Self::Virtual(vm) => vm.cols(),
        }
    }
}

/// Perform a General Matrix Multiplication ("gemm").
///
/// This computes `output = alpha * (a @ b) + beta * output` where `@` is
/// matrix multiplication.
#[allow(unused)]
pub fn gemm(
    out_data: &mut [f32],
    out_row_stride: usize,
    a: Matrix,
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    // This heap-allocates a new kernel on each call. That's OK because this
    // is very cheap relative to the large matmuls we expect to be doing, but
    // would be good to avoid for small inputs.
    GemmExecutor::new().gemm(
        out_data,
        out_row_stride,
        GemmInputA::Unpacked(a),
        GemmInputB::Unpacked(b),
        alpha,
        beta,
    );
}

/// Executes matrix multiplication operations.
///
/// For simple use cases, the standalone [gemm] function can be used.
/// GemmExecutor provides a more advanced API that enables features such as
/// performing matrix multiplications with pre-packed inputs.
///
/// ## Prepacking
///
/// Prepacking is useful when an input will be reused in multiple GEMM
/// operations. In this case the work to pack (re-layout) the input for maximum
/// computational efficiency, which is normally does internally on each call,
/// can be done just once for the reused input.
pub struct GemmExecutor {
    kernel: Box<dyn Kernel<f32, f32, f32>>,
    kernel_type: KernelType,
}

/// Arguments for [GemmExecutor::with_kernel] specifying which kernel to use.
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

impl GemmExecutor {
    /// Create a [GemmExecutor] using the preferred kernel for the current system.
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

    /// Return the name of the kernel that this executor is using.
    #[allow(dead_code)]
    pub fn kernel_name(&self) -> &str {
        self.kernel.name()
    }

    /// Return the type of kernel being used.
    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    /// Create a [GemmExecutor] using the given kernel. Returns `None` if the
    /// kernel is not supported.
    #[allow(dead_code)] // Currently only used in tests
    pub fn with_kernel(hint: KernelType) -> Option<GemmExecutor> {
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
    fn with_generic_kernel() -> GemmExecutor {
        let kernel = GenericKernel::new().unwrap();
        GemmExecutor {
            kernel: Box::new(kernel),
            kernel_type: KernelType::Generic,
        }
    }

    /// Prepack a matrix for use as the left-hand or "A" input.
    #[allow(unused)]
    pub fn prepack_a(&self, a: Matrix) -> PackedAMatrix<f32> {
        self.prepack_a_in(GlobalAlloc::new(), a)
    }

    /// Variant of [`prepack_a`](GemmExecutor::prepack_a) which takes an
    /// allocator.
    pub fn prepack_a_in<A: Alloc>(&self, alloc: A, a: Matrix) -> PackedAMatrix<f32> {
        let kc = depth_block_size(a.cols());
        let mr = self.kernel.mr();
        let mc = row_block_size(a.rows(), mr);
        let panel_len = kc * mc;
        let row_blocks = a.rows().div_ceil(mc);
        let depth_blocks = a.cols().div_ceil(kc);

        let packed_len = depth_blocks * row_blocks * panel_len;
        let mut data = alloc.alloc(packed_len);

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = data.spare_capacity_mut()[..packed_len].chunks_exact_mut(panel_len);
        let mut n_init = 0;
        for depth_range in range_chunks(0..a.cols(), kc) {
            for row_range in range_chunks(0..a.rows(), mc) {
                let out_panel = out_panels.next().unwrap();
                let used_size = row_range.len().next_multiple_of(mr) * depth_range.len();
                let (used, unused) = out_panel.split_at_mut(used_size);

                self.kernel
                    .pack_a_block(used, a, row_range, depth_range.clone());

                unused.fill(MaybeUninit::new(0.));
                n_init += out_panel.len();
            }
        }

        // Safety: We used `pack_a_block` to initialize `packed_len` elements.
        assert!(n_init == packed_len);
        unsafe {
            data.set_len(packed_len);
        }

        PackedAMatrix {
            data,
            rows: a.rows(),
            cols: a.cols(),
            panel_len,
            row_blocks,
        }
    }

    /// Return the panel width used when packing the "B" matrix.
    ///
    /// This information is useful for implementations of [VirtualMatrix].
    pub fn b_panel_width(&self) -> usize {
        self.kernel.nr()
    }

    /// Prepack a matrix for use as the right-hand or "B" matrix input.
    #[allow(unused)]
    pub fn prepack_b(&self, b: Matrix) -> PackedBMatrix<f32> {
        self.prepack_b_in(GlobalAlloc::new(), b)
    }

    /// Variant of [`prepack_b`](GemmExecutor::prepack_b) which takes an
    /// allocator.
    pub fn prepack_b_in<A: Alloc>(&self, alloc: A, b: Matrix) -> PackedBMatrix<f32> {
        let nr = self.kernel.nr();
        let nc = col_block_size(b.cols(), nr);
        let kc = depth_block_size(b.rows());
        let panel_len = nc * kc;
        let depth_blocks = b.rows().div_ceil(kc);
        let col_blocks = b.cols().div_ceil(nc);

        let packed_len = col_blocks * depth_blocks * panel_len;
        let mut out = alloc.alloc(packed_len);

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = out.spare_capacity_mut()[..packed_len].chunks_exact_mut(panel_len);
        let mut n_init = 0;
        for col_range in range_chunks(0..b.cols(), nc) {
            for depth_range in range_chunks(0..b.rows(), kc) {
                let out_panel = out_panels.next().unwrap();
                let used_size = col_range.len().next_multiple_of(nr) * depth_range.len();
                let (used, unused) = out_panel.split_at_mut(used_size);

                self.kernel
                    .pack_b_block(used, b, depth_range, col_range.clone());

                unused.fill(MaybeUninit::new(0.));
                n_init += out_panel.len();
            }
        }

        // Safety: We used `pack_b_block` to initialize `packed_len` elements.
        assert!(n_init == packed_len);
        unsafe {
            out.set_len(packed_len);
        }

        PackedBMatrix {
            data: out,
            rows: b.rows(),
            cols: b.cols(),
            depth_blocks,
            panel_len,
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
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA<f32>,
        b: GemmInputB<f32>,
        alpha: f32,
        beta: f32,
    ) {
        gemm_impl(
            &*self.kernel,
            out_data,
            out_row_stride,
            a,
            b,
            alpha,
            beta,
            None,
        )
    }

    /// Perform a General Matrix Multiplication ("gemm").
    ///
    /// This is the same as [GemmExecutor::gemm] but takes an uninitialized
    /// output slice. The `beta` value is implicitly set to zero.
    pub fn gemm_uninit(
        &self,
        out_data: &mut [MaybeUninit<f32>],
        out_row_stride: usize,
        a: GemmInputA<f32>,
        b: GemmInputB<f32>,
        alpha: f32,
    ) {
        self.gemm_uninit_bias(out_data, out_row_stride, a, b, alpha, None);
    }

    /// Perform a matrix multiplication with fused bias vector addition.
    ///
    /// This computes `output = alpha * (a @ b) + beta * output + bias` where
    /// `@` is matrix multiplication.
    ///
    /// If `bias` is present, it is treated as a column vector whose length
    /// must match the rows of `a`.
    #[allow(unused)]
    pub fn gemm_bias(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA<f32>,
        b: GemmInputB<f32>,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
    ) {
        gemm_impl(
            &*self.kernel,
            out_data,
            out_row_stride,
            a,
            b,
            alpha,
            beta,
            bias,
        )
    }

    /// Perform a matrix multiplication with fused bias vector addition.
    ///
    /// This computes `output = alpha * (a @ b) + bias` where
    /// `@` is matrix multiplication.
    ///
    /// If `bias` is present, it is treated as a column vector whose length
    /// must match the rows of `a`.
    pub fn gemm_uninit_bias(
        &self,
        out_data: &mut [MaybeUninit<f32>],
        out_row_stride: usize,
        a: GemmInputA<f32>,
        b: GemmInputB<f32>,
        alpha: f32,
        bias: Option<&[f32]>,
    ) {
        gemm_impl(
            &*self.kernel,
            // Safety: When beta is zero, we initialize all output elements
            // and ignore existing values.
            unsafe { transmute::<&mut [MaybeUninit<f32>], &mut [f32]>(out_data) },
            out_row_stride,
            a,
            b,
            alpha,
            0., /* beta */
            bias,
        )
    }
}

/// Return the block size for the K / depth dimension of a GEMM operation.
fn depth_block_size(a_cols: usize) -> usize {
    256.min(a_cols)
}

/// Return the block size for the N / column dimension of a GEMM operation.
///
/// The result is always a multiple of `nr`.
fn col_block_size(b_cols: usize, nr: usize) -> usize {
    // In the BLIS library which formulated the GEMM algorithm we use,
    // the column block size is chosen so that blocks fit in the L3 cache
    // (see https://dl.acm.org/doi/pdf/10.1145/2925987, p 12:7).
    //
    // In this library that constraint provides an upper bound, but the value
    // is also adjusted to control parallelism.
    let parallelism = rayon::current_num_threads();
    let lower_bound = 128.min(b_cols);
    let unrounded = (b_cols / parallelism).max(lower_bound).min(1024);
    unrounded.next_multiple_of(nr)
}

/// Return the block size for the M / row dimension of a GEMM operation.
///
/// The result is always a multiple of `mr`.
fn row_block_size(a_rows: usize, mr: usize) -> usize {
    64.min(a_rows).next_multiple_of(mr)
}

/// A single tile of the output matrix.
struct OutputTile<T> {
    /// Pointer to first element in this tile.
    ptr: *mut T,

    /// Stride between rows of this tile. Note the column stride is always 1.
    row_stride: usize,

    /// Number of rows in this tile. Will be <= the [Kernel]'s `MR` constant.
    used_rows: usize,

    /// Number of columns in this tile. Will be <= the [Kernel]'s `NR` constant.
    used_cols: usize,
}

/// Wrapper around the GEMM output matrix which divides it into a grid of tiles.
/// This can be shared across threads, but each individual tile must only be
/// operated on by one thread at a time.
struct OutputTiles<T> {
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
}

/// Safety: Caller must ensure they do not operate on overlapping tiles
/// concurrently.
unsafe impl<T> Sync for OutputTiles<T> {}

impl<T> OutputTiles<T> {
    /// Expose `data` as a grid of tiles, each with a maximum size of
    /// `tile_rows` * `tile_cols`.
    fn new(mut data: MatrixMut<T>, tile_rows: usize, tile_cols: usize) -> OutputTiles<T> {
        OutputTiles {
            data: data.data_mut().unwrap().as_mut_ptr(),
            rows: data.rows(),
            cols: data.cols(),
            row_stride: data.stride(0),
            tile_rows,
            tile_cols,
            n_row_tiles: data.rows().div_ceil(tile_rows),
            n_col_tiles: data.cols().div_ceil(tile_cols),
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
    mut output_mat: MatrixMut<OutT>,
    alpha: f32,
    beta: OutT,
    bias: Option<OutT>,
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
                let b_block = b.slice::<2, _>((k_block, col_block.clone()));
                kernel.gemv_kernel(out_chunk, a_block, b_block, alpha, effective_beta);

                // Reset `beta` so that subsequent updates for each column
                // accumulate into the first update.
                effective_beta = OutT::one();
            }

            if let Some(bias) = bias {
                for x in out_chunk {
                    *x = *x + bias;
                }
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
    out_data: &mut [OutT],
    out_row_stride: usize,
    a: GemmInputA<LhsT>,
    b: GemmInputB<RhsT>,
    alpha: f32,
    beta: OutT,
    bias: Option<&[OutT]>,
) {
    assert!(
        a.cols() == b.rows(),
        "Columns of matrix `a` must match rows of matrix `b`"
    );
    assert!(
        bias.map(|b| b.len()).unwrap_or(a.rows()) == a.rows(),
        "Bias vector length must match rows of matrix `a`"
    );

    // Handle case where output is empty.
    if a.rows() == 0 || b.cols() == 0 {
        return;
    }

    // Handle case where depth is zero. We still need to initialize the output
    // in this case.
    if a.cols() == 0 {
        for x in out_data {
            let tmp = if beta == OutT::zero() {
                OutT::zero()
            } else {
                *x
            };
            *x = beta * tmp;
        }
        return;
    }

    // Construct a Matrix from the implied dimensions, to validate the slice length.
    let mut output_mat = MatrixMut::<OutT>::from_data_with_strides(
        [a.rows(), b.cols()],
        out_data,
        [out_row_stride, 1],
    )
    .expect("Output buffer should be large enough");

    // Use optimized path for vector-matrix products.
    if let (1, GemmInputA::Unpacked(a), GemmInputB::Unpacked(b)) = (a.rows(), a, b) {
        gemv(
            kernel,
            a.slice::<1, _>(0),
            b,
            output_mat.view_mut(),
            alpha,
            beta,
            // nb. We checked above that, if present, the bias length matches `a.rows()`.
            bias.map(|b| b[0]),
        );
        return;
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

    // Buffers for packed blocks of the matrix.
    //
    // These use `u64` rather than LhsT / RhsT because statics cannot be generic.
    // `u64` is used to ensure alignment is a m
    thread_local!(static PACKED_A: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) });
    thread_local!(static PACKED_B: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) });
    assert!(align_of::<LhsT>() <= align_of::<u64>());
    assert!(align_of::<RhsT>() <= align_of::<u64>());

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

            // Loop over depth blocks. This is not parallelized because output
            // tiles are shared across iterations.
            for (depth_idx, depth_range) in range_chunks(0..a.cols(), kc).enumerate() {
                // Borrowed packing buffer for current thread. Returned after
                // the GEMM block is computed.
                let mut thread_local_packed_b: Option<Vec<u64>> = None;
                let panel_length = depth_range.len();
                let packed_b_size = (col_end - col_start).next_multiple_of(nr) * panel_length;

                let packed_b = match b {
                    GemmInputB::Unpacked(_) | GemmInputB::Virtual(_) => PACKED_B.with(|cell| {
                        let mut packed_b = cell.take();
                        packed_b.clear();
                        packed_b
                            .reserve(packed_b_size.div_ceil(size_of::<u64>() / size_of::<RhsT>()));

                        let packed_b_slice =
                            cast_pod_mut_slice(packed_b.spare_capacity_mut()).unwrap();
                        let packed_b_slice = &mut packed_b_slice[..packed_b_size];

                        match b {
                            GemmInputB::Unpacked(b) => kernel.pack_b_block(
                                packed_b_slice,
                                b,
                                depth_range.clone(),
                                col_start..col_end,
                            ),
                            GemmInputB::Virtual(vm) => vm.pack_b(
                                packed_b_slice,
                                kernel.nr(),
                                depth_range.clone(),
                                col_start..col_end,
                            ),
                            GemmInputB::Packed(_) => unreachable!(),
                        }

                        // Safety: The packing call initialized `packed_b_size` elements.
                        unsafe {
                            packed_b.set_len(packed_b_size);
                        }
                        thread_local_packed_b = Some(packed_b);
                        cast_pod_slice::<_, RhsT>(thread_local_packed_b.as_deref().unwrap())
                            .unwrap()
                    }),
                    GemmInputB::Packed(pm) => pm.block(col_idx, depth_idx),
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
                        let packed_a_size =
                            (row_end - row_start).next_multiple_of(mr) * depth_range.len();

                        // Borrowed packing buffer for current thread. Returned after
                        // the GEMM block is computed.
                        let mut thread_local_packed_a: Option<Vec<u64>> = None;

                        let packed_a = match a {
                            GemmInputA::Unpacked(a) => PACKED_A.with(|cell| {
                                let mut packed_a = cell.take();
                                packed_a.clear();
                                packed_a.reserve(
                                    packed_a_size.div_ceil(size_of::<u64>() / size_of::<LhsT>()),
                                );

                                let packed_a_block = cast_pod_mut_slice::<_, MaybeUninit<LhsT>>(
                                    packed_a.spare_capacity_mut(),
                                )
                                .unwrap();
                                kernel.pack_a_block(
                                    &mut packed_a_block[..packed_a_size],
                                    a,
                                    row_start..row_end,
                                    depth_range.clone(),
                                );
                                // Safety: `pack_a_block` will have initialized
                                // `packed_a_size` elements.
                                unsafe {
                                    packed_a.set_len(packed_a_size);
                                }
                                thread_local_packed_a = Some(packed_a);
                                cast_pod_slice::<_, LhsT>(thread_local_packed_a.as_deref().unwrap())
                                    .unwrap()
                            }),
                            GemmInputA::Packed(pm) => pm.block(row_idx, depth_idx),
                        };

                        gemm_block(
                            kernel,
                            &output_tiles,
                            col_start / nr..col_end.div_ceil(nr),
                            row_start / mr..row_end.div_ceil(mr),
                            depth_range.start == 0,
                            packed_a,
                            packed_b,
                            panel_length,
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
}

/// Process a single block (ie. a slice along each of the M/N/K dimensions) of a
/// matrix multiplication.
///
/// `col_tiles` and `row_tiles` specifies the range of output tiles to update,
/// `packed_a` and `packed_b` are the corresponding packed inputs. `panel_length`
/// is the size of panels along the depth/K dimension.
///
/// `is_first` indicates whether this is the first write to the output tiles
/// in this block during the current GEMM operation.
fn gemm_block<LhsT, RhsT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    output: &OutputTiles<OutT>,
    col_tiles: Range<usize>,
    row_tiles: Range<usize>,
    first_update: bool,
    packed_a: &[LhsT],
    packed_b: &[RhsT],
    panel_length: usize,
    alpha: f32,
    beta: OutT,
    bias: Option<&[OutT]>,
) {
    // Maximum tile size of all supported kernels.
    const MAX_MR: usize = 8;
    const MAX_NR: usize = 32;

    let (mr, nr) = (kernel.mr(), kernel.nr());

    assert!(nr <= MAX_NR && mr <= MAX_MR);

    let b_panel_size = panel_length * nr;
    let a_panel_size = mr * panel_length;

    // Loop over column tiles.
    //
    // TODO - This should be parallel, but threading overhead needs to be reduced.
    col_tiles
        .enumerate()
        .for_each(|(block_col_tile, col_tile)| {
            let b_panel_offset = block_col_tile * b_panel_size;
            let b_panel = &packed_b[b_panel_offset..b_panel_offset + b_panel_size];

            // Loop over row tiles.
            for (block_row_tile, row_tile) in row_tiles.clone().enumerate() {
                let a_panel_offset = block_row_tile * a_panel_size;
                let a_panel = &packed_a[a_panel_offset..a_panel_offset + a_panel_size];

                // Safety:
                //  - The loops in this function and its caller are set up so that
                //    every output tile is processed by one thread at a time.
                let out_tile = unsafe { output.tile(row_tile, col_tile) };

                if out_tile.used_rows == mr && out_tile.used_cols == nr {
                    // Safety:
                    //  - Tile size is MR * NR
                    unsafe {
                        kernel.kernel(
                            out_tile.ptr,
                            out_tile.row_stride,
                            a_panel,
                            b_panel,
                            panel_length,
                            alpha,
                            beta,
                        );
                    }
                } else {
                    // If this is not a full size tile, run the kernel on a
                    // temporary buffer that is the size of a full tile, then
                    // copy the results back to the output. This allows the same
                    // kernel implementation to be used whether the tile is
                    // full-sized or not.
                    let mut tmp_out_tile = [MaybeUninit::<OutT>::uninit(); MAX_MR * MAX_NR];

                    // Safety:
                    //  - Tile size is <= MAX_MR * MAX_NR
                    unsafe {
                        kernel.kernel(
                            transmute::<*mut MaybeUninit<OutT>, *mut OutT>(
                                tmp_out_tile.as_mut_ptr(),
                            ),
                            nr,
                            a_panel,
                            b_panel,
                            panel_length,
                            alpha,
                            OutT::zero(), // Multiplication with `beta` is handled below.
                        );
                    }

                    for i in 0..out_tile.used_rows {
                        for j in 0..out_tile.used_cols {
                            // Safety: Row and column indices are < used rows /
                            // cols in this tile.
                            unsafe {
                                let out_el = out_tile.ptr.add(out_tile.row_stride * i + j);
                                let tmp = if beta == OutT::zero() {
                                    OutT::zero()
                                } else {
                                    *out_el
                                };
                                *out_el = beta * tmp
                                    + tmp_out_tile.get_unchecked(i * nr + j).assume_init();
                            }
                        }
                    }
                }

                // Add bias vector on first write to an output tile.
                if let (Some(bias), true) = (bias, first_update) {
                    for row in 0..out_tile.used_rows {
                        for col in 0..out_tile.used_cols {
                            // Safety:
                            //  - Row and column indices are valid for current tile
                            //  - Bias length was checked at start of `gemm_impl`
                            unsafe {
                                let out_el = out_tile.ptr.add(row * out_tile.row_stride + col);
                                *out_el = *out_el + *bias.get_unchecked(row_tile * mr + row);
                            }
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::mem::MaybeUninit;
    use std::ops::Range;
    use std::time::Instant;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor, Tensor};

    use super::{gemm, GemmExecutor, GemmInputA, GemmInputB, KernelType, VirtualMatrix};

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
        bias: Option<&[f32]>,
        kernel: Option<KernelType>,
    ) {
        let out_row_stride = output.stride(0);
        let gemm = if let Some(kernel) = kernel {
            GemmExecutor::with_kernel(kernel).expect("kernel not available")
        } else {
            GemmExecutor::new()
        };

        gemm.gemm_bias(
            output.data_mut().expect("expected contiguous input"),
            out_row_stride,
            GemmInputA::Unpacked(a.nd_view()),
            GemmInputB::Unpacked(b.nd_view()),
            alpha,
            beta,
            bias,
        );
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
        bias: Option<&[f32]>,
    ) {
        let [a_rows, a_cols]: [usize; 2] = a.shape().try_into().expect("input should be a matrix");
        let [_b_rows, b_cols]: [usize; 2] = b.shape().try_into().expect("input should be a matrix");

        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut accum = 0.0;
                for k in 0..a_cols {
                    accum += a[[r, k]] * b[[k, c]];
                }
                output[[r, c]] =
                    alpha * accum + beta * output[[r, c]] + bias.map(|b| b[r]).unwrap_or(0.);
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
    #[should_panic(expected = "Output buffer should be large enough")]
    fn test_gemm_panics_if_output_is_too_short() {
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![5., 6., 7., 8.]);

        let mut output = vec![1., 2.];

        gemm(
            &mut output,
            2,
            a.nd_view(),
            b.nd_view(),
            1., /* alpha */
            1., /* beta */
        );
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

        let a = Tensor::rand(&[10, 5], &mut rng);
        let b = Tensor::rand(&[5, 15], &mut rng);
        let bias: Vec<f32> = (0..a.shape()[0]).map(|b| b as f32).collect();

        let mut result = Tensor::zeros(&[10, 15]);
        let mut expected = result.clone();

        for kernel in [None, Some(KernelType::Generic)] {
            run_gemm(&mut result, &a, &b, 1., 0., Some(&bias), kernel);
            reference_gemm(&mut expected, &a, &b, 1., 0., Some(&bias));
        }

        expect_equal(&result, &expected)?;

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
                1.,
                1.,
            );

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
                1.,
                1.,
            );

            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_gemm_virtual() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        struct Packer<'a> {
            tensor: Matrix<'a, f32>,
        }

        // Safety: `pack_b` initializes the entire buffer.
        unsafe impl<'a> VirtualMatrix<f32> for Packer<'a> {
            fn rows(&self) -> usize {
                self.tensor.rows()
            }

            fn cols(&self) -> usize {
                self.tensor.cols()
            }

            fn pack_b(
                &self,
                out: &mut [MaybeUninit<f32>],
                panel_width: usize,
                rows: Range<usize>,
                cols: Range<usize>,
            ) {
                let out_cols = cols.len().next_multiple_of(panel_width);
                let mut out_iter = out.iter_mut();

                for panel_start_col in (0..out_cols).step_by(panel_width) {
                    for row in rows.clone() {
                        for panel_col in 0..panel_width {
                            let col = panel_start_col + panel_col;
                            out_iter.next().unwrap().write(
                                self.tensor
                                    .get([row, cols.start + col])
                                    .copied()
                                    .unwrap_or(0.),
                            );
                        }
                    }
                }
            }
        }

        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            // Single block along all dimensions.
            Case {
                m: 10,
                k: 20,
                n: 30,
            },
            // Multiple depth blocks.
            Case {
                m: 10,
                k: DEPTH_BLOCK_SIZE + 50,
                n: 20,
            },
            // Multiple column blocks
            Case {
                m: 10,
                k: 10,
                n: COL_BLOCK_SIZE + 50,
            },
        ];

        for case in cases {
            let mut result = Tensor::zeros(&[case.m, case.n]);
            let result_row_stride = result.stride(0);
            let mut expected = result.clone();

            let a = Tensor::rand(&[case.m, case.k], &mut rng);
            let b = Tensor::rand(&[case.k, case.n], &mut rng);
            let gemm = GemmExecutor::new();

            let packer = Packer {
                tensor: b.nd_view(),
            };

            gemm.gemm(
                result.data_mut().unwrap(),
                result_row_stride,
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Virtual(&packer),
                1.,
                1.,
            );
            reference_gemm(&mut expected, &a, &b, 1., 1., None);

            expect_equal(&result, &expected)?;
        }

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
                bias_array.as_ref().map(|b| b.as_slice()),
                None,
            );

            let expected =
                reference_matmul_alpha_beta(&a, &b, alpha, beta).map(|x| x + bias.unwrap_or(0.));
            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    // Run with `cargo test --release bench_gemm -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_gemm() {
        struct Case {
            m: usize,
            n: usize,
            k: usize,
            transpose_b: bool,
        }

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

        println!("Testing kernel {}", GemmExecutor::new().kernel_name());

        for case in cases {
            let Case {
                m,
                n,
                k,
                transpose_b,
            } = case;

            // Adjust number of iterations based on a target amount of work,
            // so that each case takes roughly the same amount of time, assuming
            // equal efficiency.
            let target_ops: u64 = 512 * 512 * 512 * 1000;
            let iters = target_ops / (m * n * k) as u64;

            // Cap the number of iterations, for cases where the equal-efficiency
            // assumption is untrue.
            let iters = iters.min(1000);

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

            println!(
                "m {} n {} k {} iters {}. Duration {}ms ({}ms/iter). GFLOPS {}",
                m,
                n,
                k,
                iters,
                duration_ms,
                duration_ms / iters as f64,
                gflops,
            );
        }
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

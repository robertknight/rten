//! Machine-learning oriented matrix multiplication functions.
//!
//! This module provides a subset of BLAS-like functions that are used by neural
//! network operators. The primary functionality is general matrix
//! multiplication (gemm) with ML-oriented additions, but there are also
//! operations like vector-scalar products.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Range;

use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{Matrix, MatrixLayout, MatrixMut, NdTensorView};

use crate::iter_util::{range_chunks, unroll_loop, MaybeParIter};

mod kernels;
mod packing;

use kernels::{BaseKernel, Kernel};

/// Return `a / b`, rounding up if `b` does not evenly divide `a`.
pub fn div_ceil(a: usize, b: usize) -> usize {
    if b == 1 {
        // Fast path
        return a;
    }
    let rounding = usize::from(a % b != 0);
    a / b + rounding
}

/// Compute `dest += src * scale`, also known as a vector-scalar product or
/// "axpy" operation.
///
/// `dest_stride` and `src_stride` specifies the strides to use when iterating
/// over `dest` and `src` respectively. The lengths of `dest` and `src` must
/// match after accounting for their respective strides.
#[inline]
pub fn add_scaled_vector(
    dest: &mut [f32],
    src: &[f32],
    dest_stride: usize,
    src_stride: usize,
    scale: f32,
) {
    // Fast path for non-strided case. We write a trivial loop and leave the
    // compiler to optimize it.
    if src_stride == 1 && dest_stride == 1 {
        assert!(
            src.len() == dest.len(),
            "src and dest vector sizes do not match"
        );
        for i in 0..dest.len() {
            dest[i] += src[i] * scale;
        }
        return;
    }

    let src_els = src.len().div_ceil(src_stride);
    let dest_els = dest.len().div_ceil(dest_stride);
    assert!(
        src_els == dest_els,
        "src and dest vector sizes do not match"
    );

    unroll_loop!(0..src_els, i, 4, {
        unsafe {
            *dest.get_unchecked_mut(i * dest_stride) += *src.get_unchecked(i * src_stride) * scale;
        }
    });
}

/// Return the smallest multiple of `factor` that is >= `val`.
pub fn round_up(val: usize, factor: usize) -> usize {
    let rem = val % factor;
    if rem == 0 {
        val
    } else {
        (val + factor) - rem
    }
}

/// Left-hand or "A" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedAMatrix<'a> {
    /// Sequence of packed row panels.
    data: Cow<'a, [f32]>,

    /// Number of elements in each row panel.
    panel_len: usize,

    /// Number of blocks that the matrix was divided into along the M dimension.
    row_blocks: usize,

    /// Number of rows in the unpacked matrix.
    rows: usize,

    /// Number of columns in the unpacked matrix.
    cols: usize,
}

impl<'a> PackedAMatrix<'a> {
    fn block(&self, row_block_idx: usize, depth_block_idx: usize) -> &[f32] {
        let panel_idx = depth_block_idx * self.row_blocks + row_block_idx;
        let offset = panel_idx * self.panel_len;
        &self.data[offset..offset + self.panel_len]
    }
}

/// Right-hand or "B" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedBMatrix {
    /// Sequence of packed column panels.
    data: Vec<f32>,

    /// Number of elements in each column panel.
    panel_len: usize,

    /// Number of blocks that the matrix was divided into along the K dimension.
    depth_blocks: usize,

    /// Number of rows in the unpacked matrix.
    rows: usize,

    /// Number of columns in the unpacked matrix.
    cols: usize,
}

impl PackedBMatrix {
    fn block(&self, col_block_idx: usize, depth_block_idx: usize) -> &[f32] {
        let panel_idx = col_block_idx * self.depth_blocks + depth_block_idx;
        let offset = panel_idx * self.panel_len;
        &self.data[offset..offset + self.panel_len]
    }
}

/// Left-hand or "A" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputA<'a> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a>),

    /// A matrix which has been pre-packed by [GemmExecutor::prepack_a].
    Packed(&'a PackedAMatrix<'a>),
    // TODO - Support virtual "A" inputs, like `GemmInputB::Virtual`.
}

impl<'a> GemmInputA<'a> {
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

/// A virtual matrix which has a known size, but may not actually be
/// materialized in memory. The GEMM implementation will call
/// [VirtualMatrix::pack_b] to pack blocks of this matrix into a buffer as it
/// needs them.
///
/// This is useful for operations such as im2col-based convolution, which
/// involve creating potentially large temporary matrices.
pub trait VirtualMatrix: Sync {
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
    fn pack_b(&self, out: &mut [f32], panel_width: usize, rows: Range<usize>, cols: Range<usize>);
}

/// Right-hand or "B" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputB<'a> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a>),

    /// A matrix which has been pre-packed by [GemmExecutor::prepack_b].
    Packed(&'a PackedBMatrix),

    /// A virtual matrix, blocks of which will be materialized on-demand
    /// during GEMM execution. See [VirtualMatrix].
    Virtual(&'a dyn VirtualMatrix),
}

impl<'a> GemmInputB<'a> {
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
    kernel: Box<dyn Kernel>,
}

/// Arguments for [GemmExecutor::with_kernel] specifying which kernel to use.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Currently only used in tests
pub enum KernelHint {
    /// Use the preferred kernel for the current platform. Always available.
    Auto,

    /// Use the fallback/base kernel. Always available.
    Base,

    /// Use the AVX 2 + FMA kernel. Intel x64 only.
    Fma,

    /// Use the AVX 512 kernel. Intel x64 only.
    Avx512,

    /// Use the ARM NEON kernel. ARM 64 only.
    ArmNeon,

    /// Use the WASM SIMD kernel. WASM only.
    Wasm,
}

impl GemmExecutor {
    /// Create a [GemmExecutor] using the preferred kernel for the current system.
    pub fn new() -> GemmExecutor {
        #[cfg(feature = "avx512")]
        #[cfg(target_arch = "x86_64")]
        if let Some(gemm) = Self::with_kernel(KernelHint::Avx512) {
            return gemm;
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(gemm) = Self::with_kernel(KernelHint::Fma) {
            return gemm;
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(gemm) = Self::with_kernel(KernelHint::ArmNeon) {
            return gemm;
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(gemm) = Self::with_kernel(KernelHint::Wasm) {
            return gemm;
        }
        Self::with_base_kernel()
    }

    /// Return the name of the kernel that this executor is using.
    #[allow(dead_code)]
    pub fn kernel_name(&self) -> &str {
        self.kernel.name()
    }

    /// Create a [GemmExecutor] using the given kernel. Returns `None` if the
    /// kernel is not supported.
    #[allow(dead_code)] // Currently only used in tests
    pub fn with_kernel(kernel: KernelHint) -> Option<GemmExecutor> {
        fn make_kernel<K: Kernel + 'static>() -> Option<GemmExecutor> {
            K::new().map(|kernel| GemmExecutor {
                kernel: Box::new(kernel),
            })
        }

        match kernel {
            KernelHint::Auto => Some(Self::new()),
            #[cfg(feature = "avx512")]
            #[cfg(target_arch = "x86_64")]
            KernelHint::Avx512 => make_kernel::<kernels::x86_64::Avx512Kernel>(),
            #[cfg(target_arch = "x86_64")]
            KernelHint::Fma => make_kernel::<kernels::x86_64::FmaKernel>(),
            #[cfg(target_arch = "aarch64")]
            KernelHint::ArmNeon => make_kernel::<kernels::aarch64::ArmNeonKernel>(),
            #[cfg(target_arch = "wasm32")]
            KernelHint::Wasm => make_kernel::<kernels::wasm::WasmKernel>(),
            KernelHint::Base => Some(Self::with_base_kernel()),
            // Fail by default if requested kernel is never supported on
            // current platform (eg. requesting Arm Neon on x64).
            _ => None,
        }
    }

    /// Construct a GemmExecutor that uses the generic kernel.
    fn with_base_kernel() -> GemmExecutor {
        let kernel = BaseKernel::new().unwrap();
        GemmExecutor {
            kernel: Box::new(kernel),
        }
    }

    /// Prepack a matrix for use as the left-hand or "A" input.
    pub fn prepack_a(&self, a: Matrix) -> PackedAMatrix<'static> {
        let kc = depth_block_size(a.cols());
        let mc = row_block_size(a.rows(), self.kernel.mr());
        let panel_len = kc * mc;
        let row_blocks = div_ceil(a.rows(), mc);
        let depth_blocks = div_ceil(a.cols(), kc);

        let packed_len = depth_blocks * row_blocks * panel_len;
        let mut data = Vec::with_capacity(packed_len);

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = data.spare_capacity_mut()[..packed_len].chunks_exact_mut(panel_len);
        let mut n_init = 0;
        for depth_range in range_chunks(0..a.cols(), kc) {
            for row_range in range_chunks(0..a.rows(), mc) {
                let out_panel = out_panels.next().unwrap();
                self.kernel
                    .pack_a_block(out_panel, a, row_range, depth_range.clone());
                n_init += out_panel.len();
            }
        }

        // Safety: We used `pack_a_block` to initialize `packed_len` elements.
        assert!(n_init == packed_len);
        unsafe {
            data.set_len(packed_len);
        }

        PackedAMatrix {
            data: Cow::Owned(data),
            rows: a.rows(),
            cols: a.cols(),
            panel_len,
            row_blocks,
        }
    }

    /// Prepack a matrix for use as the right-hand or "B" matrix input.
    pub fn prepack_b(&self, b: Matrix, a_cols: usize) -> PackedBMatrix {
        let nc = col_block_size(b.cols(), self.kernel.nr());
        let kc = depth_block_size(a_cols);
        let panel_len = nc * kc;
        let depth_blocks = div_ceil(a_cols, kc);
        let col_blocks = div_ceil(b.cols(), nc);

        let packed_len = col_blocks * depth_blocks * panel_len;
        let mut out = Vec::with_capacity(packed_len);

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = out.spare_capacity_mut()[..packed_len].chunks_exact_mut(panel_len);
        let mut n_init = 0;
        for col_range in range_chunks(0..b.cols(), nc) {
            for depth_range in range_chunks(0..a_cols, kc) {
                let out_panel = out_panels.next().unwrap();
                self.kernel
                    .pack_b_block(out_panel, b, depth_range, col_range.clone());
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
        a: GemmInputA,
        b: GemmInputB,
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

    /// Perform a matrix multiplication with fused bias vector addition.
    ///
    /// This computes `output = alpha * (a @ b) + beta * output + bias` where
    /// `@` is matrix multiplication.
    ///
    /// If `bias` is present, it is treated as a column vector whose length
    /// must match the rows of `a`.
    pub fn gemm_bias(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
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
    round_up(unrounded, nr)
}

/// Return the block size for the M / row dimension of a GEMM operation.
///
/// The result is always a multiple of `mr`.
fn row_block_size(a_rows: usize, mr: usize) -> usize {
    round_up(64.min(a_rows), mr)
}

/// A single tile of the output matrix.
struct OutputTile {
    /// Pointer to first element in this tile.
    ptr: *mut f32,

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
struct OutputTiles {
    data: *mut f32,

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
unsafe impl Sync for OutputTiles {}

impl OutputTiles {
    /// Expose `data` as a grid of tiles, each with a maximum size of
    /// `tile_rows` * `tile_cols`.
    fn new(mut data: MatrixMut, tile_rows: usize, tile_cols: usize) -> OutputTiles {
        OutputTiles {
            data: data.data_mut().unwrap().as_mut_ptr(),
            rows: data.rows(),
            cols: data.cols(),
            row_stride: data.stride(0),
            tile_rows,
            tile_cols,
            n_row_tiles: div_ceil(data.rows(), tile_rows),
            n_col_tiles: div_ceil(data.cols(), tile_cols),
        }
    }

    /// Return the output tile with the given coordinates in the grid of
    /// output tiles.
    ///
    /// Safety: The caller must guarantee that every tile is operated on by
    /// only a single thread at a time.
    unsafe fn tile(&self, row: usize, col: usize) -> OutputTile {
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
fn gemv(
    kernel: &dyn Kernel,
    a: NdTensorView<f32, 1>,
    b: Matrix,
    mut output_mat: MatrixMut,
    alpha: f32,
    beta: f32,
    bias: Option<f32>,
) {
    assert!(a.is_contiguous());
    assert!(b.is_contiguous());
    assert!(output_mat.is_contiguous());

    let a_cols = a.size(0);
    let b_cols = b.cols();
    let out_data = output_mat.data_mut().unwrap();

    let b_block_size = 256;
    let k_block_size = 256;

    // Partition the matrix and vector into blocks, to achieve effective
    // cache usage and enable parallelism.
    range_chunks(0..b_cols, b_block_size)
        .zip(out_data.chunks_mut(b_block_size))
        .par_bridge()
        .for_each(|(b_block, out_chunk)| {
            let a_data = a.data().unwrap();
            let mut effective_beta = beta;

            for k_block in range_chunks(0..a_cols, k_block_size) {
                let a_block = &a_data[k_block.clone()];
                let b_block = b.slice::<2, _>((k_block, b_block.clone()));

                kernel.gemv_kernel(out_chunk, a_block, b_block, alpha, effective_beta);

                // Reset `beta` so that subsequent updates for each column
                // accumulate into the first update.
                effective_beta = 1.0;
            }

            if let Some(bias) = bias {
                for x in out_chunk {
                    *x += bias;
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
fn gemm_impl(
    kernel: &dyn Kernel,
    out_data: &mut [f32],
    out_row_stride: usize,
    a: GemmInputA,
    b: GemmInputB,
    alpha: f32,
    beta: f32,
    bias: Option<&[f32]>,
) {
    assert!(
        a.cols() == b.rows(),
        "Columns of matrix `a` must match rows of matrix `b`"
    );
    assert!(
        bias.map(|b| b.len()).unwrap_or(a.rows()) == a.rows(),
        "Bias vector length must match rows of matrix `a`"
    );

    if a.rows() == 0 || b.cols() == 0 {
        // Output is empty.
        return;
    }

    // Construct a Matrix from the implied dimensions, to validate the slice length.
    let mut output_mat = MatrixMut::<f32>::from_data_with_strides(
        [a.rows(), b.cols()],
        out_data,
        [out_row_stride, 1],
    )
    .expect("Output buffer should be large enough");

    // Use optimized path for vector-matrix products.
    if let (1, GemmInputA::Unpacked(a), GemmInputB::Unpacked(b)) = (a.rows(), a, b) {
        if let (Some(_), Some(_)) = (a.data(), b.data()) {
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

    let packed_b_size = kc * nc;
    let packed_a_size = mc * kc;

    // Buffers for packed blocks of the matrix.
    //
    // These currently have no alignment specified. The paper mentioned above
    // suggests that aligning to cache-line (ie. 64-byte) boundaries may help
    // performance.
    thread_local!(static PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) });
    thread_local!(static PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) });

    let n_col_blocks = div_ceil(b.cols(), nc);
    let n_row_blocks = div_ceil(a.rows(), mc);

    // In a single-threaded context we get better performance by avoiding Rayon
    // overhead altogether.
    let parallel = rayon::current_num_threads() > 1;

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
                let mut thread_local_packed_b: Option<Vec<f32>> = None;
                let panel_length = depth_range.len();

                let packed_b = match b {
                    GemmInputB::Unpacked(b) => PACKED_B.with(|cell| {
                        let mut packed_b = cell.take();
                        packed_b.clear();
                        packed_b.reserve(packed_b_size);
                        kernel.pack_b_block(
                            &mut packed_b.spare_capacity_mut()[..packed_b_size],
                            b,
                            depth_range.clone(),
                            col_start..col_end,
                        );
                        // Safety: pack_b_block initialized `packed_b_size`
                        // elements.
                        unsafe {
                            packed_b.set_len(packed_b_size);
                        }
                        thread_local_packed_b = Some(packed_b);
                        thread_local_packed_b.as_deref().unwrap()
                    }),
                    GemmInputB::Packed(pm) => pm.block(col_idx, depth_idx),
                    GemmInputB::Virtual(vm) => PACKED_B.with(|cell| {
                        let mut packed_b = cell.take();
                        packed_b.resize(packed_b_size, 0.);
                        vm.pack_b(
                            &mut packed_b,
                            kernel.nr(),
                            depth_range.clone(),
                            col_start..col_end,
                        );
                        thread_local_packed_b = Some(packed_b);
                        thread_local_packed_b.as_deref().unwrap()
                    }),
                };

                // Only use provided `beta` on the first write to this output
                // tile. For subsequent updates accumulate.
                let effective_beta = if depth_range.start == 0 { beta } else { 1.0 };

                // Loop over row blocks.
                (0..n_row_blocks)
                    .maybe_par_iter(parallel)
                    .for_each(|row_idx| {
                        let row_start = row_idx * mc;
                        let row_end = (row_start + mc).min(a.rows());

                        // Borrowed packing buffer for current thread. Returned after
                        // the GEMM block is computed.
                        let mut thread_local_packed_a: Option<Vec<f32>> = None;

                        let packed_a = match a {
                            GemmInputA::Unpacked(a) => PACKED_A.with(|cell| {
                                let mut packed_a = cell.take();
                                packed_a.clear();
                                packed_a.reserve(packed_a_size);
                                kernel.pack_a_block(
                                    &mut packed_a.spare_capacity_mut()[..packed_a_size],
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
                                thread_local_packed_a.as_deref().unwrap()
                            }),
                            GemmInputA::Packed(pm) => pm.block(row_idx, depth_idx),
                        };

                        gemm_block(
                            kernel,
                            &output_tiles,
                            col_start / kernel.nr()..div_ceil(col_end, kernel.nr()),
                            row_start / kernel.mr()..div_ceil(row_end, kernel.mr()),
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
fn gemm_block(
    kernel: &dyn Kernel,
    output: &OutputTiles,
    col_tiles: Range<usize>,
    row_tiles: Range<usize>,
    first_update: bool,
    packed_a: &[f32],
    packed_b: &[f32],
    panel_length: usize,
    alpha: f32,
    beta: f32,
    bias: Option<&[f32]>,
) {
    // Maximum tile size of all supported kernels.
    const MAX_MR: usize = 8;
    const MAX_NR: usize = 32;
    assert!(kernel.nr() <= MAX_NR && kernel.mr() <= MAX_MR);

    let b_panel_size = panel_length * kernel.nr();
    let a_panel_size = kernel.mr() * panel_length;

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

                if out_tile.used_rows == kernel.mr() && out_tile.used_cols == kernel.nr() {
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
                    let mut tmp_out_tile =
                        [std::mem::MaybeUninit::<f32>::uninit(); MAX_MR * MAX_NR];

                    // Safety:
                    //  - Tile size is <= MAX_MR * MAX_NR
                    unsafe {
                        kernel.kernel(
                            std::mem::transmute(tmp_out_tile.as_mut_ptr()),
                            kernel.nr(),
                            a_panel,
                            b_panel,
                            panel_length,
                            alpha,
                            0., // Multiplication with `beta` is handled below.
                        );
                    }

                    for i in 0..out_tile.used_rows {
                        for j in 0..out_tile.used_cols {
                            // Safety: Row and column indices are < used rows /
                            // cols in this tile.
                            unsafe {
                                let out_el = out_tile.ptr.add(out_tile.row_stride * i + j);
                                let tmp = if beta == 0. { 0. } else { *out_el };
                                *out_el = beta * tmp
                                    + tmp_out_tile
                                        .get_unchecked(i * kernel.nr() + j)
                                        .assume_init();
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
                                *out_tile.ptr.add(row * out_tile.row_stride + col) +=
                                    *bias.get_unchecked(row_tile * kernel.mr() + row);
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
    use std::ops::Range;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Matrix, MatrixLayout, NdTensor, Tensor};

    use super::{
        add_scaled_vector, gemm, round_up, GemmExecutor, GemmInputA, GemmInputB, KernelHint,
        VirtualMatrix,
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

    fn run_gemm(
        output: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
        kernel: KernelHint,
    ) {
        let out_row_stride = output.stride(0);
        let gemm = GemmExecutor::with_kernel(kernel).expect("kernel not available");

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

    #[test]
    fn test_add_scaled_vector() {
        let mut dest = vec![1.0, 2.0, 3.0, 4.0];
        let src = vec![10.0, 20.0, 30.0, 40.0];

        add_scaled_vector(&mut dest, &src, 1, 1, 2.0);

        assert_eq!(&dest, &[21.0, 42.0, 63.0, 84.0]);
    }

    #[test]
    fn test_add_scaled_vector_src_stride() {
        let mut dest = vec![1.0, 2.0];
        let src = vec![10.0, 20.0, 30.0];

        add_scaled_vector(&mut dest, &src, 1, 2, 1.0);

        assert_eq!(&dest, &[11.0, 32.0]);
    }

    #[test]
    fn test_add_scaled_vector_dest_stride() {
        let mut dest = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0];

        add_scaled_vector(&mut dest, &src, 2, 1, 1.0);

        assert_eq!(&dest, &[11.0, 2.0, 23.0]);
    }

    #[test]
    #[should_panic(expected = "src and dest vector sizes do not match")]
    fn test_add_scaled_vector_size_mismatch() {
        let mut dest = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0];
        add_scaled_vector(&mut dest, &src, 1, 1, 1.0);
    }

    #[test]
    #[should_panic(expected = "src and dest vector sizes do not match")]
    fn test_add_scaled_vector_strided_size_mismatch() {
        let mut dest = vec![1.0, 2.0];
        let src = vec![10.0, 20.0];
        add_scaled_vector(&mut dest, &src, 2, 1, 1.0);
    }

    // Simplest possible test case for easy debugging.
    #[test]
    fn test_simple_gemm() -> Result<(), Box<dyn Error>> {
        let a = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(&[2, 2], vec![5., 6., 7., 8.]);
        let expected = reference_matmul(&a, &b);

        let mut result = Tensor::zeros(&[a.size(0), b.size(1)]);
        run_gemm(&mut result, &a, &b, 1., 1., None, KernelHint::Auto);
        expect_equal(&result, &expected)?;

        let mut result = Tensor::zeros(&[a.size(0), b.size(1)]);
        run_gemm(&mut result, &a, &b, 1., 1., None, KernelHint::Base);
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

    fn test_gemm_with_kernel(kernel: KernelHint) -> Result<(), Box<dyn Error>> {
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
        test_gemm_with_kernel(KernelHint::Fma)
    }

    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_gemm_with_avx512_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(KernelHint::Avx512)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_with_arm_neon_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(KernelHint::ArmNeon)
    }

    // This duplicates one of the other `test_gemm_with_XXX_kernel` tests
    // depending on what the preferred kernel is. That's OK as long as this
    // test is fast.
    #[test]
    fn test_gemm_with_auto_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(KernelHint::Auto)
    }

    #[test]
    fn test_gemm_with_base_kernel() -> Result<(), Box<dyn Error>> {
        test_gemm_with_kernel(KernelHint::Base)
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
        run_gemm(&mut result, &a, &b, 1., 1., None, KernelHint::Auto);

        let expected = reference_matmul(&a, &b);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_alpha() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        let a = Tensor::rand(&[10, 5], &mut rng);
        let b = Tensor::rand(&[5, 15], &mut rng);

        for kernel in [KernelHint::Auto, KernelHint::Base] {
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

        let a = Tensor::rand(&[10, 5], &mut rng);
        let b = Tensor::rand(&[5, 15], &mut rng);

        for kernel in [KernelHint::Auto, KernelHint::Base] {
            for beta in [0.0, 0.5, 1.0, 2.0] {
                let mut result = Tensor::rand(&[10, 15], &mut rng);
                let mut expected = result.clone();

                run_gemm(&mut result, &a, &b, 1., beta, None, kernel);
                reference_gemm(&mut expected, &a, &b, 1., beta, None);

                expect_equal(&result, &expected)?;
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
            // Vector-matrix multiplication
            Case { m: 1, n: 20, k: 20 },
        ];

        for Case { m, n, k } in cases {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);

            let mut result = Tensor::full(&[m, n], f32::NAN);
            let mut expected = Tensor::zeros(result.shape());

            run_gemm(
                &mut result,
                &a,
                &b,
                1.,
                0., /* beta */
                None,
                KernelHint::Auto,
            );
            reference_gemm(&mut expected, &a, &b, 1., 0. /* beta */, None);

            expect_equal(&result, &expected)?;
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

        for kernel in [KernelHint::Auto, KernelHint::Base] {
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
            let packed_b = gemm.prepack_b(b.nd_view(), a.size(1));

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

        impl<'a> VirtualMatrix for Packer<'a> {
            fn rows(&self) -> usize {
                self.tensor.rows()
            }

            fn cols(&self) -> usize {
                self.tensor.cols()
            }

            fn pack_b(
                &self,
                out: &mut [f32],
                panel_width: usize,
                rows: Range<usize>,
                cols: Range<usize>,
            ) {
                let out_cols = round_up(cols.len(), panel_width);
                let mut out_iter = out.iter_mut();

                for panel_start_col in (0..out_cols).step_by(panel_width) {
                    for row in rows.clone() {
                        for panel_col in 0..panel_width {
                            let col = panel_start_col + panel_col;
                            *out_iter.next().unwrap() = self
                                .tensor
                                .get([row, cols.start + col])
                                .copied()
                                .unwrap_or(0.);
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
        struct Case {
            n: usize,
            k: usize,
            alpha: f32,
            beta: f32,
            bias: Option<f32>,
        }

        let cases = [
            // Empty inputs
            Case {
                n: 0,
                k: 1,
                alpha: 1.,
                beta: 0.,
                bias: None,
            },
            Case {
                n: 1,
                k: 0,
                alpha: 1.,
                beta: 0.,
                bias: None,
            },
            // Smallest possible input
            Case {
                n: 1,
                k: 1,
                alpha: 1.,
                beta: 0.,
                bias: None,
            },
            // n is a multiple of the tile size (16 for AVX 2 / FMA)
            Case {
                n: 16,
                k: 16,
                alpha: 1.,
                beta: 0.,
                bias: None,
            },
            // n is not an exact multiple of the tile size
            Case {
                n: 20,
                k: 16,
                alpha: 1.,
                beta: 1.,
                bias: None,
            },
            // n exceeds column block size
            Case {
                n: 300,
                k: 16,
                alpha: 1.,
                beta: 1.,
                bias: None,
            },
            // k exceeds depth block size
            Case {
                n: 20,
                k: 300,
                alpha: 1.,
                beta: 1.,
                bias: None,
            },
            // beta value = 0.
            Case {
                n: 20,
                k: 300,
                alpha: 1.,
                beta: 0.,
                bias: None,
            },
            // Non-standard beta value
            Case {
                n: 20,
                k: 300,
                alpha: 1.,
                beta: 0.5,
                bias: None,
            },
            // Non-standard alpha value
            Case {
                n: 20,
                k: 20,
                alpha: 0.5,
                beta: 1.,
                bias: None,
            },
            // Test with bias
            Case {
                n: 20,
                k: 20,
                alpha: 1.,
                beta: 0.,
                bias: Some(0.5),
            },
        ];

        let mut rng = XorShiftRng::new(1234);

        for Case {
            n,
            k,
            alpha,
            beta,
            bias,
        } in cases
        {
            let a = Tensor::rand(&[1, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);
            let mut result = Tensor::zeros(&[1, n]);
            let bias_array = bias.map(|b| [b]);

            run_gemm(
                &mut result,
                &a,
                &b,
                alpha,
                beta,
                bias_array.as_ref().map(|b| b.as_slice()),
                KernelHint::Auto,
            );

            let expected =
                reference_matmul_alpha_beta(&a, &b, alpha, beta).map(|x| x + bias.unwrap_or(0.));
            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    use crate::timer::Timer;

    // Run with `cargo test --release bench_gemm -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn bench_gemm() {
        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            // Square output matrix
            Case {
                m: 512,
                n: 512,
                k: 512,
            },
            // Larger square output matrix
            Case {
                m: 1024,
                n: 1024,
                k: 1024,
            },
            // Wide output matrix
            Case {
                m: 128,
                n: 2048,
                k: 512,
            },
            // Tall output matrix
            Case {
                m: 2048,
                n: 128,
                k: 512,
            },
            // Vector-matrix. This is common in transformer decoders for example.
            Case {
                m: 1,
                n: 4096,
                k: 512,
            },
        ];

        println!("Testing kernel {}", GemmExecutor::new().kernel_name());

        for case in cases {
            let Case { m, n, k } = case;

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
            let b = Tensor::rand(&[k, n], &mut rng);

            let mut t = Timer::new();
            t.start();
            for _i in 0..iters {
                run_gemm(&mut result, &a, &b, 1., 0., None, KernelHint::Auto);
            }
            t.end();

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

            let flops = (2 * m * n * k * iters as usize) as f32 / t.elapsed_secs();
            let gflops = flops / (10f32).powi(9);

            println!(
                "m {} n {} k {} iters {}. Duration {}ms ({}ms/iter). GFLOPS {}",
                m,
                n,
                k,
                iters,
                t.elapsed_ms(),
                t.elapsed_ms() / iters as f32,
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

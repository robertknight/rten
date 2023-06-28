//! Optimized linear algebra functions.
//!
//! This module provides a subset of BLAS-like functions that are used by
//! neural network operators. This includes general matrix multiplication ("gemm"),
//! and vector-scalar products.
use std::cell::RefCell;
use std::ops::Range;

use rayon::prelude::*;
use wasnn_tensor::{Matrix, MatrixLayout, MatrixMut, NdTensorLayout};

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

    let src_els = div_ceil(src.len(), src_stride);
    let dest_els = div_ceil(dest.len(), dest_stride);
    assert!(
        src_els == dest_els,
        "src and dest vector sizes do not match"
    );

    const N: usize = 4;
    let n_blocks = src_els / N;
    let mut val = [0.0; N];

    for b in 0..n_blocks {
        for i in 0..N {
            unsafe {
                val[i] = src.get_unchecked((b * N + i) * src_stride) * scale;
            }
        }

        for i in 0..N {
            unsafe {
                *dest.get_unchecked_mut((b * N + i) * dest_stride) += val[i];
            }
        }
    }

    for i in n_blocks * N..src_els {
        unsafe {
            *dest.get_unchecked_mut(i * dest_stride) += src.get_unchecked(i * src_stride) * scale;
        }
    }
}

struct BlockIter {
    start: usize,
    end: usize,
    step: usize,
}

impl Iterator for BlockIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.start < self.end {
            let start = self.start;
            let end = (start + self.step).min(self.end);
            self.start += self.step;
            Some((start, end))
        } else {
            None
        }
    }
}

/// Return an iterator over (block_start, block_end) tuples of `step`-sized
/// blocks between `start` and `end`. If `end - start` is not a multiple of
/// `step` then the final block will be smaller.
fn blocks(start: usize, end: usize, step: usize) -> BlockIter {
    BlockIter { start, end, step }
}

/// Kernel that computes a small tile of a matrix multiplication output.
///
/// The tile size depends upon the kernel and is specified by the `MR` and `NR`
/// associated constants. The MR and NR values are chosen such that an `MR * NR`
/// tile can fit in registers.
///
/// The kernel corresponds to Loop 6 (the "microkernel") in Page 4 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987.
trait Kernel {
    /// Height of output tiles computed by the kernel.
    const MR: usize;

    /// Width of output tiles computed by the kernel.
    const NR: usize;

    /// Return true if this kernel is usable on the current system.
    ///
    /// It is unsafe to call `kernel` if this is false.
    fn supported() -> bool;

    /// Compute a tile of the output matrix. The output is stored in row-major
    /// order with `MR` rows and `NR` columns, a row stride of `tile_row_stride`
    /// and column stride of 1.
    ///
    /// The caller must ensure that the kernel is supported on the current
    /// system, and `tile_ptr` points to a buffer of the correct size.
    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    );
}

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
#[cfg(target_arch = "x86_64")]
struct FMAKernel {}

#[cfg(target_arch = "x86_64")]
impl Kernel for FMAKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;

    fn supported() -> bool {
        is_x86_feature_detected!("fma")
    }

    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        Self::kernel_fma(tile_ptr, tile_row_stride, a, b, depth, alpha, beta)
    }
}

#[cfg(target_arch = "x86_64")]
impl FMAKernel {
    #[target_feature(enable = "fma")]
    unsafe fn kernel_fma(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        use core::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_setzero_ps, _mm256_storeu_ps,
        };
        use std::mem::size_of;

        const MR: usize = FMAKernel::MR;
        const NR: usize = FMAKernel::NR;

        const REG_SIZE: usize = size_of::<__m256>() / size_of::<f32>();
        const NR_REGS: usize = NR / REG_SIZE;
        assert!(NR % REG_SIZE == 0);

        // Check that buffer accesses below are going to be valid.
        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut tmp = [[_mm256_setzero_ps(); NR_REGS]; MR];
        let mut b_rows = [_mm256_setzero_ps(); NR_REGS];

        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..NR_REGS {
                b_rows[i] = _mm256_loadu_ps(b_ptr.add(b_off + i * REG_SIZE));
            }

            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i);
                let a_broadcast = _mm256_set1_ps(a_val);

                for j in 0..NR_REGS {
                    tmp[i][j] = _mm256_fmadd_ps(a_broadcast, b_rows[j], tmp[i][j]);
                }
            }
        }

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    _mm256_storeu_ps(out_ptr, tmp[i][j]);
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm256_add_ps(_mm256_loadu_ps(out_ptr), tmp[i][j]);
                    _mm256_storeu_ps(out_ptr, out_val);
                }
            }
        } else {
            let alpha_broadcast = _mm256_set1_ps(alpha);
            let beta_broadcast = _mm256_set1_ps(beta);
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm256_mul_ps(_mm256_loadu_ps(out_ptr), beta_broadcast);
                    let out_val = _mm256_fmadd_ps(tmp[i][j], alpha_broadcast, out_val);
                    _mm256_storeu_ps(out_ptr, out_val);
                }
            }
        }
    }
}

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
struct BaseKernel {}

impl Kernel for BaseKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32.
    const NR: usize = 4;

    fn supported() -> bool {
        true
    }

    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = BaseKernel::MR;
        const NR: usize = BaseKernel::NR;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        // Accumulate into a fixed-sized array to allow the compiler to generate
        // more efficient code for the loop over `depth`.
        let mut tmp = [[0.0; NR]; MR];
        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..MR {
                for j in 0..NR {
                    tmp[i][j] += a.get_unchecked(a_off + i) * b.get_unchecked(b_off + j);
                }
            }
        }

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = tmp[i][j];
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el += tmp[i][j];
                }
            }
        } else {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = beta * *out_el + alpha * tmp[i][j];
                }
            }
        }
    }
}

/// Pack a block of the "A" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(rows.len() / K::MR)`
/// row panels. Each row panel has size `K::MR * cols.len()` and uses
/// column-major order. If `rows.len()` is not a multiple of `K::MR`, the
/// final panel is zero-padded.
fn pack_a_block<K: Kernel>(out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
    let a_rows = rows.len();
    let a_cols = cols.len();

    let n_panels = round_up(a_rows, K::MR) / K::MR;
    for panel in 0..n_panels {
        let panel_offset = panel * a_cols * K::MR;
        let panel_start_row = panel * K::MR;

        if a_rows - panel_start_row >= K::MR {
            // Optimized loop for panels that don't need any padding
            let a_offset =
                (rows.start + panel_start_row) * a.row_stride() + cols.start * a.col_stride();

            assert!(out.len() > panel_offset + (a_cols - 1) * K::MR + K::MR - 1);
            assert!(
                a.data().len()
                    > a_offset + (K::MR - 1) * a.row_stride() + (a_cols - 1) * a.col_stride()
            );

            for col in 0..a_cols {
                for row in 0..K::MR {
                    // Safety: Indexes are less than lengths asserted above.
                    unsafe {
                        *out.get_unchecked_mut(panel_offset + col * K::MR + row) = *a
                            .data()
                            .get_unchecked(a_offset + row * a.row_stride() + col * a.col_stride());
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for col in 0..a_cols {
                let out_col_offset = panel_offset + col * K::MR;
                for row in 0..K::MR {
                    let a_row = rows.start + panel_start_row + row;
                    out[out_col_offset + row] = if a_row < rows.end {
                        a.data()[a_row * a.row_stride() + (cols.start + col) * a.col_stride()]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// Pack a block of the "B" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(cols.len() /
/// K::NR)` column panels. Each column panel has size `rows.len() *
/// K::NR` and uses row-major order. If `cols.len()` is not a multiple of
/// `K::NR`, the final panel is zero-padded.
fn pack_b_block<K: Kernel>(out: &mut [f32], b: Matrix, rows: Range<usize>, cols: Range<usize>) {
    let b_cols = cols.len();
    let b_rows = rows.len();

    let n_panels = round_up(b_cols, K::NR) / K::NR;
    for panel in 0..n_panels {
        let panel_offset = panel * b_rows * K::NR;
        let panel_start_col = panel * K::NR;

        if b_cols - panel_start_col >= K::NR {
            // Optimized loop for panels that don't need any padding
            let b_offset =
                rows.start * b.row_stride() + (cols.start + panel_start_col) * b.col_stride();

            assert!(out.len() >= panel_offset + (b_rows - 1) * K::NR + K::NR);
            assert!(
                b.data().len()
                    > b_offset + (b_rows - 1) * b.row_stride() + (K::NR - 1) * b.col_stride()
            );

            for row in 0..b_rows {
                for col in 0..K::NR {
                    // Safety: Indexes are less than lengths asserted above.
                    unsafe {
                        *out.get_unchecked_mut(panel_offset + row * K::NR + col) = *b
                            .data()
                            .get_unchecked(b_offset + row * b.row_stride() + col * b.col_stride());
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for row in 0..b_rows {
                let out_row_offset = panel_offset + row * K::NR;
                let b_row_offset = (rows.start + row) * b.row_stride();

                for col in 0..K::NR {
                    let out_col = panel_start_col + col;
                    let b_offset =
                        b_row_offset + (cols.start + panel_start_col + col) * b.col_stride();

                    out[out_row_offset + col] = if out_col < b_cols {
                        b.data()[b_offset]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
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

/// Metadata for a left-hand or "A" GEMM input that has been pre-packed.
#[derive(Clone)]
pub struct PackedAMatrix {
    /// Sequence of packed row panels.
    data: Vec<f32>,

    /// Number of elements in each row panel.
    panel_len: usize,

    /// Number of blocks that the matrix was divided into along the M dimension.
    row_blocks: usize,

    /// Number of rows in the unpacked matrix.
    rows: usize,

    /// Number of columns in the unpacked matrix.
    cols: usize,
}

impl PackedAMatrix {
    fn block(&self, row_block_idx: usize, depth_block_idx: usize) -> &[f32] {
        let panel_idx = depth_block_idx * self.row_blocks + row_block_idx;
        let offset = panel_idx * self.panel_len;
        &self.data[offset..offset + self.panel_len]
    }
}

/// Metadata for a right-hand or "B" GEMM input that has been pre-packed.
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
    Packed(&'a PackedAMatrix),
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
    #[cfg(target_arch = "x86_64")]
    {
        if FMAKernel::supported() {
            return FMAKernel {}.gemm(
                out_data,
                out_row_stride,
                GemmInputA::Unpacked(a),
                GemmInputB::Unpacked(b),
                alpha,
                beta,
                None, // bias
            );
        }
    }

    BaseKernel {}.gemm(
        out_data,
        out_row_stride,
        GemmInputA::Unpacked(a),
        GemmInputB::Unpacked(b),
        alpha,
        beta,
        None, // bias
    )
}

/// Trait for kernel-specific GEMM operations.
trait GemmOps: Sync {
    fn pack_a_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>);
    fn pack_b_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>);
    fn gemm(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
    );
}

impl GemmOps for BaseKernel {
    fn pack_a_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
        pack_a_block::<Self>(out, a, rows, cols);
    }

    fn pack_b_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
        pack_b_block::<Self>(out, a, rows, cols);
    }

    fn gemm(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
    ) {
        gemm_impl::<Self, { Self::MR * Self::NR }>(
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

#[cfg(target_arch = "x86_64")]
impl GemmOps for FMAKernel {
    fn pack_a_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
        pack_a_block::<Self>(out, a, rows, cols);
    }

    fn pack_b_block(&self, out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
        pack_b_block::<Self>(out, a, rows, cols);
    }

    fn gemm(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
        alpha: f32,
        beta: f32,
        bias: Option<&[f32]>,
    ) {
        gemm_impl::<Self, { Self::MR * Self::NR }>(
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
    kernel: Box<dyn GemmOps>,

    /// Tile width used by kernel.
    nr: usize,

    /// Tile height used by kernel.
    mr: usize,
}

/// Hints for which kernel to use.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Currently only used in tests
pub enum KernelHint {
    /// Use the preferred kernel for the current platform
    Auto,
    /// Use the fallback/base kernel
    Base,
}

impl GemmExecutor {
    /// Create a [GemmExecutor] using the preferred kernel for the current system.
    pub fn new() -> GemmExecutor {
        #[cfg(target_arch = "x86_64")]
        {
            if FMAKernel::supported() {
                return GemmExecutor {
                    kernel: Box::new(FMAKernel {}),
                    nr: FMAKernel::NR,
                    mr: FMAKernel::MR,
                };
            }
        }
        Self::new_with_base_kernel()
    }

    /// Create a [GemmExecutor] using the given kernel.
    #[allow(dead_code)] // Currently only used in tests
    pub fn new_with_kernel(kernel: KernelHint) -> GemmExecutor {
        match kernel {
            KernelHint::Auto => Self::new(),
            KernelHint::Base => Self::new_with_base_kernel(),
        }
    }

    fn new_with_base_kernel() -> GemmExecutor {
        GemmExecutor {
            kernel: Box::new(BaseKernel {}),
            nr: BaseKernel::NR,
            mr: BaseKernel::MR,
        }
    }

    /// Prepack a matrix for use as the left-hand or "A" input.
    pub fn prepack_a(&self, a: Matrix) -> PackedAMatrix {
        let kc = depth_block_size(a.cols());
        let mc = row_block_size(a.rows(), self.mr);
        let panel_len = kc * mc;
        let row_blocks = div_ceil(a.rows(), mc);
        let depth_blocks = div_ceil(a.cols(), kc);

        let mut out = vec![0.; depth_blocks * row_blocks * panel_len];

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = out.chunks_exact_mut(panel_len);
        for (depth_start, depth_end) in blocks(0, a.cols(), kc) {
            for (row_start, row_end) in blocks(0, a.rows(), mc) {
                let out_panel = out_panels.next().unwrap();
                self.kernel
                    .pack_a_block(out_panel, a, row_start..row_end, depth_start..depth_end);
            }
        }

        PackedAMatrix {
            data: out,
            rows: a.rows(),
            cols: a.cols(),
            panel_len,
            row_blocks,
        }
    }

    /// Prepack a matrix for use as the right-hand or "B" matrix input.
    pub fn prepack_b(&self, b: Matrix, a_cols: usize) -> PackedBMatrix {
        let nc = col_block_size(b.cols(), self.nr);
        let kc = depth_block_size(a_cols);
        let panel_len = nc * kc;
        let depth_blocks = div_ceil(a_cols, kc);
        let col_blocks = div_ceil(b.cols(), nc);

        let mut out = vec![0.; col_blocks * depth_blocks * panel_len];

        // Pack blocks in the order they will be accessed by the GEMM
        // implementation.
        let mut out_panels = out.chunks_exact_mut(panel_len);
        for (col_start, col_end) in blocks(0, b.cols(), nc) {
            for (depth_start, depth_end) in blocks(0, a_cols, kc) {
                let out_panel = out_panels.next().unwrap();
                self.kernel
                    .pack_b_block(out_panel, b, depth_start..depth_end, col_start..col_end);
            }
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
    pub fn gemm(
        &self,
        out_data: &mut [f32],
        out_row_stride: usize,
        a: GemmInputA,
        b: GemmInputB,
        alpha: f32,
        beta: f32,
    ) {
        self.kernel
            .gemm(out_data, out_row_stride, a, b, alpha, beta, None)
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
        self.kernel
            .gemm(out_data, out_row_stride, a, b, alpha, beta, bias)
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
    round_up(1024.min(b_cols), nr)
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
            data: data.data_mut().as_mut_ptr(),
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

/// Perform matrix multiplication with a given kernel.
///
/// `MR_NR` should be computed as `K::MR * K::NR`. This function can't compute
/// that itself due to Rust limitations on using generic parameters in const
/// expressions.
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
fn gemm_impl<K: Kernel, const MR_NR: usize>(
    out_data: &mut [f32],
    out_row_stride: usize,
    a: GemmInputA,
    b: GemmInputB,
    alpha: f32,
    beta: f32,
    bias: Option<&[f32]>,
) {
    assert!(K::supported());
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
    let output_mat =
        MatrixMut::<f32>::from_data(out_data, [a.rows(), b.cols()], Some([out_row_stride, 1]))
            .expect("Output buffer should be large enough");
    let output_tiles = OutputTiles::new(output_mat, K::MR, K::NR);

    // The constant values for block sizes below were taken from the
    // matrixmultiply crate. See https://dl.acm.org/doi/pdf/10.1145/2925987 for
    // an explanation of how suitable values are determined. Since we don't know
    // exactly which CPU this code will be run on, we try to pick something that
    // will work well on most systems.

    // Sizes of blocks that the width (nc), depth (kc) and height (mc)
    // dimensions are partitioned into in the outer loops. These are chosen
    // so that blocks can fit in specific cache levels.
    let nc = col_block_size(b.cols(), K::NR);
    let mc = row_block_size(a.rows(), K::MR);
    let kc = depth_block_size(a.cols());

    let packed_b_size = kc * nc;
    let packed_a_size = mc * kc;

    // Buffers for packed blocks of the matrix.
    //
    // These currently have no alignment specified. The paper mentioned above
    // suggests that aligning to cache-line (ie. 64-byte) boundaries may help
    // performance.
    thread_local!(static PACKED_A: RefCell<Vec<f32>> = RefCell::new(Vec::new()));
    thread_local!(static PACKED_B: RefCell<Vec<f32>> = RefCell::new(Vec::new()));

    let n_col_blocks = div_ceil(b.cols(), nc);
    let n_row_blocks = div_ceil(a.rows(), mc);

    // Loop over column blocks.
    (0..n_col_blocks).into_par_iter().for_each(|col_idx| {
        let col_start = col_idx * nc;
        let col_end = (col_start + nc).min(b.cols());

        // Loop over depth blocks. This is not parallelized because output
        // tiles are shared across iterations.
        for (depth_idx, (depth_start, depth_end)) in blocks(0, a.cols(), kc).enumerate() {
            // Borrowed packing buffer for current thread. Returned after
            // the GEMM block is computed.
            let mut thread_local_packed_b: Option<Vec<f32>> = None;
            let panel_length = depth_end - depth_start;

            let packed_b = match b {
                GemmInputB::Unpacked(b) => PACKED_B.with(|cell| {
                    let mut packed_b = cell.take();
                    packed_b.resize(packed_b_size, 0.);
                    pack_b_block::<K>(&mut packed_b, b, depth_start..depth_end, col_start..col_end);
                    thread_local_packed_b = Some(packed_b);
                    thread_local_packed_b.as_deref().unwrap()
                }),
                GemmInputB::Packed(pm) => pm.block(col_idx, depth_idx),
                GemmInputB::Virtual(vm) => PACKED_B.with(|cell| {
                    let mut packed_b = cell.take();
                    packed_b.resize(packed_b_size, 0.);
                    vm.pack_b(
                        &mut packed_b,
                        K::NR,
                        depth_start..depth_end,
                        col_start..col_end,
                    );
                    thread_local_packed_b = Some(packed_b);
                    thread_local_packed_b.as_deref().unwrap()
                }),
            };

            // Only use provided `beta` on the first write to this output
            // tile. For subsequent updates accumulate.
            let effective_beta = if depth_start == 0 { beta } else { 1.0 };

            // Loop over row blocks.
            //
            // TODO - This should be parallel, but overhead in single-threaded
            // environments (ie. `RAYON_NUM_THREADS=1`) needs to be reduced.
            (0..n_row_blocks).for_each(|row_idx| {
                let row_start = row_idx * mc;
                let row_end = (row_start + mc).min(a.rows());

                // Borrowed packing buffer for current thread. Returned after
                // the GEMM block is computed.
                let mut thread_local_packed_a: Option<Vec<f32>> = None;

                let packed_a = match a {
                    GemmInputA::Unpacked(a) => PACKED_A.with(|cell| {
                        let mut packed_a = cell.take();
                        packed_a.resize(packed_a_size, 0.);
                        pack_a_block::<K>(
                            &mut packed_a,
                            a,
                            row_start..row_end,
                            depth_start..depth_end,
                        );
                        thread_local_packed_a = Some(packed_a);
                        thread_local_packed_a.as_deref().unwrap()
                    }),
                    GemmInputA::Packed(pm) => pm.block(row_idx, depth_idx),
                };

                gemm_block::<K, MR_NR>(
                    &output_tiles,
                    col_start / K::NR..div_ceil(col_end, K::NR),
                    row_start / K::MR..div_ceil(row_end, K::MR),
                    depth_start == 0,
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
fn gemm_block<K: Kernel, const MR_NR: usize>(
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
    let b_panel_size = panel_length * K::NR;
    let a_panel_size = K::MR * panel_length;

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

                if out_tile.used_rows == K::MR && out_tile.used_cols == K::NR {
                    // Safety:
                    //  - Tile size is MR * NR
                    //  - We checked this kernel is supported
                    unsafe {
                        K::kernel(
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
                    let mut tmp_out_tile = [0.; MR_NR];

                    // Safety:
                    //  - Tile size is MR * NR
                    //  - We checked this kernel is supported
                    unsafe {
                        K::kernel(
                            tmp_out_tile.as_mut_ptr(),
                            K::NR,
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
                                *out_el =
                                    beta * *out_el + tmp_out_tile.get_unchecked(i * K::NR + j);
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
                                    *bias.get_unchecked(row_tile * K::MR + row);
                            }
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use wasnn_tensor::rng::XorShiftRng;
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{Matrix, MatrixLayout, Tensor, TensorLayout};

    use crate::linalg::{
        add_scaled_vector, gemm, round_up, GemmExecutor, GemmInputA, GemmInputB, KernelHint,
        VirtualMatrix,
    };

    fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let [a_rows, _a_cols] = a.dims();
        let [_b_rows, b_cols] = b.dims();
        let mut output = Tensor::zeros(&[a_rows, b_cols]);

        reference_gemm(&mut output, a, b, 1.0, 0.0, None);

        output
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
        let gemm = GemmExecutor::new_with_kernel(kernel);

        gemm.gemm_bias(
            output.data_mut(),
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
        let [a_rows, a_cols] = a.dims();
        let [_b_rows, b_cols] = b.dims();

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
    fn test_simple_gemm() -> Result<(), String> {
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

    fn test_gemm_with_kernel(kernel: KernelHint) -> Result<(), String> {
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
                return Err(err);
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_with_fastest_kernel() -> Result<(), String> {
        test_gemm_with_kernel(KernelHint::Auto)
    }

    #[test]
    fn test_gemm_with_base_kernel() -> Result<(), String> {
        test_gemm_with_kernel(KernelHint::Base)
    }

    #[test]
    fn test_gemm_transposed() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let mut a = Tensor::rand(&[20, 30], &mut rng);
        let mut b = Tensor::rand(&[10, 20], &mut rng);

        // Transpose the input matrices. This will alter their row and column
        // strides and shapes, but not re-order the data.
        a.permute(&[1, 0]);
        b.permute(&[1, 0]);

        let [a_rows, _] = a.dims();
        let [_, b_cols] = b.dims();

        let mut result = Tensor::zeros(&[a_rows, b_cols]);
        run_gemm(&mut result, &a, &b, 1., 1., None, KernelHint::Auto);

        let expected = reference_matmul(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_alpha() -> Result<(), String> {
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
    fn test_gemm_beta() -> Result<(), String> {
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
    fn test_gemm_bias() -> Result<(), String> {
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
    fn test_gemm_prepack() -> Result<(), String> {
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
                result.data_mut(),
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
                expected.data_mut(),
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
    fn test_gemm_virtual() -> Result<(), String> {
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
                result.data_mut(),
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

    // TODO - Add a set of tests for use with Miri. These should exercise all
    // unsafe code, but be adjusted to run quickly.
}

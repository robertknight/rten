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
use rten_tensor::{
    Alloc, AssumeInit, GlobalAlloc, Matrix, MatrixLayout, MatrixMut, NdTensor, NdTensorView,
    Storage,
};

use crate::iter_util::{range_chunks, MaybeParIter};
use crate::number::Identities;
use crate::slice_cast::Pod;

mod errors;
mod im2col;
mod kernels;
mod packing;
mod prepack;
mod tiles;

pub use errors::GemmError;
pub use im2col::{ColOffsets, Im2Col, RowOffsets};
use kernels::generic::GenericKernel;
pub use kernels::QuantParams;
use kernels::{Kernel, MatVecOutput};
use packing::PackingBuffer;
pub use prepack::{PackedAMatrix, PackedBMatrix};
use tiles::OutputTiles;

pub type GemmResult<T = ()> = Result<T, GemmError>;

/// Left-hand or "A" input for a GEMM operation.
#[derive(Copy, Clone)]
pub enum GemmInputA<'a, T> {
    /// A standard unpacked matrix.
    Unpacked(Matrix<'a, T>),

    /// A matrix which has been pre-packed by [`GemmExecutor::prepack_a`].
    Packed(&'a PackedAMatrix<T>),
}

impl<T> GemmInputA<'_, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.rows(),
            Self::Packed(pm) => pm.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.cols(),
        }
    }
}

/// Trait implemented by GEMM input types.
pub trait GemmInT: Copy + Default + Send + Sync + Identities + Pod {}
impl GemmInT for i8 {}
impl GemmInT for u8 {}
impl GemmInT for f32 {}

/// Trait implemented by GEMM output types.
pub trait GemmOutT:
    Copy
    + Default
    + PartialEq
    + Send
    + Sync
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + Identities
    + Pod
{
}
impl GemmOutT for i32 {}
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
            Self::Packed(pm) => pm.rows(),
            Self::Im2Col(im) => im.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Unpacked(m) => m.cols(),
            Self::Packed(pm) => pm.cols(),
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

/// Argument for [`GemmExecutor::with_kernel`] specifying which kernel to use.
#[derive(Clone, Copy, Debug)]
enum F32KernelType {
    /// Use the fallback/generic kernel. Always available.
    #[allow(unused)]
    Generic,

    /// Use the AVX 2 + FMA kernel. Intel x64 only.
    #[cfg(target_arch = "x86_64")]
    Fma,

    /// Use the AVX 512 kernel. Intel x64 only.
    #[cfg(target_arch = "x86_64")]
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

#[derive(Clone, Copy, Debug)]
enum Int8KernelType {
    #[allow(unused)]
    Generic,

    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "avx512")]
    Avx512,

    #[cfg(target_arch = "aarch64")]
    ArmDot,
    #[cfg(target_arch = "aarch64")]
    ArmNeon,

    #[cfg(target_arch = "wasm32")]
    #[cfg(target_feature = "simd128")]
    Wasm,
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
pub struct GemmExecutor<LhsT: GemmInT = f32, RhsT: GemmInT = LhsT, OutT: GemmOutT = LhsT> {
    kernel: Box<dyn Kernel<LhsT, RhsT, OutT>>,
}

impl<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT> GemmExecutor<LhsT, RhsT, OutT> {
    pub fn new() -> Self
    where
        Self: Default,
    {
        Self::default()
    }

    /// Return the name of the kernel that this executor is using.
    #[allow(dead_code)]
    pub fn kernel_name(&self) -> &str {
        self.kernel.name()
    }

    /// Prepack a matrix for use as the left-hand or "A" input.
    #[allow(unused)]
    pub fn prepack_a(&self, a: Matrix<LhsT>) -> PackedAMatrix<LhsT> {
        self.prepack_a_in(GlobalAlloc::new(), a)
    }

    /// Variant of [`prepack_a`](GemmExecutor::prepack_a) which takes an
    /// allocator.
    pub fn prepack_a_in<A: Alloc>(&self, alloc: A, a: Matrix<LhsT>) -> PackedAMatrix<LhsT> {
        prepack::prepack_a(&*self.kernel, alloc, a)
    }

    /// Return column count step for building an [`Im2Col`] input.
    ///
    /// The number of columns in [`ColOffsets`] must be a multiple of this.
    pub fn im2col_col_count_step(&self) -> usize {
        self.kernel.im2col_col_count_step()
    }

    /// Return row count step for building an [`Im2Col`] input.
    ///
    /// The number of rows in [`RowOffsets`] must be a multiple of this.
    pub fn im2col_row_count_step(&self) -> usize {
        self.kernel.im2col_row_count_step()
    }

    /// Prepack a matrix for use as the right-hand or "B" matrix input.
    #[allow(unused)]
    pub fn prepack_b(&self, b: Matrix<RhsT>) -> PackedBMatrix<RhsT> {
        self.prepack_b_in(GlobalAlloc::new(), b)
    }

    /// Variant of [`prepack_b`](GemmExecutor::prepack_b) which takes an
    /// allocator.
    pub fn prepack_b_in<A: Alloc>(&self, alloc: A, b: Matrix<RhsT>) -> PackedBMatrix<RhsT> {
        prepack::prepack_b(&*self.kernel, alloc, b)
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
        a: GemmInputA<LhsT>,
        b: GemmInputB<RhsT>,
        alpha: f32,
        beta: OutT,
        bias: Option<BiasVector<OutT>>,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    ) -> GemmResult {
        gemm_impl(
            &*self.kernel,
            // Safety: `gemm_impl` only writes initialized values to `out_data`.
            unsafe { std::mem::transmute::<&mut [OutT], &mut [MaybeUninit<OutT>]>(out_data) },
            a,
            b,
            alpha,
            beta,
            bias,
            a_quant,
            b_quant,
        )
        .map(|_| ())
    }

    /// Perform a General Matrix Multiplication ("gemm").
    ///
    /// This is the same as [`GemmExecutor::gemm`] but takes an uninitialized
    /// output slice. The `beta` value is implicitly set to zero.
    pub fn gemm_uninit<'a>(
        &self,
        out_data: &'a mut [MaybeUninit<OutT>],
        a: GemmInputA<LhsT>,
        b: GemmInputB<RhsT>,
        alpha: f32,
        bias: Option<BiasVector<OutT>>,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    ) -> GemmResult<&'a mut [OutT]> {
        gemm_impl(
            &*self.kernel,
            out_data,
            a,
            b,
            alpha,
            OutT::zero(),
            bias,
            a_quant,
            b_quant,
        )
    }

    /// Perform a batched matrix multiplication.
    ///
    /// This performs a series of matrix multiplications between `M x K` sized
    /// matrices in `a` and `K x N` sized matrices in `b`, writing `M x N`-sized
    /// results into `out_data`.
    pub fn batched_gemm_uninit<'a>(
        &self,
        out_data: &'a mut [MaybeUninit<OutT>],
        a: &[GemmInputA<LhsT>],
        b: &[GemmInputB<RhsT>],
        alpha: f32,
        bias: Option<BiasVector<OutT>>,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    ) -> GemmResult<&'a mut [OutT]> {
        if a.len() != b.len() {
            return Err(GemmError::BatchSizeMismatch);
        }

        let out_mat_stride = match (a, b) {
            ([a, ..], [b, ..]) => a.rows() * b.cols(),
            _ => 0,
        };

        if a.len() * out_mat_stride != out_data.len() {
            return Err(GemmError::OutputSizeMismatch);
        }

        match (a, b) {
            ([], []) => {
                // Safety: Output is empty and thus already initialized.
                Ok(unsafe { out_data.assume_init() })
            }
            ([a], [b]) => {
                // Skip parallel iteration for batch size of 1
                self.gemm_uninit(out_data, *a, *b, alpha, bias, a_quant, b_quant)
            }
            (a, b) => {
                a.par_iter()
                    .zip(b)
                    .zip(out_data.par_chunks_mut(out_mat_stride))
                    .try_for_each(|((a_mat, b_mat), out_mat)| {
                        self.gemm_uninit(out_mat, *a_mat, *b_mat, alpha, bias, a_quant, b_quant)
                            .map(|_| ())
                    })?;

                // Safety: All elements in `out_data` have been initialized by calls to
                // `gemm_uninit`.
                Ok(unsafe { out_data.assume_init() })
            }
        }
    }

    /// Return true if the GEMM kernel may encounter saturation in a data type
    /// that is smaller than the output.
    ///
    /// In this case the output may be incorrect if the range of the inputs
    /// is not restricted to account for this.
    ///
    /// The main offender is the `vpmaddubsw` instruction used in int8 kernels
    /// on x64, on systems which don't support VNNI.
    pub fn may_saturate(&self) -> bool {
        self.kernel.may_saturate()
    }

    fn from_kernel<K: Kernel<LhsT, RhsT, OutT> + 'static>() -> Option<Self> {
        K::new().map(|kernel| GemmExecutor {
            kernel: Box::new(kernel),
        })
    }
}

/// Try to construct a [`GemmExecutor`] with a given kernel type.
macro_rules! try_kernel {
    ($hint:expr) => {
        if let Some(gemm) = Self::with_kernel($hint) {
            return gemm;
        }
    };
}

/// Trait for instantiating a [`GemmExecutor`] with a particular kernel.
///
/// This primarily exists to support creating tests which are run against
/// all available kernels.
trait WithKernel: Sized {
    /// Enum specifying kernel to use.
    type KernelType;

    /// Try to instantiate this executor with a given kernel. Returns None if
    /// the kernel is not supported on this system.
    fn with_kernel(kern_type: Self::KernelType) -> Option<Self>;

    /// Instantiate this executor with the generic/fallback kernel.
    fn with_generic_kernel() -> Self;

    /// Return all the kernel types supported on the current system.
    #[allow(unused)]
    fn kernel_types() -> Vec<Self::KernelType>;
}

impl WithKernel for GemmExecutor<f32, f32, f32> {
    type KernelType = F32KernelType;

    /// Create a [`GemmExecutor`] using the given kernel. Returns `None` if the
    /// kernel is not supported.
    #[allow(dead_code)] // Currently only used in tests
    fn with_kernel(hint: F32KernelType) -> Option<Self> {
        match hint {
            #[cfg(feature = "avx512")]
            #[cfg(target_arch = "x86_64")]
            F32KernelType::Avx512 => Self::from_kernel::<kernels::x86_64::Avx512Kernel>(),
            #[cfg(target_arch = "x86_64")]
            F32KernelType::Fma => Self::from_kernel::<kernels::x86_64::FmaKernel>(),
            #[cfg(target_arch = "aarch64")]
            F32KernelType::ArmNeon => Self::from_kernel::<kernels::aarch64::ArmNeonKernel>(),
            #[cfg(target_arch = "wasm32")]
            #[cfg(target_feature = "simd128")]
            F32KernelType::Wasm => Self::from_kernel::<kernels::wasm::WasmKernel>(),
            F32KernelType::Generic => Some(Self::with_generic_kernel()),
        }
    }

    fn kernel_types() -> Vec<F32KernelType> {
        let mut types = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            types.push(F32KernelType::Avx512);
            types.push(F32KernelType::Fma);
        }

        #[cfg(target_arch = "aarch64")]
        {
            types.push(F32KernelType::ArmNeon);
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            types.push(F32KernelType::Wasm);
        }

        types.push(F32KernelType::Generic);

        types
    }

    /// Construct a GemmExecutor that uses the generic kernel.
    fn with_generic_kernel() -> Self {
        Self::from_kernel::<GenericKernel>().unwrap()
    }
}

impl Default for GemmExecutor<f32, f32, f32> {
    fn default() -> Self {
        #[cfg(feature = "avx512")]
        #[cfg(target_arch = "x86_64")]
        try_kernel!(F32KernelType::Avx512);
        #[cfg(target_arch = "x86_64")]
        try_kernel!(F32KernelType::Fma);
        #[cfg(target_arch = "aarch64")]
        try_kernel!(F32KernelType::ArmNeon);
        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        try_kernel!(F32KernelType::Wasm);
        Self::with_generic_kernel()
    }
}

impl WithKernel for GemmExecutor<u8, i8, i32> {
    type KernelType = Int8KernelType;

    fn with_kernel(hint: Int8KernelType) -> Option<Self> {
        match hint {
            #[cfg(feature = "avx512")]
            #[cfg(target_arch = "x86_64")]
            Int8KernelType::Avx512 => Self::from_kernel::<kernels::x86_64::Avx512Int8Kernel>(),
            #[cfg(target_arch = "x86_64")]
            Int8KernelType::Avx2 => Self::from_kernel::<kernels::x86_64::Avx2Int8Kernel>(),
            #[cfg(target_arch = "aarch64")]
            Int8KernelType::ArmNeon => Self::from_kernel::<kernels::aarch64::ArmInt8Kernel>(),
            #[cfg(target_arch = "aarch64")]
            Int8KernelType::ArmDot => Self::from_kernel::<kernels::aarch64::ArmInt8DotKernel>(),
            #[cfg(target_arch = "wasm32")]
            #[cfg(target_feature = "simd128")]
            Int8KernelType::Wasm => Self::from_kernel::<kernels::wasm::WasmInt8Kernel>(),
            Int8KernelType::Generic => Self::from_kernel::<GenericKernel>(),
        }
    }

    fn kernel_types() -> Vec<Int8KernelType> {
        let mut types = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            types.push(Int8KernelType::Avx512);
            types.push(Int8KernelType::Avx2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            types.push(Int8KernelType::ArmDot);
            types.push(Int8KernelType::ArmNeon);
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            types.push(Int8KernelType::Wasm);
        }

        types.push(Int8KernelType::Generic);

        types
    }

    fn with_generic_kernel() -> Self {
        Self::from_kernel::<GenericKernel>().unwrap()
    }
}

impl Default for GemmExecutor<u8, i8, i32> {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            try_kernel!(Int8KernelType::Avx512);
            try_kernel!(Int8KernelType::Avx2);
        }
        #[cfg(target_arch = "aarch64")]
        {
            try_kernel!(Int8KernelType::ArmDot);
            try_kernel!(Int8KernelType::ArmNeon);
        }
        #[cfg(target_arch = "wasm32")]
        {
            try_kernel!(Int8KernelType::Wasm);
        }
        Self::with_generic_kernel()
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

/// Compute a vector-matrix product.
///
/// This operation is called "gemv" in BLAS APIs.
fn gemv<'a, LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    a: NdTensorView<LhsT, 1>,
    b: Matrix<RhsT>,
    out_data: &'a mut [MaybeUninit<OutT>],
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
) -> &'a mut [OutT] {
    let a_cols = a.size(0);
    let b_cols = b.cols();

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

            let b_quant = b_quant.map(|bq| QuantParams {
                zero_point: &bq.zero_point[col_block.clone()],
            });

            for (k_block, a_block) in
                range_chunks(0..a_cols, k_block_size).zip(a_data.chunks(k_block_size))
            {
                let b_block = b.slice((k_block, col_block.clone()));
                let mat_vec_out = if effective_beta == OutT::zero() {
                    MatVecOutput::from_uninit_slice(out_chunk)
                } else {
                    // Safety: Output is initialized if `effective_beta` is non-zero.
                    MatVecOutput::from_slice(unsafe { out_chunk.assume_init() }, effective_beta)
                };

                kernel.gemv_kernel(mat_vec_out, a_block, b_block, alpha, a_quant, b_quant);

                // Reset `beta` so that subsequent updates for each column
                // accumulate into the first update.
                effective_beta = OutT::one();
            }

            // Safety: Calls to `gemv_kernel` initialized all output elements.
            let out_chunk = unsafe { out_chunk.assume_init() };
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

    // Safety: All output elements were initialized.
    unsafe { out_data.assume_init() }
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
fn gemm_impl<'a, LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    out_data: &'a mut [MaybeUninit<OutT>],
    a: GemmInputA<LhsT>,
    b: GemmInputB<RhsT>,
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
) -> GemmResult<&'a mut [OutT]> {
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

    if let Some(a_quant) = a_quant {
        if a_quant.zero_point.len() != a.rows() {
            return Err(GemmError::WrongQuantParamSize);
        }
    }

    if let Some(b_quant) = b_quant {
        if b_quant.zero_point.len() != b.cols() {
            return Err(GemmError::WrongQuantParamSize);
        }
    }

    // Construct a Matrix from the implied dimensions, to validate the slice length.
    let mut output_mat =
        MatrixMut::<MaybeUninit<OutT>>::try_from_data([a.rows(), b.cols()], out_data)
            .map_err(|_| GemmError::OutputSizeMismatch)?;

    // Handle case where output is empty.
    if a.rows() == 0 || b.cols() == 0 {
        let empty = NdTensor::zeros(output_mat.shape());
        return Ok(output_mat.init_from(&empty).into_slice_mut().unwrap());
    }

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
                    NdTensorView::from_data([a.rows(), 1], bias).broadcast([a.rows(), b.cols()])
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
        return Ok(output_mat.into_slice_mut().unwrap());
    }

    // Use optimized path for vector-matrix products.
    if let (1, GemmInputA::Unpacked(a), GemmInputB::Unpacked(b)) = (a.rows(), a, b) {
        let output = gemv(
            kernel,
            a.slice(0),
            b,
            output_mat.into_slice_mut().unwrap(),
            alpha,
            beta,
            // nb. We checked above that, if present, the bias length matches
            // `a.rows()` or `b.cols()` as appropriate.
            bias,
            a_quant,
            b_quant,
        );
        return Ok(output);
    }

    let output_tiles = OutputTiles::new(output_mat.view_mut(), kernel.mr(), kernel.nr());

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
        packed.validate(kernel, kc)?;
    }
    if let GemmInputB::Packed(packed) = &b {
        packed.validate(kernel, kc)?;
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

                        let layout =
                            kernel.packed_b_layout(depth_range.len(), col_end - col_start, b_quant);
                        let packed_uninit = packed_b.alloc(layout.size(), layout.align());

                        match b {
                            GemmInputB::Unpacked(b) => kernel.pack_b_block(
                                packed_uninit,
                                b,
                                depth_range.clone(),
                                col_start..col_end,
                                b_quant,
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
                                    a_quant,
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
                                    a_quant,
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
                            a_quant,
                            b_quant,
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

    // Safety: All elements of output matrix have been initialized.
    let output = unsafe { output_mat.assume_init() };

    Ok(output.into_slice_mut().unwrap())
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
fn gemm_block<LhsT: Sync, RhsT: Sync, OutT: GemmOutT>(
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
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
) {
    let (mr, nr) = (kernel.mr(), kernel.nr());

    // Sanity check input length here rather than inner loop.
    if let LhsBlock::Unpacked(mat) = &a {
        assert!(mat.rows().div_ceil(mr) >= row_tiles.end);
        assert!(mat.cols() >= depth_range.end);
    }

    // Loop over column tiles.
    col_tiles
        .enumerate()
        .for_each(|(block_col_tile, col_tile)| {
            let b_panel_offset = block_col_tile * b.panel_stride;
            let b_panel = &b.data[b_panel_offset..b_panel_offset + b.panel_stride];
            let b_quant_tile = b_quant.map(|bq| {
                let col_range = col_tile * nr..(col_tile * nr + nr).min(bq.zero_point.len());
                QuantParams {
                    zero_point: &bq.zero_point[col_range],
                }
            });

            // Loop over row tiles.
            for (block_row_tile, row_tile) in row_tiles.clone().enumerate() {
                // Safety:
                //  - The loops in this function and its caller are set up so that
                //    every output tile is processed by one thread at a time.
                let out_tile = unsafe { output.tile(row_tile, col_tile) };

                let a_quant_tile = a_quant.map(|aq| {
                    let row_range = row_tile * mr..(row_tile * mr + mr).min(aq.zero_point.len());
                    QuantParams {
                        zero_point: &aq.zero_point[row_range],
                    }
                });

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
                        a_quant_tile,
                        b_quant_tile,
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
mod reduced_range_rng;

#[cfg(test)]
pub use reduced_range_rng::ReducedRangeRng;

#[cfg(test)]
mod tests;

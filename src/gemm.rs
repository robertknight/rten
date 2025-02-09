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
use kernels::Kernel;
pub use kernels::QuantParams;
use packing::PackingBuffer;
pub use prepack::{PackedAMatrix, PackedBMatrix};
use tiles::OutputTiles;

pub type GemmResult = Result<(), GemmError>;

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
        out_row_stride: usize,
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
            out_row_stride,
            a,
            b,
            alpha,
            beta,
            bias,
            a_quant,
            b_quant,
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
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
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
            a_quant,
            b_quant,
        )
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
fn gemv<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    a: NdTensorView<LhsT, 1>,
    b: Matrix<RhsT>,
    mut output_mat: MatrixMut<MaybeUninit<OutT>>,
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<OutT>>,
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
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

            let b_quant = b_quant.map(|bq| QuantParams {
                zero_point: &bq.zero_point[col_block.clone()],
            });

            for (k_block, a_block) in
                range_chunks(0..a_cols, k_block_size).zip(a_data.chunks(k_block_size))
            {
                let b_block = b.slice((k_block, col_block.clone()));
                kernel.gemv_kernel(
                    out_chunk,
                    a_block,
                    b_block,
                    alpha,
                    effective_beta,
                    a_quant,
                    b_quant,
                );

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
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
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
            a_quant,
            b_quant,
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
mod tests {
    use std::error::Error;
    use std::time::Instant;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{expect_equal, ApproxEq};
    use rten_tensor::{Matrix, MatrixLayout, MatrixMut, NdTensor, NdTensorView, RandomSource};

    use super::{
        BiasVector, ColOffsets, F32KernelType, GemmError, GemmExecutor, GemmInT, GemmInputA,
        GemmInputB, GemmOutT, Im2Col, QuantParams, ReducedRangeRng, RowOffsets, WithKernel,
    };

    /// Scale a possibly non-float value by a float.
    ///
    /// Used for scaling by alpha in `C = alpha * AB + beta * C`.
    trait MulFloat {
        fn mul_float(self, scale: f32) -> Self;
    }

    impl MulFloat for f32 {
        fn mul_float(self, scale: f32) -> Self {
            self * scale
        }
    }

    impl MulFloat for i32 {
        fn mul_float(self, scale: f32) -> Self {
            (self as f32 * scale) as Self
        }
    }

    /// Type that can be used as the output for the reference GEMM
    /// implementation.
    trait RefGemmOutT<LhsT, RhsT>:
        Default
        + GemmOutT
        + From<LhsT>
        + From<RhsT>
        + MulFloat
        + ApproxEq
        + std::fmt::Debug
        + std::ops::Sub<Output = Self>
    {
    }

    impl<LhsT, RhsT> RefGemmOutT<LhsT, RhsT> for f32
    where
        f32: From<LhsT>,
        f32: From<RhsT>,
    {
    }

    impl<LhsT, RhsT> RefGemmOutT<LhsT, RhsT> for i32
    where
        i32: From<LhsT>,
        i32: From<RhsT>,
    {
    }

    #[derive(Clone)]
    struct GemmOpts<'a, LhsT, RhsT, OutT> {
        alpha: f32,
        beta: OutT,
        bias: Option<BiasVector<'a, OutT>>,
        a_quant: Option<QuantParams<'a, LhsT>>,
        b_quant: Option<QuantParams<'a, RhsT>>,
    }

    impl<LhsT, RhsT, OutT: GemmOutT> Default for GemmOpts<'_, LhsT, RhsT, OutT> {
        fn default() -> Self {
            GemmOpts {
                alpha: 1.,
                beta: OutT::zero(),
                bias: None,
                a_quant: None,
                b_quant: None,
            }
        }
    }

    /// Reference implementation. This should produce the same results as the
    /// optimized GEMM, but small numerical differences will appear in problems
    /// with a large K dimension, due to the different ordering of
    /// floating-point operations.
    fn reference_gemm<LhsT, RhsT, OutT>(
        mut output: MatrixMut<OutT>,
        a: Matrix<LhsT>,
        b: Matrix<RhsT>,
        opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
    ) where
        LhsT: GemmInT,
        RhsT: GemmInT,
        OutT: RefGemmOutT<LhsT, RhsT>,
    {
        let GemmOpts {
            alpha,
            beta,
            bias,
            a_quant,
            b_quant,
        } = opts.unwrap_or_default();

        for r in 0..a.rows() {
            let a_zero = a_quant
                .as_ref()
                .map(|aq| OutT::from(aq.zero_point[r]))
                .unwrap_or(OutT::zero());
            for c in 0..b.cols() {
                let b_zero = b_quant
                    .as_ref()
                    .map(|bq| OutT::from(bq.zero_point[c]))
                    .unwrap_or(OutT::zero());

                let mut accum = OutT::zero();
                for k in 0..a.cols() {
                    let a_el = OutT::from(a[[r, k]]) - a_zero;
                    let b_el = OutT::from(b[[k, c]]) - b_zero;
                    accum = accum + a_el * b_el;
                }
                let bias = match bias {
                    Some(BiasVector::Row(b)) => b[c],
                    Some(BiasVector::Column(b)) => b[r],
                    None => OutT::zero(),
                };
                output[[r, c]] = accum.mul_float(alpha) + beta * output[[r, c]] + bias;
            }
        }
    }

    fn reference_matmul<LhsT, RhsT, OutT>(
        a: Matrix<LhsT>,
        b: Matrix<RhsT>,
        opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
    ) -> NdTensor<OutT, 2>
    where
        LhsT: GemmInT,
        RhsT: GemmInT,
        OutT: RefGemmOutT<LhsT, RhsT>,
    {
        if let Some(opts) = &opts {
            assert_eq!(
                opts.beta,
                OutT::zero(),
                "beta has no effect in `reference_matmul`"
            );
        }
        let mut output = NdTensor::full([a.rows(), b.cols()], OutT::zero());
        reference_gemm(output.view_mut(), a, b, opts);
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

    fn run_gemm<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
        mut output: MatrixMut<OutT>,
        a: Matrix<LhsT>,
        b: Matrix<RhsT>,
        opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
        gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
    ) -> super::GemmResult
    where
        GemmExecutor<LhsT, RhsT, OutT>: Default,
    {
        let out_row_stride = output.stride(0);
        let default_gemm = GemmExecutor::default();
        let gemm = gemm.unwrap_or(&default_gemm);
        let GemmOpts {
            alpha,
            beta,
            bias,
            a_quant,
            b_quant,
        } = opts.unwrap_or_default();

        gemm.gemm(
            output.data_mut().expect("expected contiguous input"),
            out_row_stride,
            GemmInputA::Unpacked(a),
            GemmInputB::Unpacked(b),
            alpha,
            beta,
            bias,
            a_quant,
            b_quant,
        )
    }

    fn run_matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT + Default>(
        a: Matrix<LhsT>,
        b: Matrix<RhsT>,
        opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
        gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
    ) -> Result<NdTensor<OutT, 2>, GemmError>
    where
        GemmExecutor<LhsT, RhsT, OutT>: Default,
    {
        let mut output = NdTensor::zeros([a.rows(), b.cols()]);
        run_gemm(output.view_mut(), a, b, opts, gemm)?;
        Ok(output)
    }

    /// Run a matmul with the reference and real implementations and verify
    /// the results match.
    fn run_compare_matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: RefGemmOutT<LhsT, RhsT>>(
        a: Matrix<LhsT>,
        b: Matrix<RhsT>,
        opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
        gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
    ) where
        GemmExecutor<LhsT, RhsT, OutT>: Default,
    {
        let result = run_matmul(a.view(), b.view(), opts.clone(), gemm).unwrap();
        let expected = reference_matmul(a.view(), b.view(), opts);
        expect_equal(&result, &expected).unwrap();
    }

    /// Return `GemmExecutor`s with all of the available kernels for the given
    /// input and output types.
    fn all_gemms<L, R, O>() -> impl Iterator<Item = GemmExecutor<L, R, O>>
    where
        L: GemmInT,
        R: GemmInT,
        O: GemmOutT,
        GemmExecutor<L, R, O>: WithKernel,
    {
        GemmExecutor::<L, R, O>::kernel_types()
            .into_iter()
            .filter_map(|kern_type| GemmExecutor::<L, R, O>::with_kernel(kern_type))
    }

    // Simplest possible test case for easy debugging.
    #[test]
    fn test_simple_gemm_f32() -> Result<(), Box<dyn Error>> {
        let a = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
        let b = NdTensor::from_data([2, 2], vec![5., 6., 7., 8.]);
        run_compare_matmul(a.view(), b.view(), None, None);
        run_compare_matmul(
            a.view(),
            b.view(),
            None,
            Some(&GemmExecutor::<f32>::with_kernel(F32KernelType::Generic).unwrap()),
        );
        Ok(())
    }

    #[test]
    fn test_simple_gemm_u8i8_i32() -> Result<(), Box<dyn Error>> {
        let a = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
        let b = NdTensor::from_data([2, 2], vec![5, 6, 7, 8]);
        run_compare_matmul::<u8, i8, i32>(a.view(), b.view(), None, None);
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
                None, // a_quant
                None, // b_quant
            );
            assert_eq!(result, Err(expected));
        }
    }

    /// Test a GEMM kernel using all square matrices up to a given size, plus
    /// various other "interesting" size combinations.
    fn test_gemm_various_input_sizes<LhsT, RhsT, OutT>(
        gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
        mut lhs_gen: Option<&mut dyn FnMut() -> LhsT>,
        mut rhs_gen: Option<&mut dyn FnMut() -> RhsT>,
    ) -> Result<(), Box<dyn Error>>
    where
        LhsT: GemmInT,
        RhsT: GemmInT,
        GemmExecutor<LhsT, RhsT, OutT>: Default,
        XorShiftRng: RandomSource<LhsT>,
        XorShiftRng: RandomSource<RhsT>,
        OutT: RefGemmOutT<LhsT, RhsT>,
    {
        // "Interesting" sizes for the row, column and depth dimensions of the
        // computation. These are chosen to cover cases that are less than,
        // equal to and above the tile/block sizes which the algorithm divides
        // the problem into along each dimension.
        //
        // This also covers the case where each dimension is a vector.
        let col_steps = [0, 1, 2, 4, 5, 8, 1024, 1025];
        let depth_steps = [0, 1, 2, 20, 256, 300];
        let row_steps = [0, 1, 2, 8, 10, 16, 64, 80];

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
            let a = if let Some(lhs_gen) = lhs_gen.as_mut() {
                NdTensor::<LhsT, 2>::from_simple_fn(lhs_size, lhs_gen)
            } else {
                NdTensor::<LhsT, 2>::rand(lhs_size, &mut rng)
            };
            let b = if let Some(rhs_gen) = rhs_gen.as_mut() {
                NdTensor::<RhsT, 2>::from_simple_fn(rhs_size, rhs_gen)
            } else {
                NdTensor::<RhsT, 2>::rand(rhs_size, &mut rng)
            };

            let result = run_matmul(a.view(), b.view(), None, gemm).unwrap();
            let expected = reference_matmul(a.view(), b.view(), None);

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

    #[test]
    fn test_gemm_f32() -> Result<(), Box<dyn Error>> {
        for gemm in all_gemms::<f32, f32, f32>() {
            test_gemm_various_input_sizes(Some(&gemm), None, None)?;
        }
        Ok(())
    }

    #[test]
    fn test_gemm_u8i8_i32() -> Result<(), Box<dyn Error>> {
        for gemm in all_gemms::<u8, i8, i32>() {
            let mut rng = ReducedRangeRng::new(gemm.may_saturate(), 1234);
            test_gemm_various_input_sizes(Some(&gemm), None, Some(&mut || rng.next()))?;
        }
        Ok(())
    }

    #[test]
    fn test_gemm_u8i8_i32_zero_point() {
        #[derive(Copy, Clone)]
        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            // Matrix-matrix
            Case { m: 5, n: 7, k: 10 },
            // Vector-matrix product.
            Case { m: 1, n: 5, k: 10 },
            Case { m: 1, n: 8, k: 4 },
            Case { m: 1, n: 16, k: 4 },
            // Vector-matrix product, K not a multiple of 4 (tile size used by
            // int8 dot product instructions).
            Case { m: 1, n: 1, k: 2 },
            // Vector-matrix, where n is large enough that work should be
            // divided into multiple blocks.
            Case {
                m: 1,
                n: 256,
                k: 10,
            },
        ];

        for gemm in all_gemms::<u8, i8, i32>() {
            let mut lhs_rng = XorShiftRng::new(1234);
            let mut rhs_rng = ReducedRangeRng::new(gemm.may_saturate(), 5678);

            for Case { m, n, k } in cases {
                let a = NdTensor::<u8, 2>::rand([m, k], &mut lhs_rng);
                let b = NdTensor::<i8, 2>::rand([k, n], &mut rhs_rng);

                let a_zero_point: Vec<_> = (0..a.rows()).map(|x| x as u8).collect();
                let b_zero_point: Vec<_> = (0..b.cols()).map(|x| x as i8).collect();
                let opts = Some(GemmOpts {
                    a_quant: Some(QuantParams {
                        zero_point: &a_zero_point,
                    }),
                    b_quant: Some(QuantParams {
                        zero_point: &b_zero_point,
                    }),
                    ..Default::default()
                });
                run_compare_matmul(a.view(), b.view(), opts, Some(&gemm));
            }
        }
    }

    #[test]
    fn test_gemm_u8i8_i32_invalid_zero_point() {
        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::<u8, 2>::rand([5, 10], &mut rng);
        let b = NdTensor::<i8, 2>::rand([10, 3], &mut rng);

        fn gemm_opts<'a>(
            a_zero_point: &'a [u8],
            b_zero_point: &'a [i8],
        ) -> GemmOpts<'a, u8, i8, i32> {
            GemmOpts {
                a_quant: Some(QuantParams {
                    zero_point: a_zero_point,
                }),
                b_quant: Some(QuantParams {
                    zero_point: b_zero_point,
                }),
                ..Default::default()
            }
        }
        let a_zero_point: Vec<_> = (0..a.rows()).map(|row| row as u8).collect();
        let b_zero_point: Vec<_> = (0..b.cols()).map(|col| col as i8).collect();

        // LHS zero point does not match LHS rows.
        let result = run_matmul(
            a.view(),
            b.view(),
            Some(gemm_opts(&[1, 2, 3], &b_zero_point)),
            None,
        );
        assert_eq!(result, Err(GemmError::WrongQuantParamSize));

        // RHS zero point does not match RHS columns.
        let result = run_matmul(
            a.view(),
            b.view(),
            Some(gemm_opts(&a_zero_point, &[1, 2, 3, 4])),
            None,
        );
        assert_eq!(result, Err(GemmError::WrongQuantParamSize));
    }

    #[test]
    fn test_gemm_transposed() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let mut a = NdTensor::<f32, 2>::rand([20, 30], &mut rng);
        let mut b = NdTensor::rand([10, 20], &mut rng);

        // Transpose the input matrices. This will alter their row and column
        // strides and shapes, but not re-order the data.
        a.permute([1, 0]);
        b.permute([1, 0]);

        run_compare_matmul(a.view(), b.view(), None, None);

        Ok(())
    }

    #[test]
    fn test_gemv_u8i8_i32_transposed() -> Result<(), Box<dyn Error>> {
        struct Case {
            n: usize,
            k: usize,
        }

        let cases = [
            // K multiple of 4
            Case { k: 8, n: 5 },
            // K not a multiple of 4
            Case { k: 2, n: 5 },
        ];

        for gemm in all_gemms::<u8, i8, i32>() {
            let mut lhs_rng = XorShiftRng::new(1234);
            let mut rhs_rng = ReducedRangeRng::new(gemm.may_saturate(), 5678);

            for &Case { k, n } in &cases {
                let a = NdTensor::<u8, 2>::rand([1, k], &mut lhs_rng);
                let mut b = NdTensor::<i8, 2>::rand([n, k], &mut rhs_rng);

                // Transpose the input B matrix. This will alter the row and column
                // strides and shapes, but not re-order the data.
                b.permute([1, 0]);

                run_compare_matmul(a.view(), b.view(), None, Some(&gemm));
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_alpha() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        let a = NdTensor::rand([10, 5], &mut rng);
        let b = NdTensor::rand([5, 15], &mut rng);

        for gemm in all_gemms::<f32, f32, f32>() {
            for alpha in [0.0, 0.5, 1.0, 2.0] {
                let opts = Some(GemmOpts {
                    alpha,
                    ..Default::default()
                });
                run_compare_matmul(a.view(), b.view(), opts.clone(), Some(&gemm));
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
            let a = NdTensor::rand([m, k], &mut rng);
            let b = NdTensor::rand([k, n], &mut rng);

            for gemm in all_gemms::<f32, f32, f32>() {
                for beta in [0.5, 1.0, 2.0] {
                    let mut result = NdTensor::rand([m, n], &mut rng);
                    let mut expected = result.clone();
                    let opts = Some(GemmOpts {
                        beta,
                        ..Default::default()
                    });

                    run_gemm(
                        result.view_mut(),
                        a.view(),
                        b.view(),
                        opts.clone(),
                        Some(&gemm),
                    )?;
                    reference_gemm(expected.view_mut(), a.view(), b.view(), opts);

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
            let a = NdTensor::rand([m, k], &mut rng);
            let b = NdTensor::rand([k, n], &mut rng);

            // Create output buffer with NANs. This will cause incorrect
            // output if the GEMM impl incorrectly does `C = beta * C * alpha *
            // AB` instead of `C = alpha * AB` where beta is zero.
            let mut result = NdTensor::full([m, n], f32::NAN);

            // Test alpha values for which we may have special cases (0, 1) and
            // the general case.
            for alpha in [0., 0.5, 1.] {
                let opts = Some(GemmOpts {
                    alpha,
                    ..Default::default()
                });
                run_gemm(result.view_mut(), a.view(), b.view(), opts.clone(), None)?;
                let expected = reference_matmul(a.view(), b.view(), opts);
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
            Case { m: 5, n: 7, k: 0 },
        ];

        for Case { m, n, k } in cases {
            let a = NdTensor::rand([m, k], &mut rng);
            let b = NdTensor::rand([k, n], &mut rng);

            // Column vector bias
            let bias: Vec<f32> = (0..a.rows()).map(|b| b as f32).collect();
            let opts = Some(GemmOpts {
                bias: Some(BiasVector::Column(&bias)),
                ..Default::default()
            });
            run_compare_matmul(a.view(), b.view(), opts, None);

            // Row vector bias
            let bias: Vec<f32> = (0..b.cols()).map(|b| b as f32).collect();
            let opts = Some(GemmOpts {
                bias: Some(BiasVector::Row(&bias)),
                ..Default::default()
            });
            run_compare_matmul(a.view(), b.view(), opts, None);
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
            let a = NdTensor::rand([m, k], &mut rng);
            let b = NdTensor::rand([k, n], &mut rng);

            let gemm = GemmExecutor::new();

            let packed_a = gemm.prepack_a(a.view());
            let packed_b = gemm.prepack_b(b.view());

            let mut result = NdTensor::zeros([m, n]);
            let result_row_stride = result.stride(0);

            gemm.gemm(
                result.data_mut().unwrap(),
                result_row_stride,
                GemmInputA::Packed(&packed_a),
                GemmInputB::Packed(&packed_b),
                1.,   // alpha
                1.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();

            // Compare the results of pre-packed GEMM to unpacked GEMM rather
            // than reference GEMM because a) it is faster for large inputs
            // and b) in the case where K is large, the accumulated numerical
            // differences will be smaller.
            let mut expected = NdTensor::zeros(result.shape());
            let expected_row_stride = expected.stride(0);
            gemm.gemm(
                expected.data_mut().unwrap(),
                expected_row_stride,
                GemmInputA::Unpacked(a.view()),
                GemmInputB::Unpacked(b.view()),
                1.,   // alpha
                1.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
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
    fn build_im2col<T: Copy>(
        image: NdTensorView<T, 3>,
        col_count_step: usize,
        row_count_step: usize,
    ) -> Im2Col<T> {
        let [chans, img_h, img_w] = image.shape();
        let [chan_stride, h_stride, w_stride] = image.strides();

        let n_cols = img_w * img_h;
        let n_cols_padded = n_cols.next_multiple_of(col_count_step);

        let rows = chans;
        let n_rows_padded = rows.next_multiple_of(row_count_step);

        let mut row_offsets = RowOffsets {
            chan: (0..rows as i32)
                .map(|chan| chan * chan_stride as i32)
                .collect(),
            y: vec![0; rows],
            x: vec![0; rows],
        };

        for _ in rows..n_rows_padded {
            row_offsets.chan.push(i32::MAX);
            row_offsets.x.push(i32::MAX);
            row_offsets.y.push(i32::MAX);
        }

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
            col_offsets.y.push(i32::MAX);
            col_offsets.x.push(i32::MAX);
        }

        let max_y_offset = (img_h - 1) * h_stride;
        let max_x_offset = (img_w - 1) * w_stride;

        Im2Col {
            image,
            row_offsets,
            col_offsets,
            n_cols,
            n_rows: rows,
            max_y_offset: max_y_offset as i32,
            max_x_offset: max_x_offset as i32,
        }
    }

    #[test]
    fn test_gemm_im2col_f32() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let gemm = GemmExecutor::default();

        // nb. If the test fails, debug by setting dimensions to 1.
        let img_h = 2;
        let img_w = 2;
        let img_chans = 2;
        let kernel_chans = 3;

        let img = NdTensor::<f32, 3>::rand([img_chans, img_h, img_w], &mut rng);
        let im2col = build_im2col(
            img.view(),
            gemm.im2col_col_count_step(),
            gemm.im2col_row_count_step(),
        );

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
            None, // a_quant
            None, // b_quant
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
    fn test_gemm_im2col_u8i8_i32() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);

        // nb. If the test fails, debug by setting dimensions to 1.
        let img_h = 2;
        let img_w = 2;
        let img_chans = 2;
        let kernel_chans = 3;

        let img = NdTensor::<i8, 3>::rand([img_chans, img_h, img_w], &mut rng);

        for gemm in all_gemms() {
            let im2col = build_im2col(
                img.view(),
                gemm.im2col_col_count_step(),
                gemm.im2col_row_count_step(),
            );
            let kernel_mat = NdTensor::<u8, 2>::rand([kernel_chans, img_chans], &mut rng);
            let mut output_mat = NdTensor::<i32, 2>::zeros([kernel_chans, img_h * img_w]);
            let out_row_stride = output_mat.row_stride();

            gemm.gemm(
                output_mat.data_mut().unwrap(),
                out_row_stride,
                GemmInputA::Unpacked(kernel_mat.view()),
                GemmInputB::Im2Col(&im2col),
                1.,   // alpha
                0,    // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();

            let mut expected = NdTensor::<i32, 2>::zeros([kernel_chans, im2col.cols()]);
            for i in 0..expected.rows() {
                for j in 0..expected.cols() {
                    let mut acc = 0;
                    for k in 0..kernel_mat.cols() {
                        acc += kernel_mat[[i, k]] as i32 * img[[k, j / img_w, j % img_w]] as i32;
                    }
                    expected[[i, j]] = acc;
                }
            }
            expect_equal(&output_mat, &expected)?;
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
            let a = NdTensor::rand([1, k], &mut rng);
            let mut b = NdTensor::rand([k, n], &mut rng);
            match b_strides {
                Strides::Contiguous => {}
                Strides::Transposed => {
                    b.transpose();
                }
                Strides::Other => {
                    b = NdTensor::from_data_with_strides([k, n / 2], b.to_vec(), [b.stride(0), 2])
                        .unwrap();
                }
            }

            let mut result = NdTensor::zeros([1, b.size(1)]);
            let bias_array = bias.map(|b| [b]);
            let opts = Some(GemmOpts {
                alpha,
                beta,
                bias: bias_array
                    .as_ref()
                    .map(|b| BiasVector::Column(b.as_slice())),
                ..Default::default()
            });

            run_gemm(result.view_mut(), a.view(), b.view(), opts.clone(), None).unwrap();

            let mut expected = NdTensor::zeros([1, b.size(1)]);
            reference_gemm(expected.view_mut(), a.view(), b.view(), opts);

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

    fn run_gemm_bench<LhsT, RhsT, OutT>(cases: &[BenchCase], format: Format)
    where
        GemmExecutor<LhsT, RhsT, OutT>: Default,
        LhsT: GemmInT,
        RhsT: GemmInT,
        OutT: GemmOutT + Default,
        XorShiftRng: RandomSource<LhsT>,
        XorShiftRng: RandomSource<RhsT>,
    {
        let gemm = GemmExecutor::<LhsT, RhsT, OutT>::default();
        println!("Testing kernel {}", gemm.kernel_name());

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
            let mut result = NdTensor::<OutT, 2>::zeros([m, n]);
            let a = NdTensor::<LhsT, 2>::rand([m, k], &mut rng);
            let b = if transpose_b {
                let mut b = NdTensor::<RhsT, 2>::rand([n, k], &mut rng);
                b.transpose();
                b
            } else {
                NdTensor::<RhsT, 2>::rand([k, n], &mut rng)
            };

            let start = Instant::now();
            for _i in 0..iters {
                run_gemm(result.view_mut(), a.view(), b.view(), None, Some(&gemm)).unwrap();
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

            let flops =
                (2 * m as u64 * n as u64 * k as u64 * iters as u64) as f32 / duration.as_secs_f32();
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

        println!("f32 x f32 -> f32");
        run_gemm_bench::<f32, f32, f32>(&cases, Format::Pretty);

        println!("u8 x i8 -> i32");
        run_gemm_bench::<u8, i8, i32>(&cases, Format::Pretty);
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
        run_gemm_bench::<f32, f32, f32>(&cases, Format::Csv);
    }

    #[test]
    #[ignore]
    fn bench_prepack_a() {
        let gemm = GemmExecutor::<f32>::new();
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
}

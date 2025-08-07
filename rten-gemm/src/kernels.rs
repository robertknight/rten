use std::mem::MaybeUninit;
use std::ops::Range;

use super::{GemmOutT, Im2Col};
use rten_tensor::Matrix;

pub mod generic;
mod simd_generic;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod wasm;

/// LHS / A input to a GEMM kernel.
#[derive(Clone, Copy)]
pub enum Lhs<'a, T> {
    /// Input packed by the kernel's [`pack_a_block`](Kernel::pack_a_block)
    /// impl.
    Packed(&'a [u8]),

    /// Unpacked input with a column stride of 1 and row stride of `row_stride`.
    ///
    /// `data` is a pointer to the first element needed to compute the output
    /// tile. `(data, len)` is not passed as a slice because it is a pointer to
    /// part of a larger tensor and there may be mutable references in existence
    /// to other parts of that tensor.
    Unpacked {
        data: *const T,
        len: usize,
        row_stride: usize,
    },
}

/// Layout metadata for a packed block of an input matrix.
///
/// The packed block is expected to be organized as a sequence of panels with
/// stride [`panel_stride`](PackedInfo::panel_stride), where each panel contains
/// elements and associated metadata for an `MR x KC` (for packed LHS) or
/// `KC x NR` (for packed RHS) block of the input. The kernel is free to choose
/// the layout within each panel.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedLayout {
    /// Size required for a packed block.
    size: usize,

    /// Minimum alignment required for a packed block.
    align: usize,

    /// The stride between panels in the packed block.
    panel_stride: usize,

    /// True if the input must be packed to be used by the kernel.
    pub must_pack: bool,
}

impl PackedLayout {
    /// Construct a new packing buffer descriptor.
    ///
    /// `size`, `align` and `panel_stride` specify the minimum size of the
    /// packing buffer, its alignment and the stride between panels
    /// respectively. All units are in bytes. The size must be a multiple of
    /// both the alignment and panel stride.
    pub fn new(size: usize, align: usize, panel_stride: usize) -> PackedLayout {
        debug_assert_eq!(size % align, 0);
        debug_assert_eq!(size % panel_stride, 0);

        PackedLayout {
            size,
            align,
            panel_stride,
            must_pack: false,
        }
    }

    /// Return size of the packed block in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return minimum alignment of the packed block.
    pub fn align(&self) -> usize {
        self.align
    }

    /// Return stride between panels in bytes.
    pub fn panel_stride(&self) -> usize {
        self.panel_stride
    }
}

/// Parameters required to perform matrix multiplication on quantized tensors.
#[derive(Debug)]
pub struct QuantParams<'a, T> {
    /// Values that correspond to zero in each row (for LHS inputs) or column
    /// (for RHS inputs).
    pub zero_point: &'a [T],
}

// Make QuantParams Copy/Clone regardless of `T`.
impl<T> Clone for QuantParams<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for QuantParams<'_, T> {}

/// Kernel that computes a small tile of a general matrix multiplication (GEMM)
/// or general matrix-vector multiplication (GEMV).
///
/// This trait is an interface for the architecture-specific inner loop for
/// matrix multiplication and matrix-vector multiplication, as well as the
/// methods that pack the input matrices into a format that is efficient for the
/// kernel to use.
///
/// # Tile size selection
///
/// For a typical f32 kernel using FMA instructions (eg. AVX2), the tile size is
/// chosen such that an `MR x NR` tile of the output fits in registers. Each
/// iteration over the K dimension accumulates into this tile. Additionally one
/// of the dimensions (usually NR) is a multiple of the vector size and the
/// other is large enough such that enough cycles elapse between one update to
/// an accumulator register and the next that it doesn't encounter a delay
/// waiting for the previous update to complete. There is a small overhead for
/// each call into the kernel, so making tiles larger can improve performance by
/// reducing the overall number of tiles that need to be processed. See [^1]
/// for more details.
///
/// # Safety
///
/// - It must only be possible to construct the kernel using `new` if the
///   instructions it uses are supported on the current system.
/// - Kernels must initialize all output elements when `beta` is zero.
///
/// [^1]: https://dl.acm.org/doi/pdf/10.1145/2925987
pub unsafe trait Kernel<LhsT, RhsT, OutT>: Sync {
    /// Construct a new instance of this kernel, if supported on the current
    /// system.
    fn new() -> Option<Self>
    where
        Self: Sized;

    /// Return the number of rows in each tile.
    fn mr(&self) -> usize;

    /// Return the number of columns in each tile.
    fn nr(&self) -> usize;

    /// Return a name for this kernel for use in logging etc.
    ///
    /// The naming convention is `{arch}-{dtypes}-{variant}` where `dtypes`
    /// is either a triple of `{lhs}{rhs}{out}` if the LHS, RHS and output types
    /// are different, or just the type if all are the same. `variant` refers to
    /// target features being used (eg. "dotprod") or variants.
    fn name(&self) -> &'static str;

    /// Return true if this kernel may encounter saturation in a data type that
    /// is smaller than the accumulator.
    ///
    /// The caller will have to prepare inputs (usually the weights) to avoid
    /// this. This is primarily an issue for x64 systems without VNNI.
    /// See https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html.
    fn may_saturate(&self) -> bool {
        false
    }

    /// Step size used when packing an image usage [`pack_im2col`](Kernel::pack_im2col).
    ///
    /// This should match the K tile size used when packing the RHS / B matrix.
    /// For example, if RHS packing pads the K dimension to a multiple of 4,
    /// then `im2col_row_count_step` would also be 4.
    ///
    /// The length of the offset arrays in [`Im2Col::row_offsets`] must be a
    /// multiple of this.
    fn im2col_row_count_step(&self) -> usize {
        1
    }

    /// Step size used when packing an image usage [`pack_im2col`](Kernel::pack_im2col).
    ///
    /// The length of the offset arrays in [`Im2Col::col_offsets`] must be a
    /// multiple of this.
    fn im2col_col_count_step(&self) -> usize {
        self.nr()
    }

    /// Return the layout of a packing buffer required to pack an A / LHS input.
    fn packed_a_layout(
        &self,
        a: Matrix<LhsT>,
        rows: usize,
        cols: usize,
        quant: Option<QuantParams<LhsT>>,
    ) -> PackedLayout;

    /// Pack a block of the LHS / "A" input for use by this kernel.
    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix<LhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
        quant: Option<QuantParams<LhsT>>,
    );

    /// Return the layout of a packing buffer required to pack a block of a "B"
    /// / RHS input of size `rows x cols`.
    ///
    /// Unlike `packed_a_layout` this doesn't take the matrix as an argument.
    /// `packed_a_layout` may use this to indicate that the A input does not
    /// need to be packed. For the B input it is assumed this is always packed.
    fn packed_b_layout(
        &self,
        rows: usize,
        cols: usize,
        quant: Option<QuantParams<RhsT>>,
    ) -> PackedLayout;

    /// Pack a block of the RHS / "B" input for use by this kernel.
    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix<RhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
        quant: Option<QuantParams<RhsT>>,
    );

    /// Pack a block of an image as the B input for use by this kernel, using
    /// an im2col transformation to flatten the image into a matrix.
    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<RhsT>,
        rows: Range<usize>,
        cols: Range<usize>,
        zero_point: Option<RhsT>,
    );

    /// Compute a tile of the output matrix.
    ///
    /// The output is stored in row-major order with `used_rows` rows and `NR`
    /// columns, a row stride of `tile_row_stride` and column stride of 1. The
    /// maximum size of the tile will be `MR` rows and `NR` columns.
    ///
    /// The `a` input is either an unpacked matrix with a column stride of 1 or
    /// a buffer packed with `pack_a_block`. The `b` input is the RHS matrix
    /// packed with `pack_b_block`.
    ///
    /// `depth` specifies the number of columns of A and rows B that should be
    /// summed over to compute the output tile.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `tile_ptr` points to a buffer of the correct
    /// size. If `beta` is zero then the output may be uninitialized and must
    /// not be read by the implementation. If `beta` is non-zero then the output
    /// must be initialized and the implementation will read from it.
    unsafe fn kernel(
        &self,
        tile_ptr: *mut OutT,
        tile_row_stride: usize,
        a: Lhs<LhsT>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: OutT,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    );

    /// Compute an output block of a vector-matrix product ("gemv").
    ///
    /// This computes `y = alpha * (a B) + beta * y` where `a` is a row vector
    /// and `B` is a matrix.
    ///
    /// This is a simplified version of the matrix multiplication kernel that
    /// operates on unpacked data, since the overhead of packing outweighs the
    /// benefits for this operation.
    ///
    /// The length of vector `a` must match `b.rows()` and the length of `out`
    /// must match `b.cols()`. The `b` matrix must have a column stride of 1.
    ///
    /// # Safety
    ///
    /// If `beta` is zero then the output may be uninitialized and must not be
    /// read by the implementation. After the kernel has run, all elements will
    /// be initialized.
    fn gemv_kernel(
        &self,
        out: MatVecOutput<OutT>,
        a: &[LhsT],
        b: Matrix<RhsT>,
        alpha: f32,
        a_quant: Option<QuantParams<LhsT>>,
        b_quant: Option<QuantParams<RhsT>>,
    );
}

/// A temporary stack-allocated tile used by some kernels to handle edge tiles
/// which are smaller than the kernel's (MR, NR) size.
///
/// The kernel first allocates a [`TempTile`] and accumulates into it, then
/// calls [`accumulate_into`] to add the results to the matrix multiplication
/// output, for rows and columns that are actually valid.
///
/// The temporary tile is uninitialized, so callers must always set `beta=0`
/// when running the kernel on it. The row stride of the temporary tile is
/// always `NR`.
pub struct TempTile<T: GemmOutT, const MR: usize, const NR: usize> {
    data: [[MaybeUninit<T>; NR]; MR],
}

impl<T: GemmOutT, const MR: usize, const NR: usize> TempTile<T, MR, NR> {
    // Create an uninitialized temporary tile.
    //
    // Since the data is uninitialized, the only cost is reserving stack space.
    pub fn new() -> Self {
        TempTile {
            data: [[MaybeUninit::<T>::uninit(); NR]; MR],
        }
    }

    /// Return a pointer to the temporary tile data.
    pub fn as_mut_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.data.as_mut_ptr() as *mut MaybeUninit<T>
    }

    /// Update the first `n_rows` rows and `n_cols` columns of the tile pointed
    /// to by `dest` with the values from this temporary tile.
    ///
    /// If `beta` is zero, `dest` is treated as uninitialized. Otherwise
    /// it must be initialized.
    pub unsafe fn accumulate_into(
        &self,
        dest: *mut MaybeUninit<T>,
        n_rows: usize,
        n_cols: usize,
        row_stride: usize,
        beta: T,
    ) {
        if beta != T::zero() {
            // Accumulate into initialized output.
            for i in 0..n_rows {
                for j in 0..n_cols {
                    // Safety: Row and column indices are < `n_rows`/`n_cols`
                    unsafe {
                        let out_el = dest.add(row_stride * i + j);
                        let tmp = (*out_el).assume_init();
                        out_el.write(MaybeUninit::new(
                            beta * tmp + self.data.get_unchecked(i).get_unchecked(j).assume_init(),
                        ));
                    }
                }
            }
        } else {
            // Copy into possible uninitialized output.
            for i in 0..n_rows {
                for j in 0..n_cols {
                    // Safety: Row and column indices are < `n_rows`/`n_cols`
                    unsafe {
                        let out_el = dest.add(row_stride * i + j);
                        out_el.write(MaybeUninit::new(
                            self.data.get_unchecked(i).get_unchecked(j).assume_init(),
                        ));
                    }
                }
            }
        }
    }
}

/// Trait for computing dot products of SIMD vectors containing 8-bit integers.
///
/// This should use native instructions where available (VNNI on x64, UDOT/SDOT
/// on Arm etc.) or a fallback otherwise.
///
/// # Safety
///
/// Types implementing this trait must ensure they can only be constructed if
/// the instructions are supported.
unsafe trait Int8DotProduct {
    /// SIMD vector with `i8` or `u8` elements. The signed-ness depends on the
    /// implementation.
    type X8;

    /// SIMD vector with `i32` elements.
    type I32;

    /// Return true if [`indexed_dot_product`](Int8DotProduct::indexed_dot_product)
    /// is supported.
    fn supports_indexed_dot_product() -> bool {
        false
    }

    /// Compute the dot product of groups of 4 8-bit integers in `a` and `b`,
    /// accumulating into 32-bit integers in `c`.
    ///
    /// The signed-ness of `a` and `b` depends on the implementation.
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32;

    /// Broadcast 4 8-bit integers from a 32-bit lane of `b` selected by `IDX`
    /// and then compute the dot product as with
    /// [`dot_product`](Int8DotProduct::dot_product).
    #[allow(unused_variables)]
    fn indexed_dot_product<const IDX: u32>(
        self,
        a: Self::X8,
        b: Self::X8,
        c: Self::I32,
    ) -> Self::I32
    where
        Self: Sized,
    {
        unimplemented!("indexed_dot_product not supported")
    }
}

/// Output for matrix-vector multiplication.
///
/// This consists of:
///
/// - An output data slice
/// - An accumulation scale factor (`beta`), used as `C = beta * C + AB`. If
///   `beta` is zero, the output data may be uninitialized.
pub struct MatVecOutput<'a, T, B = T> {
    data: &'a mut [MaybeUninit<T>],
    beta: B,
}

impl<'a, T, B: Copy + Default + PartialEq> MatVecOutput<'a, T, B> {
    /// Create matrix-vector product output from an uninitialized slice.
    ///
    /// `beta` is implicitly set to zero.
    pub fn from_uninit_slice(data: &'a mut [MaybeUninit<T>]) -> Self {
        MatVecOutput {
            data,
            beta: B::default(),
        }
    }

    /// Create matrix-vector product output from a slice and beta value.
    pub fn from_slice(data: &'a mut [T], beta: B) -> Self {
        let data = unsafe { std::mem::transmute::<&'a mut [T], &'a mut [MaybeUninit<T>]>(data) };
        MatVecOutput { data, beta }
    }

    pub fn slice_mut(&mut self, range: Range<usize>) -> MatVecOutput<'_, T, B> {
        MatVecOutput {
            data: &mut self.data[range],
            beta: self.beta,
        }
    }

    /// Convert the accumulation scale (beta) to a boolean.
    ///
    /// This effectively changes the operation from `C = C * beta + AB` to
    /// `C = AB` or `C += AB` depending on the value of beta.
    pub fn as_bool_beta(&mut self) -> MatVecOutput<'_, T, bool> {
        MatVecOutput {
            data: self.data,
            beta: self.beta != B::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{AssumeInit, MatrixLayout, NdTensor, RandomSource};

    use super::{Kernel, Lhs};
    use crate::PackingBuffer;

    /// Benchmark a kernel with pre-packed input.
    ///
    /// Compared to the GEMM tests this eliminates the impact of threading,
    /// packing and transferring data between main memory/L3/L2 and L1.
    fn run_kernel_bench<LhsT: Clone + Default, RhsT: Clone + Default, OutT: Clone + Default>(
        kernel: &dyn Kernel<LhsT, RhsT, OutT>,
    ) where
        XorShiftRng: RandomSource<LhsT> + RandomSource<RhsT>,
    {
        let mut output = NdTensor::zeros([kernel.mr(), kernel.nr()]);

        // Choose a target size so that the panel of A, panel of B and output
        // all fit in the L1 cache.
        //
        // We use 24KB here on the assumption that the L1 cache will be 32KB or
        // larger. Making the value larger can lead to higher results because
        // the kernel will spend more time in the main loop over K.
        let target_size = 24 * 1024;

        let m = kernel.mr();
        let n = kernel.nr();

        // Packed LHS panel size is MR * K * lhs_elem_size. Packed RHS panel
        // size is NR * K * rhs_elem_size.
        //
        // Here we assume the packed elements have the same size as the input
        // matrices. This is true for f32 and int8 kernels using dot product
        // instructions.
        let k = target_size / (m * size_of::<LhsT>() + n * size_of::<RhsT>());

        // Create fixed inputs and prepack them.
        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::rand([m, k], &mut rng);
        let b = NdTensor::rand([k, n], &mut rng);

        let a_layout = kernel.packed_a_layout(a.view(), a.rows(), a.cols(), None);
        let b_layout = kernel.packed_b_layout(b.rows(), b.cols(), None);

        let mut packed_a_buf = PackingBuffer::new();
        let packed_a = packed_a_buf.alloc(a_layout.size(), a_layout.align());
        kernel.pack_a_block(packed_a, a.view(), 0..a.rows(), 0..a.cols(), None);
        let packed_a = unsafe { packed_a.assume_init() };

        let mut packed_b_buf = PackingBuffer::new();
        let packed_b = packed_b_buf.alloc(b_layout.size(), b_layout.align());
        kernel.pack_b_block(packed_b, b.view(), 0..b.rows(), 0..b.cols(), None);
        let packed_b = unsafe { packed_b.assume_init() };

        let n_iters = 1_000_000;
        let start = std::time::Instant::now();

        for _ in 0..n_iters {
            // Safety: Output has size `mr` * `nr`.
            unsafe {
                let out_row_stride = output.stride(0);
                kernel.kernel(
                    output.data_mut().unwrap().as_mut_ptr(),
                    out_row_stride,
                    Lhs::Packed(packed_a),
                    packed_b,
                    kernel.mr(),
                    kernel.nr(),
                    k,
                    1.,              /* alpha */
                    OutT::default(), // beta
                    None,            // a_quant
                    None,            // b_quant
                );
            }
            // Zero output after each iteration to avoid the possibility of
            // overflow.
            output.fill(OutT::default());
        }

        let elapsed = start.elapsed().as_secs_f64();
        let n_ops: u64 = 2 * m as u64 * n as u64 * k as u64 * n_iters as u64;
        let gflops = (n_ops as f64 / 1e9) / elapsed;

        println!(
            "{}: {} iters in {:.4}s = {:.2} GFLOPS",
            kernel.name(),
            n_iters,
            elapsed,
            gflops
        );
    }

    struct KernelBench<LhsT, RhsT, OutT> {
        kernels: Vec<Box<dyn Kernel<LhsT, RhsT, OutT>>>,
    }

    impl<LhsT, RhsT, OutT> KernelBench<LhsT, RhsT, OutT> {
        fn new() -> Self {
            Self {
                kernels: Vec::new(),
            }
        }

        /// Register a kernel to benchmark, if supported on the current system.
        fn add<K: Kernel<LhsT, RhsT, OutT> + 'static>(&mut self) {
            if let Some(kernel) = K::new() {
                self.kernels.push(Box::new(kernel))
            }
        }

        /// Run benchmarks for all registered kernels.
        ///
        /// Set the `RTEN_BENCH_FILTER` environment variable to filter tests
        /// by kernel name.
        fn run_bench(&self)
        where
            LhsT: Clone + Default,
            RhsT: Clone + Default,
            OutT: Clone + Default,
            XorShiftRng: RandomSource<RhsT> + RandomSource<LhsT>,
        {
            let filter = std::env::var("RTEN_BENCH_FILTER").ok();

            for kernel in &self.kernels {
                let filter_match = filter
                    .as_ref()
                    .map(|f| kernel.name().contains(f))
                    .unwrap_or(true);
                if !filter_match {
                    continue;
                }

                run_kernel_bench(kernel.as_ref())
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_kernel_f32() {
        let mut kernels = KernelBench::<f32, f32, f32>::new();

        kernels.add::<super::generic::GenericKernel>();

        #[cfg(target_arch = "aarch64")]
        {
            kernels.add::<super::aarch64::ArmNeonKernel>();
        }

        #[cfg(target_arch = "x86_64")]
        {
            kernels.add::<super::x86_64::FmaKernel>();
            kernels.add::<super::x86_64::Avx512Kernel>();
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            kernels.add::<super::wasm::WasmKernel>();
        }

        kernels.run_bench();
    }

    #[test]
    #[ignore]
    fn bench_kernel_int8() {
        let mut kernels = KernelBench::<u8, i8, i32>::new();

        kernels.add::<super::generic::GenericKernel>();

        #[cfg(target_arch = "aarch64")]
        {
            kernels.add::<super::aarch64::ArmInt8MlalKernel>();
            kernels.add::<super::aarch64::ArmInt8DotKernel>();
            kernels.add::<super::aarch64::ArmInt8MMKernel>();
        }

        #[cfg(target_arch = "x86_64")]
        {
            kernels.add::<super::x86_64::Avx2Int8Kernel>();
            kernels.add::<super::x86_64::Avx512Int8Kernel>();
        }

        #[cfg(target_arch = "wasm32")]
        #[cfg(target_feature = "simd128")]
        {
            kernels.add::<super::wasm::WasmInt8Kernel>();
        }

        kernels.run_bench();
    }
}

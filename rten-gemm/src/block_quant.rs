use rten_tensor::{Contiguous, Layout, NdTensorView};

use crate::errors::BlockQuantizedError;

/// Matrix which is quantized into blocks along the K dimension.
///
/// The data layout and supported bit/block sizes follow ONNX Runtime's MatMulNBits
/// operator. See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md>.
#[derive(Copy, Clone)]
pub struct BlockQuantizedMatrix<'a, T> {
    /// Quantized data of shape (N, k_blocks, block_size).
    quant: Contiguous<NdTensorView<'a, u8, 3>>,

    /// Scales of shape (N, k_blocks)
    scales: Contiguous<NdTensorView<'a, T, 2>>,

    /// Bits per quantized element.
    ///
    /// Must be a divisor of `block_size * 8`.
    bits: u8,
}

impl<'a, T: Copy> BlockQuantizedMatrix<'a, T> {
    /// Minimum supported number of elements per block.
    pub const MIN_BLOCK_SIZE: usize = 16;

    /// Create a block-quantized RHS matrix input.
    ///
    /// `quant` is the block-quantized input with shape (cols, k_blocks,
    /// block_size). `scales` are the per-block scales with shape (cols,
    /// k_blocks). `bits` specifies the number of bits per element in each
    /// block.
    ///
    /// The number of elements per quantized block must be a power of 2 of at
    /// least `MIN_BLOCK_SIZE`.
    pub fn new(
        quant: Contiguous<NdTensorView<'a, u8, 3>>,
        scales: Contiguous<NdTensorView<'a, T, 2>>,
        bits: u8,
    ) -> Result<Self, BlockQuantizedError> {
        // ONNX Runtime currently supports 2, 4 or 8 bits per element. These
        // values have the convenient property that a byte is a whole number
        // of elements. We only support 4 bits for the moment.
        if bits != 4 {
            return Err(BlockQuantizedError::UnsupportedElementSize);
        }
        let n_elem = 8 / bits;

        let [_batch, _k_blocks, block_bytes] = quant.shape();

        let block_size = block_bytes * n_elem as usize;
        if !block_size.is_power_of_two() || block_size < Self::MIN_BLOCK_SIZE {
            return Err(BlockQuantizedError::UnsupportedBlockSize);
        }

        Ok(Self {
            quant,
            scales,
            bits,
        })
    }

    /// Return the number of rows in the dequantized matrix.
    pub fn rows(&self) -> usize {
        self.blocks_per_column() * self.elements_per_block()
    }

    /// Return the number of columns in the dequantized matrix.
    pub fn cols(&self) -> usize {
        self.quant.size(0)
    }

    /// Return the number of bits per element
    pub fn n_bits(&self) -> u8 {
        self.bits
    }

    /// Return the number of blocks in each column.
    pub(crate) fn blocks_per_column(&self) -> usize {
        self.quant.size(1)
    }

    pub(crate) fn elements_per_block(&self) -> usize {
        (self.bytes_per_block() * 8) / self.bits as usize
    }

    pub(crate) fn bytes_per_block(&self) -> usize {
        self.quant.size(2)
    }

    /// Return the packed data for a range of blocks in a column.
    pub(crate) fn column_data(
        &self,
        col: usize,
        start_block: usize,
        n_blocks: usize,
    ) -> Option<&[u8]> {
        if self.quant.size(1) < start_block + n_blocks {
            return None;
        }
        let offset = self.quant.offset([col, start_block, 0])?;
        let len = self.quant.size(2);

        Some(unsafe {
            std::slice::from_raw_parts(self.quant.data_ptr().add(offset), len * n_blocks)
        })
    }

    /// Return the scale factors for a range of blocks in a column.
    pub(crate) fn column_scales(
        &self,
        col: usize,
        start_block: usize,
        n_blocks: usize,
    ) -> Option<&[T]> {
        if self.scales.size(1) < start_block + n_blocks {
            return None;
        }
        let offset = self.scales.offset([col, start_block])?;

        Some(unsafe { std::slice::from_raw_parts(self.scales.data_ptr().add(offset), n_blocks) })
    }
}

/// Return the default zero point for n-bit quantization.
///
/// This is an i16 because the maximum value is 128 (when n_bits=8) and the
/// value is used in signed subtractions.
///
/// See docs for `zero_points` input to MatMulNBits in
/// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md.
pub const fn nbit_zero_point(n_bits: u8) -> i16 {
    assert!(n_bits >= 2 && n_bits <= 8);
    1 << (n_bits - 1)
}

#[cfg(test)]
pub fn pack_4bit_elements(vals: &[i8], zero_point: i8) -> Vec<u8> {
    let (chunks, tail) = vals.as_chunks::<2>();
    assert!(tail.is_empty());
    chunks
        .iter()
        .copied()
        .map(|[even, odd]| {
            let lo = (even + zero_point) as u8;
            let hi = (odd + zero_point) as u8;
            (lo & 0x0F) | (hi << 4)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rten_tensor::{AsView, Contiguous, NdTensor, NdTensorView};

    use super::{BlockQuantizedMatrix, nbit_zero_point, pack_4bit_elements};

    #[test]
    fn test_block_quantized_matrix() {
        let zero_point = nbit_zero_point(4) as i8;
        let elems: Vec<i8> = (-8..8).cycle().take(256).collect();
        let packed = pack_4bit_elements(&elems, zero_point);

        let block_bytes = 16;
        let cols = 4;
        let k_blocks = 2;
        let n_bits = 4;

        let quants = NdTensor::from_data([cols, k_blocks, block_bytes], packed);
        let scales = NdTensorView::from_data([cols, k_blocks], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let mat = BlockQuantizedMatrix::new(
            Contiguous::new(quants.view()).unwrap(),
            Contiguous::new(scales.view()).unwrap(),
            n_bits,
        )
        .unwrap();

        assert_eq!(mat.rows(), 64);
        assert_eq!(mat.cols(), cols);
        assert_eq!(mat.elements_per_block(), 32);
        assert_eq!(mat.blocks_per_column(), k_blocks);

        assert_eq!(
            mat.column_data(0, 0, 1).unwrap(),
            pack_4bit_elements(&elems[..32], zero_point)
        );
        assert_eq!(mat.column_scales(0, 0, 2), Some([1.0, 2.0].as_slice()));
        assert_eq!(
            mat.column_data(3, 1, 1).unwrap(),
            pack_4bit_elements(&elems[256 - 32..], zero_point)
        );
        assert_eq!(mat.column_scales(3, 1, 1), Some([8.0].as_slice()));

        // Out of bounds column.
        assert_eq!(mat.column_data(4, 1, 1), None);
        // Out of bounds K block.
        assert_eq!(mat.column_data(3, 2, 1), None);
    }
}

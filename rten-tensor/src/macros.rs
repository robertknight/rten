/// Construct a `Tensor`.
///
/// ```
/// use rten_tensor::tensor;
///
/// // Create a scalar (0D tensor)
/// tensor!(2.);
///
/// // Create a vector (1D tensor)
/// tensor!([1, 2, 3]);
///
/// // Create a 3D tensor with shape [1, 2, 2] and elements [1, 2, 3, 4].
/// tensor!((1, 2, 2); [1, 2, 3, 4]);
/// ```
#[macro_export]
macro_rules! tensor {
    [[$($elem:expr),*]] => {
        {
            use $crate::Tensor;
            Tensor::from_vec(vec![$($elem),*])
        }
    };

    // As above, but with trailing comma.
    [[$($elem:expr),*,]] => {
        tensor!([$($elem),*])
    };

    (($($dim:expr),+); [$($elem:expr),*]) => {
        {
            use $crate::Tensor;
            Tensor::from_data(&[$($dim),+], vec![$($elem),*])
        }
    };

    // As above, but with trailing comma.
    (($($dim:expr),+); [$($elem:expr),*,]) => {
        {
            use $crate::Tensor;
            Tensor::from_data(&[$($dim),+], vec![$($elem),*])
        }
    };

    ($elem:expr) => {
        {
            use $crate::Tensor;
            Tensor::from_scalar($elem)
        }
    };
}

/// Construct an `NdTensor`.
///
/// ```
/// use rten_tensor::ndtensor;
///
/// // Create a scalar (0D tensor)
/// ndtensor!(2.);
///
/// // Create a vector (1D tensor)
/// ndtensor!([1, 2, 3]);
///
/// // Create a 3D tensor with shape [1, 2, 2] and elements [1, 2, 3, 4].
/// ndtensor!((1, 2, 2); [1, 2, 3, 4]);
/// ```
#[macro_export]
macro_rules! ndtensor {
    [[$($elem:expr),*]] => {
        {
            use $crate::NdTensor;
            let data = vec![$($elem),*];
            let len = data.len();
            NdTensor::from_data([len], data)
        }
    };

    // As above, but with trailing comma.
    [[$($elem:expr),*,]] => {
        ndtensor!([$($elem),*])
    };

    (($($dim:expr),+); [$($elem:expr),*]) => {
        {
            use $crate::NdTensor;
            let data = vec![$($elem),*];
            NdTensor::from_data([$($dim),+], data)
        }
    };

    // As above, but with trailing comma.
    (($($dim:expr),+); [$($elem:expr),*,]) => {
        {
            use $crate::NdTensor;
            let data = vec![$($elem),*];
            NdTensor::from_data([$($dim),+], data)
        }
    };

    ($elem:expr) => {
        {
            use $crate::NdTensor;
            NdTensor::from_data([], vec![$elem])
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{ndtensor, tensor, NdTensor, Tensor};

    #[test]
    fn test_tensor_scalar() {
        let x = tensor!(5.);
        assert_eq!(x, Tensor::from_scalar(5.));
    }

    #[test]
    fn test_tensor_vector() {
        let x = tensor!([1, 2, 3]);
        assert_eq!(x, Tensor::from_vec(vec![1, 2, 3]));
    }

    #[test]
    fn test_tensor_nd() {
        let x = tensor!((1, 2, 2); [1, 2, 3, 4]);
        assert_eq!(x, Tensor::from_data(&[1, 2, 2], vec![1, 2, 3, 4]));

        // As above, but with trailing comma in data
        let x = tensor!((1, 2, 2); [1, 2, 3, 4,]);
        assert_eq!(x, Tensor::from_data(&[1, 2, 2], vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_ndtensor_scalar() {
        let x = ndtensor!(5.);
        assert_eq!(x, NdTensor::from_data([], vec![5.]));
    }

    #[test]
    fn test_ndtensor_vector() {
        let x = ndtensor!([1, 2, 3]);
        assert_eq!(x, NdTensor::from_data([3], vec![1, 2, 3]));
    }

    #[test]
    fn test_ndtensor_nd() {
        let x = ndtensor!((1, 2, 2); [1, 2, 3, 4]);
        assert_eq!(x, NdTensor::from_data([1, 2, 2], vec![1, 2, 3, 4]));

        // As above, but with trailing comma in data
        let x = ndtensor!((1, 2, 2); [1, 2, 3, 4,]);
        assert_eq!(x, NdTensor::from_data([1, 2, 2], vec![1, 2, 3, 4]));
    }
}

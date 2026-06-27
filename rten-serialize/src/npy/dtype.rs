//! Mapping between Rust scalar types and NumPy array element types.

mod sealed {
    pub trait Sealed {}
}

/// NumPy array-protocol type kind: bool, signed integer, unsigned integer or
/// float.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    Bool,
    Int,
    Uint,
    Float,
}

impl ElementKind {
    /// Parse the single-character type code used in a NumPy dtype string, e.g.
    /// `i` for a signed integer.
    pub(crate) fn from_char(c: char) -> Option<ElementKind> {
        match c {
            'b' => Some(ElementKind::Bool),
            'i' => Some(ElementKind::Int),
            'u' => Some(ElementKind::Uint),
            'f' => Some(ElementKind::Float),
            _ => None,
        }
    }

    /// Return the single-character type code used in a NumPy dtype string.
    pub(crate) fn as_char(self) -> char {
        match self {
            ElementKind::Bool => 'b',
            ElementKind::Int => 'i',
            ElementKind::Uint => 'u',
            ElementKind::Float => 'f',
        }
    }
}

/// A scalar type that can be used as the element type of a NumPy array.
///
/// This is implemented for Rust's primitive integer and floating point types
/// and `bool`.
pub trait Element: sealed::Sealed + Copy {
    /// Size of the element in bytes.
    #[doc(hidden)]
    const ITEM_SIZE: usize;

    /// dtype string written to `.npy` headers, using little-endian byte order.
    #[doc(hidden)]
    const DESCR: &'static str;

    /// The little-endian byte representation of an element, `[u8; ITEM_SIZE]`.
    #[doc(hidden)]
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + for<'a> TryFrom<&'a [u8]>;

    /// Decode an element from its little-endian representation.
    #[doc(hidden)]
    fn from_le_bytes(bytes: Self::Bytes) -> Self;

    /// Encode an element as its little-endian representation.
    #[doc(hidden)]
    fn to_le_bytes(self) -> Self::Bytes;
}

macro_rules! impl_element {
    ($ty:ty, $descr:literal) => {
        impl sealed::Sealed for $ty {}

        impl Element for $ty {
            const ITEM_SIZE: usize = size_of::<$ty>();
            const DESCR: &'static str = $descr;

            type Bytes = [u8; size_of::<$ty>()];

            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                <$ty>::from_le_bytes(bytes)
            }

            fn to_le_bytes(self) -> Self::Bytes {
                <$ty>::to_le_bytes(self)
            }
        }
    };
}

// Single-byte types use the `|` ("not applicable") byte-order marker, matching
// the output of `numpy.dtype.str`.
impl_element!(i8, "|i1");
impl_element!(i16, "<i2");
impl_element!(i32, "<i4");
impl_element!(i64, "<i8");
impl_element!(u8, "|u1");
impl_element!(u16, "<u2");
impl_element!(u32, "<u4");
impl_element!(u64, "<u8");
impl_element!(f32, "<f4");
impl_element!(f64, "<f8");

impl sealed::Sealed for bool {}

impl Element for bool {
    const ITEM_SIZE: usize = 1;
    const DESCR: &'static str = "|b1";

    type Bytes = [u8; 1];

    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        bytes[0] != 0
    }

    fn to_le_bytes(self) -> Self::Bytes {
        [self as u8]
    }
}

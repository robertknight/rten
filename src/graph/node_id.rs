use std::num::NonZero;

/// ID of a node in a [`Model`](crate::Model) graph.
///
/// This is used to identify input and output values as well as internal nodes.
///
/// Node IDs are u32 values <= `i32::MAX`.
#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(NonZero<u32>);

impl NodeId {
    /// Return the underlying u32 value of the ID.
    pub fn as_u32(self) -> u32 {
        self.0.get() - 1
    }

    /// Return the underlying ID value as a usize, for slice indexing.
    pub fn as_usize(self) -> usize {
        self.as_u32() as usize
    }

    /// Construct a node ID from a u32 value.
    ///
    /// Panics if the value exceeds `i32::MAX`.
    pub fn from_u32(value: u32) -> NodeId {
        // Node IDs are limited to `i32::MAX` because the `OperatorNode` type
        // in the FlatBuffers schema represents operator input and output IDs
        // as `i32`. Negative values are used as a niche to represent missing
        // optional inputs.
        assert!(value <= i32::MAX as u32);

        // Valid node IDs are in the range `[0, i32::MAX]`, so we store them as
        // values in `[1, i32::MAX + 1]` internally and reserve 0 as a niche to
        // make `Option<NodeId>` the same size as `NodeId`.
        NodeId(unsafe {
            // Safety: `value + 1` cannot be zero
            NonZero::new_unchecked(value + 1)
        })
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_u32().fmt(f)
    }
}

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.as_u32())
    }
}

use rustc_hash::FxHashMap;

use crate::graph::NodeId;
use crate::operator::PrepackedInput;

/// Key identifying a prepacked operator input in a [`WeightCache`].
///
/// The key contains the prepacked input and its zero point (see
/// [`Operator::prepack_zero_point`]). Operators which use the same nodes for
/// these inputs share prepacked data.
///
/// [`Operator::prepack_zero_point`]: crate::operator::Operator::prepack_zero_point
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrepackKey {
    /// Index and node ID of the prepacked input.
    input: (u8, NodeId),

    /// Index and node ID of the zero point, if the prepacked data depends on
    /// one.
    zero_point: Option<(u8, NodeId)>,
}

impl PrepackKey {
    /// Create a key for a prepacked input, given the `(index, node_id)` of
    /// the input and of its zero point.
    ///
    /// Panics if an input index does not fit in a `u8`.
    pub fn new(input: (usize, NodeId), zero_point: Option<(usize, NodeId)>) -> Self {
        let to_u8 = |(index, node_id): (usize, NodeId)| {
            let index = u8::try_from(index).expect("input index should fit in u8");
            (index, node_id)
        };
        PrepackKey {
            input: to_u8(input),
            zero_point: zero_point.map(to_u8),
        }
    }

    /// Return the node ID of the prepacked input.
    pub fn input_id(&self) -> NodeId {
        self.input.1
    }

    /// Return the node ID of the prepacked input's zero point, if the
    /// prepacked data depends on one.
    pub fn zero_point_id(&self) -> Option<NodeId> {
        self.zero_point.map(|(_, id)| id)
    }
}

/// A cache of prepacked weights for graph operators.
///
/// The weight cache has a hierarchical structure which mirrors the model
/// graph. At the top level is the root graph. For each operator with a
/// subgraph (eg. control flow operators) there are separate sub-caches.
pub struct WeightCache {
    /// Map of prepacked operator input to prepacked weights.
    cache: FxHashMap<PrepackKey, PrepackedInput>,

    /// Map of operator ID to caches for the operator's subgraphs.
    subgraph_caches: FxHashMap<NodeId, Vec<WeightCache>>,
}

impl WeightCache {
    /// Create an empty cache.
    pub fn new() -> WeightCache {
        WeightCache {
            cache: FxHashMap::default(),
            subgraph_caches: FxHashMap::default(),
        }
    }

    /// Check if a pre-packed weight exists for a given key.
    pub fn contains(&self, key: &PrepackKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Add a prepacked weight to the cache.
    pub fn insert(&mut self, key: PrepackKey, packed: PrepackedInput) {
        self.cache.insert(key, packed);
    }

    /// Look up weight in the cache.
    pub fn get(&self, key: &PrepackKey) -> Option<&PrepackedInput> {
        self.cache.get(key)
    }

    /// Add caches for subgraphs belonging to an operator.
    pub fn insert_subgraph_caches(&mut self, operator_id: NodeId, caches: Vec<WeightCache>) {
        self.subgraph_caches.insert(operator_id, caches);
    }

    /// Look up caches for an operator's subgraphs.
    pub fn get_subgraph_caches(&self, operator_id: NodeId) -> Option<&[WeightCache]> {
        self.subgraph_caches
            .get(&operator_id)
            .map(|wcs| wcs.as_slice())
    }

    /// Return the total number of cached weights, including in subgraphs.
    pub fn len(&self) -> usize {
        self.cache.len()
            + self
                .subgraph_caches
                .values()
                .flat_map(|caches| caches.iter())
                .map(|cache| cache.len())
                .sum::<usize>()
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        WeightCache::new()
    }
}

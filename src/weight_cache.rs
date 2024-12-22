use rustc_hash::FxHashMap;

use crate::graph::NodeId;
use crate::ops::PrepackedInput;

/// A cache of prepacked weights for graph operators.
pub struct WeightCache {
    /// Map of constant node ID to prepacked weights.
    cache: FxHashMap<NodeId, PrepackedInput>,

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

    /// Check if a pre-packed weight exists for a given constant node ID.
    pub fn contains(&self, node: NodeId) -> bool {
        self.cache.contains_key(&node)
    }

    /// Add a prepacked weight to the cache.
    pub fn insert(&mut self, node: NodeId, packed: PrepackedInput) {
        self.cache.insert(node, packed);
    }

    /// Look up weight in the cache.
    pub fn get(&self, node: NodeId) -> Option<&PrepackedInput> {
        self.cache.get(&node)
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
}

impl Default for WeightCache {
    fn default() -> Self {
        WeightCache::new()
    }
}

use rustc_hash::FxHashMap;

use super::NodeId;
use crate::value::Value;

/// Map used to store operator outputs during graph execution.
pub struct ValueMap {
    values: FxHashMap<NodeId, Value>,

    /// Whether to update `current_bytes` and `max_bytes` when values are
    /// added and removed. This saves calls to `Value::bytes` when the
    /// information is not needed.
    enable_mem_profiling: bool,

    /// Combined size of all values currently stored in the map.
    current_bytes: usize,

    /// Maximum value of `current_bytes` at any point in time.
    max_bytes: usize,
}

impl ValueMap {
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            enable_mem_profiling: false,
            current_bytes: 0,
            max_bytes: 0,
        }
    }

    /// Set whether memory usage statistics are updated when values are added
    /// or removed from the map.
    pub fn enable_mem_profiling(&mut self, enable: bool) {
        self.enable_mem_profiling = enable
    }

    /// Add a value to the map and update memory usage stats.
    pub fn insert(&mut self, id: NodeId, value: Value) {
        if self.enable_mem_profiling {
            self.current_bytes += value.bytes();
        }

        let old_value = self.values.insert(id, value);

        if self.enable_mem_profiling {
            self.current_bytes -= old_value.map(|v| v.bytes()).unwrap_or(0);
            self.max_bytes = self.max_bytes.max(self.current_bytes);
        }
    }

    /// Remove a value from the map and update memory usage stats.
    pub fn remove(&mut self, id: NodeId) -> Option<Value> {
        let value = self.values.remove(&id)?;
        if self.enable_mem_profiling {
            self.current_bytes -= value.bytes();
        }
        Some(value)
    }

    pub fn get(&self, id: NodeId) -> Option<&Value> {
        self.values.get(&id)
    }

    /// Add all values from `iter` to the map.
    pub fn extend(&mut self, iter: impl Iterator<Item = (NodeId, Value)>) {
        for (id, value) in iter {
            self.insert(id, value)
        }
    }

    /// Return the peak combined size of all values in the map.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

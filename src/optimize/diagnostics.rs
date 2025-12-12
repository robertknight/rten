use std::cell::RefCell;

use rustc_hash::FxHashSet;

use crate::NodeId;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    Off,
    Warn,
}

/// Diagnostic reporter for graph optimizations.
pub struct Diagnostics {
    /// Nodes against which issues have been reported.
    nodes: RefCell<FxHashSet<NodeId>>,
    level: DiagnosticLevel,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(FxHashSet::default()),
            level: DiagnosticLevel::Off,
        }
    }

    /// Enable reporting of all messages at or above a given level.
    pub fn set_level(&mut self, level: DiagnosticLevel) {
        self.level = level;
    }

    /// Return true if diagnostic messages are enabled at a given level.
    pub fn enabled(&self, level: DiagnosticLevel) -> bool {
        self.level >= level
    }

    /// Log a diagnostic message for a given node at the [`Warn`](DiagnosticLevel::Warn) level.
    pub fn warn(&self, node: NodeId, message: std::fmt::Arguments<'_>) {
        if self.level < DiagnosticLevel::Warn || self.nodes.borrow().contains(&node) {
            return;
        }
        self.nodes.borrow_mut().insert(node);
        println!("{}", message);
    }
}

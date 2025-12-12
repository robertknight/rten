use std::cell::RefCell;

use rustc_hash::FxHashSet;

use crate::graph::{Graph, NodeId};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    /// Don't show any diagnostics.
    Off,
    /// Report only unsuccessful optimizations.
    Warn,
    /// Report all optimizations.
    Info,
}

/// Diagnostic reporter for graph optimizations.
pub struct Diagnostics {
    /// Nodes against which diagnostics have been reported at the `Warn` level
    /// or higher.
    warned_nodes: RefCell<FxHashSet<NodeId>>,
    level: DiagnosticLevel,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            warned_nodes: RefCell::new(FxHashSet::default()),
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

    /// Log a diagnostic message for a given node at the [`Info`](DiagnosticLevel::Info) level.
    pub fn info(&self, graph: &Graph, node: NodeId, message: std::fmt::Arguments<'_>) {
        if self.level < DiagnosticLevel::Info {
            return;
        }
        self.log(DiagnosticLevel::Info, graph, node, message);
    }

    /// Log a diagnostic message for a given node at the [`Warn`](DiagnosticLevel::Warn) level.
    pub fn warn(&self, graph: &Graph, node: NodeId, message: std::fmt::Arguments<'_>) {
        if self.level < DiagnosticLevel::Warn || self.warned_nodes.borrow().contains(&node) {
            return;
        }
        self.warned_nodes.borrow_mut().insert(node);
        self.log(DiagnosticLevel::Warn, graph, node, message);
    }

    fn log(
        &self,
        level: DiagnosticLevel,
        graph: &Graph,
        node: NodeId,
        message: std::fmt::Arguments<'_>,
    ) {
        let level_char = match level {
            DiagnosticLevel::Warn => 'W',
            DiagnosticLevel::Info => 'I',
            DiagnosticLevel::Off => unreachable!(),
        };
        println!(
            "{}| {}: {}",
            level_char,
            self.node_name(graph, node),
            message
        );
    }

    fn node_name<'a>(&self, g: &'a Graph, id: NodeId) -> &'a str {
        g.get_node(id).and_then(|n| n.name()).unwrap_or_default()
    }
}

//! Symbol name generator.

use std::borrow::Cow;

use crate::sym_tensor::{SymElem, Symbol};

/// Generates named symbols.
///
/// Sometimes during shape inference it may be necessary to generate a new
/// symbol to represent a value that cannot be represented as an expression.
///
/// Note that generally it is preferred to represent values computed from other
/// values as symbolic expressions. This allows the expressions to be compared,
/// simplified and otherwise manipulated.
pub struct SymbolGen {
    prefix: Cow<'static, str>,
    next_symbol_id: u32,
}

impl Default for SymbolGen {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolGen {
    pub fn new() -> Self {
        Self::with_prefix("unknown".into())
    }

    pub fn with_prefix(prefix: Cow<'static, str>) -> Self {
        Self {
            prefix,
            next_symbol_id: 0,
        }
    }

    fn gen_name(&mut self) -> String {
        self.next_symbol_id += 1;
        format!("{}_{}", self.prefix, self.next_symbol_id)
    }

    /// Generate a new symbolic value which is assumed to be positive.
    pub fn gen_positive(&mut self) -> SymElem {
        SymElem::Var(
            Symbol {
                name: self.gen_name(),
                positive: true,
            }
            .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{SymElem, SymbolGen};

    #[test]
    fn test_symbol_gen() {
        let mut sym_gen = SymbolGen::new();
        assert_eq!(sym_gen.gen_positive(), SymElem::pos_var("unknown_1"));
        assert_eq!(sym_gen.gen_positive(), SymElem::pos_var("unknown_2"));

        let mut sym_gen = SymbolGen::with_prefix("foo".into());
        assert_eq!(sym_gen.gen_positive(), SymElem::pos_var("foo_1"));
        assert_eq!(sym_gen.gen_positive(), SymElem::pos_var("foo_2"));
    }
}

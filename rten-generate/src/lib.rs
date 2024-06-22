//! Utilities to simplify running auto-regressive [RTen][rten] models such
//! as transformer decoders.
//!
//! For working examples, see the examples in the [rten-examples][rten-examples]
//! crate which import `rten_generate`.
//!
//! [rten]: https://github.com/robertknight/rten
//! [rten-examples]: https://github.com/robertknight/rten/tree/main/rten-examples

pub mod generator;
pub mod metrics;
pub mod sampler;

#[cfg(feature = "text-decoder")]
pub mod text_decoder;

pub use generator::{
    Generator, GeneratorConfig, GeneratorError, GeneratorUtils, ModelInputsConfig,
};

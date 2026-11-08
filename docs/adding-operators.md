# Adding new operators

Adding support for a new operator involves several steps:

 1. Read the ONNX operator specification to understand how the operator
    works. See https://onnx.ai/onnx/operators/.
 2. Define the implementation of the new operator in Rust code and add tests
 3. Update the Rust model loading code to support deserializing the operator
    from ONNX files.

In detail, the process is:

1. Define the implementation of the new operator in Rust. This is a struct
   that implements the `Operator` trait. Operators are defined in modules
   under `src/ops/`. Closely related operators are grouped into modules.
2. Add tests for the new operator at the bottom of the module where the
   operator is defined.
3. Export the operator from the `ops/mod.rs` module
4. Modify `op_registry/onnx_registry.rs` to implement deserialization of the
   operator from ONNX models.

## Defining fusions

If the new operator is a fusion of more primitive operations and represented by
subgraphs of these primitives in existing ONNX models, a graph fusion
optimization can be defined. Fusions pattern-match subgraphs in models and
replace the subgraph by the fused operator, in order to improve performance.
Fusions are defined in `src/optimize.rs`.

## Adding partial support for a new operator

It is OK to add a new operator without support for all features, but attempting
to convert and run a model which uses an unsupported capability must produce an
error rather than silently producing incorrect results. Depending on the
feature, this error may be detected at model conversion or load time (eg. when
an unsupported attribute has a non-default value) or only during inference (eg.
when the input has an unsupported shape).

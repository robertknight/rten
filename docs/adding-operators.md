# Adding new operators

Adding support for a new operator involves several steps:

 1. Read the ONNX operator specification to understand how the operator
    works. See https://onnx.ai/onnx/operators/.
 2. Define the implementation of the new operator in Rust code and add tests
 3. Add the new operator to the FlatBuffers model schema for the rten
    file format.
 4. Update the rten-convert tool to support the new operator
 5. Update the Rust model loading code to support deserializing the operator
    from both ONNX and rten files.

In detail, the process is:

1. Define the implementation of the new operator in Rust. This is a struct
   that implements the `Operator` trait. Operators are defined in modules
   under `src/ops/`. Closely related operators are grouped into modules.
2. Add tests for the new operator at the bottom of the module where the
   operator is defined.
3. Export the operator from the `ops/mod.rs` module
4. Add the new operator to the end of the `OperatorType` enum in schema.fbs.
5. If the new operator requires attributes, add a new table in schema.fbs and
   add the table to the end of the `OperatorAttrs` union. Some existing
   operators share attributes tables. For new operators however it is
   recommended to use a separate type per operator.
6. Run `make schema` to generate updated Rust and Python code to read the
   updated FlatBuffers schema
7. If the new operator has attributes, edit
   `rten-convert/rten_convert/converter.py` and modify
   `op_node_from_onnx_operator` to support converting the attributes for the new
   operator.
8. Modify `op_registry/{onnx_registry.rs, rten_registry.rs}` to implement
   deserialization of the operator for ONNX and rten model formats.
9. Add support for the new operator in `model/rten_builder.rs`
10. Update the `test_all_op_types` test at the bottom of model.rs to run the
    new operator with test input.

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

## FlatBuffers binary compatibility

Additions to the FlatBuffers schema for models should preserve binary
compatibility with existing model files. This is achieved for enums, unions and
tables by making additions at the end of the item.

# Adding new operators

Adding support for a new operator involves several steps:

 1. Read the ONNX operator specification to understand how the operator
    works. See https://onnx.ai/onnx/operators/.
 2. Define the implementation of the new operator in Rust code and add tests
 3. Add the new operator to the FlatBuffers model schema
 4. Update the Python ONNX conversion script to support the new operator
 5. Update the Rust code to support deserializing the operator

In detail, the process is:

1. Define the implementation of the new operator in Rust. This is a struct
   that implements the `Operator` trait. Operators are defined in modules
   under `src/ops/`. Closely related operators are grouped into modules.
2. Add tests for the new operator at the bottom of the module where the
   operator is defined.
3. Export the operator from the `ops/mod.rs` module
4. Add the new operator to the end of the `OperatorType` enum in schema.fbs.
5. If the new operator requires attributes, add a new table in schema.fbs and
   add the table to the end of the `OperatorAttrs` union. If the new operator
   uses the same attributes as an existing operator, it can re-use the
   attributes from that operator.
6. Run `make schema` to generate updated Rust and Python code to read the
   updated FlatBuffers schema
7. If the new operator has attributes, edit
   `rten-convert/rten_convert/converter.py` and modify
   `op_node_from_onnx_operator` to support converting the attributes for the new
   operator.
8. Modify `op_registry.rs` to add deserialization of the new operator from
   .rten model files
9. Add support for the new operator in `model_builder.rs`
10. Update the `test_all_op_types` test at the bottom of model.rs to run the
    new operator with test input.

## Defining fusions

If the new operator is a fusion of more primitive operations and represented by
subgraphs of these primitives in existing ONNX models, a graph fusion
optimization can be defined. Fusions pattern-match subgraphs in models and
replace the subgraph by the fused operator, in order to improve performance.
Fusions are defined in `src/optimize.rs`.

## Adding partial support for a new operator

It is OK to add a new operator without support for all features, but ONNX model
conversion and the operator implementation must report an error if an
unsupported capability is used by a model, rather than silently producing
incorrect results.

The Python ONNX conversion script will check that all attributes of an operator
in the ONNX model are read. Unsupported attributes can be ignored if they have
a value which is equal to the default.

## FlatBuffers binary compatibility

Additions to the FlatBuffers schema for models should preserve binary
compatibility with existing model files. This is achieved for enums, unions and
tables by making additions at the end of the item.

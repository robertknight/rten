# Adding new operators

Adding support for a new operator involves several steps:

 1. Reading the ONNX operator specification to understand how the operator
    works. See https://onnx.ai/onnx/operators/.
 2. Defining the implementation of the new operator in Rust code
 3. Adding the new operator to the FlatBuffers model schema, and implementing
    support for reading it on the Rust side, and writing it in the Python
    script that converts ONNX models to this library's format.
 4. Adding tests for the new operator's implementation and deserialization

In detail, the process is:

1. Add the new operator to the end of the `OperatorType` enum in schema.fbs.
2. If the new operator requires attributes, add a new table in schema.fbs and
   add the table to the end of the `OperatorAttrs` union. If the new operator
   uses the same attributes as an existing operator, it can re-use the
   attributes from that operator.
3. Run `make` to generate updated Rust and Python code to read the updated
   FlatBuffers schema
4. If the new operator has attributes, edit `tools/convert-onnx.py` to read
   the attributes from ONNX and convert to this library's model format
5. Define the implementation of the new operator in Rust. This is a struct
   that implements the `Operator` trait.
6. Add tests for the new operator at the bottom of the module where the
   operator is defined.
7. Export the operator from the `ops/mod.rs` module
8. Modify `read_operator` in `model.rs` to read the new operator from model
   files.
9. Add support for the new operator in `model_builder.rs`
10. Update the `test_all_op_types` test at the bottom of model.rs to run the
    new operator with test input.

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

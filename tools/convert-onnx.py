#!/usr/bin/env python

import array
from argparse import ArgumentParser
from typing import cast

import flatbuffers
import onnx
from onnx import TensorProto

import schema_generated as sg

AttributeValue = int | float | str | list[int]


class Node:
    def __init__(self, name: str):
        self.name = name


class ConstantNode(Node):
    def __init__(self, name: str, shape: list[int], data: array.array):
        super().__init__(name)
        self.shape = shape
        self.data = data

    def get_scalar(self):
        if self.shape != []:
            return None
        return self.data[0]


class OperatorNode(Node):
    op_type: str
    attrs: dict[str, AttributeValue]
    inputs: list[int]
    outputs: list[int]

    def __init__(
        self,
        name: str,
        op_type: str,
        attrs: dict[str, AttributeValue],
        inputs: list[int],
        outputs: list[int],
    ):
        super().__init__(name)
        self.op_type = op_type
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs


class ValueNode(Node):
    def __init__(self, name: str):
        super().__init__(name)


# Mapping of ONNX attribute types to the field on an AttributeProto which
# contains the value. Note that if you try to access the wrong field on an
# AttributeProto, you get a default value instead of an exception.
value_fields = {
    onnx.AttributeProto.FLOAT: "f",
    onnx.AttributeProto.INT: "i",
    onnx.AttributeProto.INTS: "ints",
    onnx.AttributeProto.STRING: "s",
    onnx.AttributeProto.TENSOR: "t",
}


def get_attr(attr_list: list[onnx.AttributeProto], name: str, type_: str, default):
    """Get the value of an optional operator attribute."""
    type_code = getattr(onnx.AttributeProto, type_.upper())
    for attr in attr_list:
        if attr.name == name:
            if attr.type != type_code:
                raise Exception(f"Attribute {name} type does not match {type_}")
            val = getattr(attr, value_fields[type_code])

            # String attribute values are stored as bytes, so we have to decode
            # them.
            if type_ == "string":
                val = val.decode()

            return val
    return default


def require_attr(attr_list: list[onnx.AttributeProto], name: str, type_: str):
    """Get the value of a required operator attribute."""
    val = get_attr(attr_list, name, type_, default=None)
    if val is None:
        raise Exception(f"Missing required attribute {name}")
    return val


def require_attr_or_input(
    name: str,
    type_: str,
    input_index: int,
    onnx_op: onnx.OperatorProto,
    constant_nodes: dict[str, ConstantNode],
):
    """
    Get the value of a required operator attribute or input.

    Some operator inputs changed from attributes to inputs in different ONNX
    releases. This function will look up the value for the input from both
    possible sources.

    In the case where the value comes from an input, it must be a constant
    (ie. specified via an initializer or Constant node in the graph), rather
    than a value computed at runtime.

    :param name: The name of the attribute
    :param type_: The required type of the value
    :param input_index: The index of the operator input
    :param constant_nodes: Map of all the constant values in the model
    """
    val = get_attr(onnx_op.attribute, name, type_, None)
    if val is None and len(onnx_op.input) > input_index:
        input_val = constant_nodes.get(onnx_op.input[input_index])
        if input_val is None:
            raise Exception(f'Input node nor found or not a constant for "{name}"')

        # This function currently only supports extracting scalars from
        # constants, but we could also supports lists as well here.
        scalar = input_val.get_scalar()
        if scalar is None:
            raise Exception(f'Input for "{name}" is not a scalar')
        return scalar

    if val is None:
        raise Exception(f'Missing required attribute or input "{name}"')
    return val


def check_unsupported_attr(
    attr_list: list[onnx.AttributeProto], name: str, type_, default
):
    """Check if an operator has an unsupported non-default value for an attribute."""
    val = get_attr(attr_list, name, type_, default)
    if val != default:
        raise Exception(
            f"Unsupported value {val} for attribute {name}. Default is {default}"
        )


def check_ints_length_and_value(name: str, ints: list[int], allowed_length: int):
    """
    Check that an ints attribute has a fixed length and all values are equal.

    Various ONNX operators allow for a wider range of dimensions and per-axis
    values (eg. for strides, dilations, padding...) than this library currently
    supports.
    """
    if len(ints) != allowed_length:
        raise Exception(f'Attribute "{name}" must have {allowed_length} values')
    for item in ints:
        if item != ints[0]:
            raise Exception(f'All values of attribute "{name}" must be the same')


def convert_array(src_type: str, data: bytes, dest_type: str):
    converted = [x for x in array.array(src_type, data)]
    try:
        return array.array(dest_type, converted)
    except OverflowError:
        # Some ONNX exporters use `INT_MIN` and `INT_MAX` to represent infinity
        # in certain cases, for example slicing to the end of a dimension with
        # unknown size (see
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice and
        # https://github.com/pytorch/pytorch/issues/17606).
        #
        # In the case where the value is an `int64` and we are converting this
        # to an `int32` in the model, this will cause an overflow. To resolve
        # this, clamp the value to the min/max values for the smaller integer
        # type we are using.
        MAX_INT = 2**31 - 1
        MIN_INT = -(2**31) + 1

        saturated = []

        for x in converted:
            if x > MAX_INT:
                print(f"Clamping out-of-range tensor value {x} to {MAX_INT}")
                x = MAX_INT
            elif x < MIN_INT:
                print(f"Clamping out-of-range tensor value {x} to {MIN_INT}")
                x = MIN_INT
            saturated.append(x)

        return array.array(dest_type, saturated)


def constant_node_from_onnx_initializer(tensor) -> ConstantNode:
    dims = list(tensor.dims)

    # Tensors can either store data in a type-appropriate field, or the `raw_data`
    # field. Only one of these should be set.
    tensor_data = (
        tensor.float_data or tensor.int64_data or tensor.int32_data or tensor.raw_data
    )

    # Convert the tensor data to a format supported by this library. For int64
    # tensors, we convert them to int32 and just ignore any issues with
    # overflows.
    match tensor.data_type:
        case onnx.TensorProto.FLOAT:
            data = array.array("f", tensor_data)
        case onnx.TensorProto.UINT8:
            data = convert_array("B", tensor_data, "i")
        case onnx.TensorProto.INT8:
            data = convert_array("b", tensor_data, "i")
        case onnx.TensorProto.UINT16:
            data = convert_array("H", tensor_data, "i")
        case onnx.TensorProto.INT16:
            data = convert_array("h", tensor_data, "i")
        case onnx.TensorProto.INT32:
            data = array.array("i", tensor_data)
        case onnx.TensorProto.INT64:
            data = convert_array("q", tensor_data, "i")
        case _:
            raise ValueError(f"Unsupported tensor data type {tensor.data_type}")

    return ConstantNode(name=tensor.name, shape=dims, data=data)


def constant_node_from_onnx_constant_op(onnx_op: onnx.OperatorProto) -> ConstantNode:
    tensor = require_attr(onnx_op.attribute, "value", "tensor")
    const_node = constant_node_from_onnx_initializer(tensor)

    if not len(onnx_op.output):
        raise Exception(f'Operator "{onnx_op.name}" has no outputs')
    const_node.name = onnx_op.output[0]

    return const_node


def value_node_from_onnx_value(value: onnx.ValueInfoProto) -> ValueNode:
    return ValueNode(name=value.name)


def read_pad_attrs_from_onnx_operator(
    onnx_op: onnx.OperatorProto, attrs: dict[str, AttributeValue]
):
    """
    Read a padding specification from an ONNX operator.
    """

    auto_pad = get_attr(onnx_op.attribute, "auto_pad", "string", "NOTSET")

    match auto_pad:
        case "SAME_UPPER" | "SAME_LOWER":
            attrs["pad_mode"] = "same"
        case "NOTSET":
            padding = get_attr(onnx_op.attribute, "pads", "ints", [0, 0, 0, 0])
            if len(padding) != 4:
                raise Exception('"padding" attribute must have 4 values')
            pad_top, pad_left, pad_right, pad_bottom = iter(padding)

            attrs["pad_mode"] = "fixed"
            attrs["pads"] = [pad_top, pad_left, pad_bottom, pad_right]
        case other:
            raise Exception(f"Unsupported auto_pad value {other}")


def read_stride_attr_from_onnx_operator(
    onnx_op: onnx.OperatorProto, attrs: dict[str, AttributeValue]
):
    """
    Read a stride specification from an ONNX operator.
    """
    strides = get_attr(onnx_op.attribute, "strides", "ints", [1, 1])
    if len(strides) != 2:
        raise Exception('"strides" attribute must have 2 values')
    stride_width, stride_height = iter(strides)
    if stride_width != stride_height:
        raise Exception("Strides must be the same in all dimensions")
    attrs["stride"] = stride_width


def op_node_from_onnx_operator(
    onnx_op: onnx.OperatorProto,
    node_index_from_name: dict[str, int],
    constant_nodes: dict[str, ConstantNode],
) -> OperatorNode:
    """
    Map an ONNX operator to the equivalent operator in this library.

    See https://github.com/onnx/onnx/blob/main/docs/Operators.md for list of
    available ONNX operators and attributes for each.
    """
    input_indexes = []
    for input_name in onnx_op.input:
        index = node_index_from_name.get(input_name)
        if index is None:
            raise Exception(
                f'Unable to find input "{input_name}" for operator {onnx_op.name}'
            )
        input_indexes.append(index)

    output_indexes = []
    for output_name in onnx_op.output:
        index = node_index_from_name.get(output_name)
        if index is None:
            raise Exception(
                f'Unable to find output "{output_name}" for operator {onnx_op.name}'
            )
        output_indexes.append(index)

    attrs: dict[str, AttributeValue] = {}

    match onnx_op.op_type:
        case "Add":
            op_type = "Add"

        case "AveragePool":
            op_type = "AveragePool2d"

            kernel_shape = require_attr(onnx_op.attribute, "kernel_shape", "ints")
            check_ints_length_and_value("kernel_shape", kernel_shape, 2)
            attrs["kernel_size"] = kernel_shape[0]

            read_pad_attrs_from_onnx_operator(onnx_op, attrs)
            read_stride_attr_from_onnx_operator(onnx_op, attrs)

            check_unsupported_attr(onnx_op.attribute, "ceil_mode", "int", 0)
            check_unsupported_attr(onnx_op.attribute, "count_include_pad", "int", 0)

        case "BatchNormalization":
            op_type = "BatchNormalization"

            attrs["epsilon"] = get_attr(onnx_op.attribute, "epsilon", "float", 1e-5)

        case "Cast":
            op_type = "Cast"

            to = get_attr(onnx_op.attribute, "to", "int", TensorProto.DataType.FLOAT)
            match to:
                case TensorProto.DataType.FLOAT:
                    attrs["to"] = sg.DataType.Float
                case TensorProto.DataType.INT32 | TensorProto.DataType.INT64:
                    attrs["to"] = sg.DataType.Int32
                case _:
                    raise Exception(f"Unsupported target type for cast {to}")

        case "Clip":
            op_type = "Clip"

            attrs["min"] = require_attr_or_input(
                "min", "float", 1, onnx_op, constant_nodes
            )
            attrs["max"] = require_attr_or_input(
                "max", "float", 2, onnx_op, constant_nodes
            )

        case "Concat":
            op_type = "Concat"

            attrs["dim"] = require_attr(onnx_op.attribute, "axis", "int")

        case "ConstantOfShape":
            op_type = "ConstantOfShape"

            tensor = require_attr(onnx_op.attribute, "value", "tensor")
            const_node = constant_node_from_onnx_initializer(tensor)

            if len(const_node.data) != 1:
                raise Exception(
                    "Expected ConstantOfShape value to be a 1-element tensor"
                )

            attrs["value"] = const_node.data[0]

        case "Conv":
            op_type = "Conv2d"

            attrs["groups"] = get_attr(onnx_op.attribute, "group", "int", 1)
            read_pad_attrs_from_onnx_operator(onnx_op, attrs)
            read_stride_attr_from_onnx_operator(onnx_op, attrs)

            check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])

        case "ConvTranspose":
            op_type = "ConvTranspose2d"

            read_stride_attr_from_onnx_operator(onnx_op, attrs)

            check_unsupported_attr(onnx_op.attribute, "auto_pad", "string", "NOTSET")
            check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])
            check_unsupported_attr(onnx_op.attribute, "group", "int", 1)
            check_unsupported_attr(
                onnx_op.attribute, "output_padding", "ints", [0, 0, 0, 0]
            )
            check_unsupported_attr(onnx_op.attribute, "pads", "ints", [0, 0, 0, 0])

        case "Div":
            op_type = "Div"

        case "Equal":
            op_type = "Equal"

        case "Expand":
            op_type = "Expand"

        case "Gather":
            op_type = "Gather"

            attrs["axis"] = get_attr(onnx_op.attribute, "axis", "int", 0)

        case "Gemm":
            op_type = "Gemm"

            attrs["alpha"] = get_attr(onnx_op.attribute, "alpha", "float", 1.0)
            attrs["beta"] = get_attr(onnx_op.attribute, "beta", "float", 1.0)
            attrs["transpose_a"] = bool(get_attr(onnx_op.attribute, "transA", "int", 0))
            attrs["transpose_b"] = bool(get_attr(onnx_op.attribute, "transB", "int", 0))

        case "GlobalAveragePool":
            op_type = "GlobalAveragePool"

        case "Identity":
            op_type = "Identity"

        case "LeakyRelu":
            op_type = "LeakyRelu"

            attrs["alpha"] = get_attr(onnx_op.attribute, "alpha", "float", 0.01)

        case "Less":
            op_type = "Less"

        case "MatMul":
            op_type = "MatMul"

        case "MaxPool":
            op_type = "MaxPool2d"

            kernel_shape = require_attr(onnx_op.attribute, "kernel_shape", "ints")
            check_ints_length_and_value("kernel_shape", kernel_shape, 2)
            attrs["kernel_size"] = kernel_shape[0]

            read_pad_attrs_from_onnx_operator(onnx_op, attrs)
            read_stride_attr_from_onnx_operator(onnx_op, attrs)

            check_unsupported_attr(onnx_op.attribute, "ceil_mode", "int", 0)
            check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])
            check_unsupported_attr(onnx_op.attribute, "storage_order", "int", 0)

        case "Mul":
            op_type = "Mul"

        case "Range":
            op_type = "Range"

        case "ReduceMean":
            op_type = "ReduceMean"

            attrs["axes"] = get_attr(onnx_op.attribute, "axes", "ints", None)
            attrs["keep_dims"] = bool(get_attr(onnx_op.attribute, "keepdims", "int", 1))

        case "Relu":
            op_type = "Relu"

        case "Reshape":
            op_type = "Reshape"

            check_unsupported_attr(onnx_op.attribute, "allowzero", "int", 0)

        case "Pad":
            op_type = "Pad"

            check_unsupported_attr(onnx_op.attribute, "mode", "string", "constant")

        case "Pow":
            op_type = "Pow"

        case "Shape":
            op_type = "Shape"

            check_unsupported_attr(onnx_op.attribute, "end", "int", 0)
            check_unsupported_attr(onnx_op.attribute, "start", "int", 0)

        case "Slice":
            op_type = "Slice"

        case "Sigmoid":
            op_type = "Sigmoid"

        case "Softmax":
            op_type = "Softmax"

            attrs["axis"] = get_attr(onnx_op.attribute, "axis", "int", 0)

        case "Split":
            op_type = "Split"

            attrs["axis"] = get_attr(onnx_op.attribute, "axis", "int", 0)
            attrs["split"] = get_attr(onnx_op.attribute, "split", "ints", [])

            check_unsupported_attr(onnx_op.attribute, "num_outputs", "int", 0)

        case "Sqrt":
            op_type = "Sqrt"

        case "Squeeze":
            op_type = "Squeeze"

            axes = get_attr(onnx_op.attribute, "axes", "ints", [])
            attrs["axes"] = axes

        case "Sub":
            op_type = "Sub"

        case "Transpose":
            op_type = "Transpose"

            perm = get_attr(onnx_op.attribute, "perm", "ints", [])
            attrs["perm"] = perm

        case "Unsqueeze":
            op_type = "Unsqueeze"

            axes = get_attr(onnx_op.attribute, "axes", "ints", [])
            attrs["axes"] = axes

        case "Where":
            op_type = "Where"

        case _:
            raise Exception(f"Unsupported operation {onnx_op.op_type}")

    return OperatorNode(
        name=onnx_op.name,
        op_type=op_type,
        attrs=attrs,
        inputs=input_indexes,
        outputs=output_indexes,
    )


def graph_from_onnx_graph(onnx_graph: onnx.GraphProto) -> list[Node]:
    """
    Parse an ONNX model into a graph representation compatible with this library.
    """
    nodes: list[Node] = []

    # Map from tensor ID to node index
    tensor_map: dict[str, int] = {}

    # Map of constant/initializer name to node
    constant_map: dict[str, ConstantNode] = {}

    def add_node(node: Node):
        if node.name in tensor_map:
            raise Exception(f'Node name "{node.name}" conflicts with another node')
        if isinstance(node, ConstantNode):
            constant_map[node.name] = node
        nodes.append(node)
        tensor_map[node.name] = len(nodes) - 1

    for tensor in onnx_graph.initializer:
        const_node = constant_node_from_onnx_initializer(tensor)
        add_node(const_node)
    for operator in onnx_graph.node:
        if operator.op_type != "Constant":
            continue
        const_node = constant_node_from_onnx_constant_op(operator)
        add_node(const_node)

    for value in onnx_graph.input:
        # If the same node is referenced in the ONNX model's `initializer` and
        # `input` properties, ignore the one from the input.
        if value.name in tensor_map:
            continue
        value_node = value_node_from_onnx_value(value)
        add_node(value_node)

    for operator in onnx_graph.node:
        if operator.op_type == "Constant":
            continue

        for output_name in operator.output:
            value_node = ValueNode(output_name)
            add_node(value_node)

        op_node = op_node_from_onnx_operator(operator, tensor_map, constant_map)
        add_node(op_node)

    return nodes


def build_constant_node(builder: flatbuffers.Builder, constant: ConstantNode):
    sg.ConstantNodeStartShapeVector(builder, len(constant.shape))
    for item in reversed(constant.shape):
        builder.PrependUint32(item)
    shape_vec = builder.EndVector()

    match constant.data.typecode:
        case "f":
            sg.FloatDataStartDataVector(builder, len(constant.data))
            for item in reversed(constant.data):
                builder.PrependFloat32(item)
            data_vec = builder.EndVector()

            sg.FloatDataStart(builder)
            sg.FloatDataAddData(builder, data_vec)
            const_data = sg.FloatDataEnd(builder)
            const_data_type = sg.ConstantData.FloatData
        case "i":
            sg.IntDataStartDataVector(builder, len(constant.data))
            for item in reversed(constant.data):
                builder.PrependInt32(item)
            data_vec = builder.EndVector()

            sg.IntDataStart(builder)
            sg.IntDataAddData(builder, data_vec)
            const_data = sg.IntDataEnd(builder)
            const_data_type = sg.ConstantData.IntData
        case _:
            raise ValueError(f"Unsupported data array type {constant.data.typecode}")

    sg.ConstantNodeStart(builder)
    sg.ConstantNodeAddShape(builder, shape_vec)
    sg.ConstantNodeAddDataType(builder, const_data_type)
    sg.ConstantNodeAddData(builder, const_data)
    return sg.ConstantNodeEnd(builder)


def build_operator_node(builder: flatbuffers.Builder, operator: OperatorNode):
    attrs_type = sg.OperatorAttrs.NONE
    attrs = None

    match operator.op_type:
        case "Add":
            op_type_code = sg.OperatorType.Add
        case "AveragePool2d":
            op_type_code = sg.OperatorType.AveragePool2d
            attrs_type = sg.OperatorAttrs.AveragePool2dAttrs

            if operator.attrs["pad_mode"] == "same":
                pad_mode = sg.PadMode.Same
            else:
                pad_mode = sg.PadMode.Fixed
                pads = cast(list[int], operator.attrs["pads"])
                sg.AveragePool2dAttrsStartPadsVector(builder, len(pads))
                for item in reversed(pads):
                    builder.PrependUint32(item)
                pads_vec = builder.EndVector()

            sg.AveragePool2dAttrsStart(builder)
            sg.AveragePool2dAttrsAddKernelSize(builder, operator.attrs["kernel_size"])
            sg.AveragePool2dAttrsAddPadMode(builder, pad_mode)
            if pad_mode == sg.PadMode.Fixed:
                sg.AveragePool2dAttrsAddPads(builder, pads_vec)
            sg.AveragePool2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.AveragePool2dAttrsEnd(builder)
        case "BatchNormalization":
            op_type_code = sg.OperatorType.BatchNormalization
            attrs_type = sg.OperatorAttrs.BatchNormalizationAttrs
            sg.BatchNormalizationAttrsStart(builder)
            sg.BatchNormalizationAttrsAddEpsilon(builder, operator.attrs["epsilon"])
            attrs = sg.BatchNormalizationAttrsEnd(builder)
        case "Cast":
            op_type_code = sg.OperatorType.Cast
            attrs_type = sg.OperatorAttrs.CastAttrs
            sg.CastAttrsStart(builder)
            sg.CastAttrsAddTo(builder, operator.attrs["to"])
            attrs = sg.CastAttrsEnd(builder)
        case "Clip":
            op_type_code = sg.OperatorType.Clip
            attrs_type = sg.OperatorAttrs.ClipAttrs
            sg.ClipAttrsStart(builder)
            sg.ClipAttrsAddMin(builder, operator.attrs["min"])
            sg.ClipAttrsAddMax(builder, operator.attrs["max"])
            attrs = sg.ClipAttrsEnd(builder)
        case "Concat":
            op_type_code = sg.OperatorType.Concat
            attrs_type = sg.OperatorAttrs.ConcatAttrs
            sg.ConcatAttrsStart(builder)
            sg.ConcatAttrsAddDim(builder, operator.attrs["dim"])
            attrs = sg.ConcatAttrsEnd(builder)
        case "ConstantOfShape":
            op_type_code = sg.OperatorType.ConstantOfShape
            attrs_type = sg.OperatorAttrs.ConstantOfShapeAttrs
            value = operator.attrs["value"]

            if isinstance(value, float):
                scalar_type = sg.Scalar.FloatScalar
                sg.FloatScalarStart(builder)
                sg.FloatScalarAddValue(builder, value)
                scalar = sg.FloatScalarEnd(builder)
            else:
                scalar_type = sg.Scalar.IntScalar
                sg.IntScalarStart(builder)
                sg.IntScalarAddValue(builder, value)
                scalar = sg.IntScalarEnd(builder)

            sg.ConstantOfShapeAttrsStart(builder)
            sg.ConstantOfShapeAttrsAddValueType(builder, scalar_type)
            sg.ConstantOfShapeAttrsAddValue(builder, scalar)
            attrs = sg.ConstantOfShapeAttrsEnd(builder)
        case "Conv2d":
            op_type_code = sg.OperatorType.Conv2d
            attrs_type = sg.OperatorAttrs.Conv2dAttrs

            if operator.attrs["pad_mode"] == "same":
                pad_mode = sg.PadMode.Same
            else:
                pad_mode = sg.PadMode.Fixed
                pads = cast(list[int], operator.attrs["pads"])
                sg.Conv2dAttrsStartPadsVector(builder, len(pads))
                for item in reversed(pads):
                    builder.PrependUint32(item)
                pads_vec = builder.EndVector()

            sg.Conv2dAttrsStart(builder)
            sg.Conv2dAttrsAddGroups(builder, operator.attrs["groups"])
            sg.Conv2dAttrsAddPadMode(builder, pad_mode)
            if pad_mode == sg.PadMode.Fixed:
                sg.Conv2dAttrsAddPads(builder, pads_vec)
            sg.Conv2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.Conv2dAttrsEnd(builder)
        case "ConvTranspose2d":
            op_type_code = sg.OperatorType.ConvTranspose2d
            attrs_type = sg.OperatorAttrs.ConvTranspose2dAttrs
            sg.ConvTranspose2dAttrsStart(builder)
            sg.ConvTranspose2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.ConvTranspose2dAttrsEnd(builder)
        case "Div":
            op_type_code = sg.OperatorType.Div
        case "Equal":
            op_type_code = sg.OperatorType.Equal
        case "Expand":
            op_type_code = sg.OperatorType.Expand
        case "Gather":
            op_type_code = sg.OperatorType.Gather
            attrs_type = sg.OperatorAttrs.GatherAttrs
            sg.GatherAttrsStart(builder)
            sg.GatherAttrsAddAxis(builder, operator.attrs["axis"])
            attrs = sg.GatherAttrsEnd(builder)
        case "Gemm":
            op_type_code = sg.OperatorType.Gemm
            attrs_type = sg.OperatorAttrs.GemmAttrs
            sg.GemmAttrsStart(builder)
            sg.GemmAttrsAddAlpha(builder, operator.attrs["alpha"])
            sg.GemmAttrsAddBeta(builder, operator.attrs["beta"])
            sg.GemmAttrsAddTransposeA(builder, operator.attrs["transpose_a"])
            sg.GemmAttrsAddTransposeB(builder, operator.attrs["transpose_b"])
            attrs = sg.GemmAttrsEnd(builder)
        case "GlobalAveragePool":
            op_type_code = sg.OperatorType.GlobalAveragePool
        case "Identity":
            op_type_code = sg.OperatorType.Identity
        case "LeakyRelu":
            op_type_code = sg.OperatorType.LeakyRelu
            attrs_type = sg.OperatorAttrs.LeakyReluAttrs
            sg.LeakyReluAttrsStart(builder)
            sg.LeakyReluAttrsAddAlpha(builder, operator.attrs["alpha"])
            attrs = sg.LeakyReluAttrsEnd(builder)
        case "Less":
            op_type_code = sg.OperatorType.Less
        case "MatMul":
            op_type_code = sg.OperatorType.MatMul
        case "MaxPool2d":
            op_type_code = sg.OperatorType.MaxPool2d
            attrs_type = sg.OperatorAttrs.MaxPool2dAttrs

            if operator.attrs["pad_mode"] == "same":
                pad_mode = sg.PadMode.Same
            else:
                pad_mode = sg.PadMode.Fixed
                pads = cast(list[int], operator.attrs["pads"])
                sg.MaxPool2dAttrsStartPadsVector(builder, len(pads))
                for item in reversed(pads):
                    builder.PrependUint32(item)
                pads_vec = builder.EndVector()

            sg.MaxPool2dAttrsStart(builder)
            sg.MaxPool2dAttrsAddKernelSize(builder, operator.attrs["kernel_size"])
            sg.MaxPool2dAttrsAddPadMode(builder, pad_mode)
            if pad_mode == sg.PadMode.Fixed:
                sg.MaxPool2dAttrsAddPads(builder, pads_vec)
            sg.MaxPool2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.MaxPool2dAttrsEnd(builder)
        case "Mul":
            op_type_code = sg.OperatorType.Mul
        case "Pad":
            op_type_code = sg.OperatorType.Pad
        case "Pow":
            op_type_code = sg.OperatorType.Pow
        case "Range":
            op_type_code = sg.OperatorType.Range
        case "ReduceMean":
            op_type_code = sg.OperatorType.ReduceMean
            attrs_type = sg.OperatorAttrs.ReduceMeanAttrs

            axes = cast(list[int] | None, operator.attrs["axes"])
            if axes:
                sg.ReduceMeanAttrsStartAxesVector(builder, len(axes))
                for item in reversed(axes):
                    builder.PrependInt32(item)
                axes_vec = builder.EndVector()

            sg.ReduceMeanAttrsStart(builder)
            sg.ReduceMeanAttrsAddKeepDims(builder, operator.attrs["keep_dims"])
            if axes_vec:
                sg.ReduceMeanAttrsAddAxes(builder, axes_vec)
            attrs = sg.ReduceMeanAttrsEnd(builder)
        case "Relu":
            op_type_code = sg.OperatorType.Relu
        case "Reshape":
            op_type_code = sg.OperatorType.Reshape
        case "Shape":
            op_type_code = sg.OperatorType.Shape
        case "Sigmoid":
            op_type_code = sg.OperatorType.Sigmoid
        case "Slice":
            op_type_code = sg.OperatorType.Slice
        case "Softmax":
            op_type_code = sg.OperatorType.Softmax
            attrs_type = sg.OperatorAttrs.SoftmaxAttrs
            sg.SoftmaxAttrsStart(builder)
            sg.SoftmaxAttrsAddAxis(builder, operator.attrs["axis"])
            attrs = sg.SoftmaxAttrsEnd(builder)
        case "Split":
            op_type_code = sg.OperatorType.Split
            attrs_type = sg.OperatorAttrs.SplitAttrs

            split = cast(list[int] | None, operator.attrs["split"])
            if split:
                sg.SplitAttrsStartSplitVector(builder, len(split))
                for item in reversed(split):
                    builder.PrependUint32(item)
                split_vec = builder.EndVector()

            sg.SplitAttrsStart(builder)
            sg.SplitAttrsAddAxis(builder, operator.attrs["axis"])
            if split_vec:
                sg.SplitAttrsAddSplit(builder, split_vec)
            attrs = sg.SplitAttrsEnd(builder)

        case "Sqrt":
            op_type_code = sg.OperatorType.Sqrt

        case "Squeeze":
            op_type_code = sg.OperatorType.Squeeze
            attrs_type = sg.OperatorAttrs.SqueezeAttrs

            axes = cast(list[int] | None, operator.attrs["axes"])
            if axes:
                sg.SqueezeAttrsStartAxesVector(builder, len(axes))
                for item in reversed(axes):
                    builder.PrependUint32(item)
                axes_vec = builder.EndVector()

            sg.SqueezeAttrsStart(builder)
            if axes_vec:
                sg.SqueezeAttrsAddAxes(builder, axes_vec)
            attrs = sg.SqueezeAttrsEnd(builder)
        case "Sub":
            op_type_code = sg.OperatorType.Sub
        case "Transpose":
            op_type_code = sg.OperatorType.Transpose
            attrs_type = sg.OperatorAttrs.TransposeAttrs

            perm = cast(list[int] | None, operator.attrs["perm"])
            if perm:
                sg.TransposeAttrsStartPermVector(builder, len(perm))
                for item in reversed(perm):
                    builder.PrependUint32(item)
                perm_vec = builder.EndVector()

            sg.TransposeAttrsStart(builder)
            if perm_vec:
                sg.TransposeAttrsAddPerm(builder, perm_vec)
            attrs = sg.TransposeAttrsEnd(builder)

        case "Unsqueeze":
            op_type_code = sg.OperatorType.Unsqueeze
            attrs_type = sg.OperatorAttrs.UnsqueezeAttrs

            axes = cast(list[int], operator.attrs["axes"])
            sg.UnsqueezeAttrsStartAxesVector(builder, len(axes))
            for item in reversed(axes):
                builder.PrependUint32(item)
            axes_vec = builder.EndVector()

            sg.UnsqueezeAttrsStart(builder)
            sg.UnsqueezeAttrsAddAxes(builder, axes_vec)
            attrs = sg.UnsqueezeAttrsEnd(builder)

        case "Where":
            op_type_code = sg.OperatorType.Where

        case _:
            raise Exception(f"Unsupported operator type {operator.op_type}")

    sg.OperatorNodeStartInputsVector(builder, len(operator.inputs))
    for input_index in reversed(operator.inputs):
        builder.PrependUint32(input_index)
    inputs_vec = builder.EndVector()

    sg.OperatorNodeStartOutputsVector(builder, len(operator.outputs))
    for output_index in reversed(operator.outputs):
        builder.PrependUint32(output_index)
    outputs_vec = builder.EndVector()

    sg.OperatorNodeStart(builder)
    sg.OperatorNodeAddType(builder, op_type_code)
    sg.OperatorNodeAddAttrsType(builder, attrs_type)
    if attrs:
        sg.OperatorNodeAddAttrs(builder, attrs)
    sg.OperatorNodeAddInputs(builder, inputs_vec)
    sg.OperatorNodeAddOutputs(builder, outputs_vec)
    return sg.OperatorNodeEnd(builder)


def build_value_node(builder: flatbuffers.Builder, value: ValueNode):
    sg.ValueNodeStart(builder)
    return sg.ValueNodeEnd(builder)


def write_graph(graph: list[Node], out_path: str):
    """
    Serialize a model graph into a flatbuffers model.

    This serializes the parsed graph representation into the flatbuffers-based
    model format that this library uses.
    """

    builder = flatbuffers.Builder(initialSize=1024)

    node_offsets = []
    for node in graph:
        match node:
            case ConstantNode():
                data_type = sg.NodeKind.ConstantNode
                data = build_constant_node(builder, node)
            case OperatorNode():
                data_type = sg.NodeKind.OperatorNode
                data = build_operator_node(builder, node)
            case ValueNode():
                data_type = sg.NodeKind.ValueNode
                data = build_value_node(builder, node)
            case _:
                raise Exception("Unsupported node type")

        name_str = builder.CreateString(node.name)
        sg.NodeStart(builder)
        sg.NodeAddName(builder, name_str)
        sg.NodeAddDataType(builder, data_type)
        sg.NodeAddData(builder, data)
        node_offset = sg.NodeEnd(builder)
        node_offsets.append(node_offset)

    sg.GraphStartNodesVector(builder, len(graph))
    for node_offset in reversed(node_offsets):
        builder.PrependUOffsetTRelative(node_offset)
    graph_nodes = builder.EndVector()

    sg.GraphStart(builder)
    sg.GraphAddNodes(builder, graph_nodes)
    graph = sg.GraphEnd(builder)

    sg.ModelStart(builder)
    sg.ModelAddSchemaVersion(builder, 1)
    sg.ModelAddGraph(builder, graph)
    model = sg.ModelEnd(builder)

    builder.Finish(model)
    data = builder.Output()

    with open(out_path, "wb") as output:
        output.write(data)


def main():
    parser = ArgumentParser()
    parser.add_argument("model", help="Input ONNX model")
    parser.add_argument("out_name", help="Output model file")
    args = parser.parse_args()

    model_path = args.model

    model = onnx.load(model_path)
    graph = graph_from_onnx_graph(model.graph)
    write_graph(graph, args.out_name)


if __name__ == "__main__":
    main()

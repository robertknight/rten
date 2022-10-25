#!/usr/bin/env python

from argparse import ArgumentParser

import array
import flatbuffers
import onnx

import schema_generated as sg


class Node:
    def __init__(self, name: str):
        self.name = name


class ConstantNode(Node):
    def __init__(self, name: str, shape: list[int], data: array.array):
        super().__init__(name)
        self.shape = shape
        self.data = data


class OperatorNode(Node):
    def __init__(
        self, name: str, op_type: str, attrs: dict[str, int|str], inputs: list[int]
    ):
        super().__init__(name)
        self.op_type = op_type
        self.attrs = attrs
        self.inputs = inputs


class ValueNode(Node):
    def __init__(self, name: str):
        super().__init__(name)


# Mapping of ONNX attribute types to the field on an AttributeProto which
# contains the value. Note that if you try to access the wrong field on an
# AttributeProto, you get a default value instead of an exception.
value_fields = {
    onnx.AttributeProto.FLOAT: 'f',
    onnx.AttributeProto.INT: 'i',
    onnx.AttributeProto.INTS: 'ints',
    onnx.AttributeProto.STRING: 's',
    onnx.AttributeProto.TENSOR: 't'
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


def check_unsupported_attr(attr_list: list[onnx.AttributeProto], name: str, type_, default):
    """Check if an operator has an unsupported non-default value for an attribute."""
    val = get_attr(attr_list, name, type_, default)
    if val != default:
        raise Exception(f"Unsupported value {val} for attribute {name}. Default is {default}")


def check_ints_length_and_value(name: str, ints: list[int], allowed_length: int):
    """
    Check that an ints attribute has a fixed length and all values are equal.

    Various ONNX operators allow for a wider range of dimensions and per-axis
    values (eg. for strides, dilations, padding...) than this library currently
    supports.
    """
    if len(ints) != allowed_length:
        raise Exception(f"Attribute \"{name}\" must have {allowed_length} values")
    for item in ints:
        if item != ints[0]:
            raise Exception(f"All values of attribute \"{name}\" must be the same")


def convert_array(src_type: str, data: bytes, dest_type: str):
    converted = [x for x in array.array(src_type, data)]
    return array.array(dest_type, converted)

def constant_node_from_onnx_initializer(tensor) -> ConstantNode:
    dims = list(tensor.dims)

    # Tensors can either store data in a type-appropriate field, or the `raw_data`
    # field. Only one of these should be set.
    tensor_data = tensor.float_data or tensor.int64_data or tensor.int32_data or tensor.raw_data

    # Convert the tensor data to a format supported by this library. For int64
    # tensors, we convert them to int32 and just ignore any issues with
    # overflows.
    if tensor.data_type == onnx.TensorProto.FLOAT:
        data = array.array("f", tensor_data)
    elif tensor.data_type == onnx.TensorProto.UINT8:
        data = convert_array("B", tensor_data, "i")
    elif tensor.data_type == onnx.TensorProto.INT8:
        data = convert_array("b", tensor_data, "i")
    elif tensor.data_type == onnx.TensorProto.UINT16:
        data = convert_array("H", tensor_data, "i")
    elif tensor.data_type == onnx.TensorProto.INT16:
        data = convert_array("h", tensor_data, "i")
    elif tensor.data_type == onnx.TensorProto.INT32:
        data = array.array("i", tensor_data)
    elif tensor.data_type == onnx.TensorProto.INT64:
        data = convert_array("q", tensor_data, "i")
    else:
        raise ValueError(f"Unsupported tensor data type {tensor.data_type}")

    return ConstantNode(name=tensor.name, shape=dims, data=data)


def onnx_op_output_name(onnx_op: onnx.OperatorProto) -> str:
    if not len(onnx_op.output):
        raise Exception(f'Operator "{onnx_op.name}" has no outputs')
    output_name = onnx_op.output[0]
    return output_name


def constant_node_from_onnx_constant_op(onnx_op: onnx.OperatorProto) -> ConstantNode:
    tensor = require_attr(onnx_op.attribute, "value", "tensor")
    const_node = constant_node_from_onnx_initializer(tensor)
    const_node.name = onnx_op_output_name(onnx_op)
    return const_node

def value_node_from_onnx_value(value: onnx.ValueInfoProto) -> ValueNode:
    return ValueNode(name=value.name)


def op_node_from_onnx_operator(
    onnx_op: onnx.OperatorProto, node_index_from_name: dict[str, int]
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

    attrs: dict[str, int|str] = {}

    if onnx_op.op_type == "Add":
        op_type = "Add"

    elif onnx_op.op_type == "Clip":
        op_type = "Clip"

        attrs["min"] = require_attr(onnx_op.attribute, "min", "float")
        attrs["max"] = require_attr(onnx_op.attribute, "max", "float")

    elif onnx_op.op_type == "Concat":
        op_type = "Concat"

        attrs["dim"] = require_attr(onnx_op.attribute, "axis", "int")

    elif onnx_op.op_type == "Conv":
        op_type = "Conv2d"

        attrs["groups"] = get_attr(onnx_op.attribute, "group", "int", 1)

        auto_pad = get_attr(onnx_op.attribute, "auto_pad", "string", "NOTSET")

        if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            attrs["pad_mode"] = "same"
            attrs["pad_horizontal"] = 0
            attrs["pad_vertical"] = 0
        elif auto_pad == "NOTSET":
            padding = get_attr(onnx_op.attribute, "pads", "ints", [0, 0, 0, 0])
            if len(padding) != 4:
                raise Exception("\"padding\" attribute must have 4 values")
            pad_left, pad_right, pad_top, pad_bottom = iter(padding)
            if pad_left != pad_right:
                raise Exception("Left and right padding must be the same")
            if pad_top != pad_bottom:
                raise Exception("Top and bottom padding must be the same")

            attrs["pad_mode"] = "fixed"
            attrs["pad_horizontal"] = pad_left
            attrs["pad_vertical"] = pad_top

        strides = get_attr(onnx_op.attribute, "strides", "ints", [1, 1])
        if len(strides) != 2:
            raise Exception("\"strides\" attribute must have 2 values")
        stride_width, stride_height = iter(strides)
        if stride_width != stride_height:
            raise Exception("Strides must be the same in all dimensions")
        attrs["stride"] = stride_width

        check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])

    elif onnx_op.op_type == "ConvTranspose":
        op_type = "ConvTranspose2d"

        strides = get_attr(onnx_op.attribute, "strides", "ints", [1, 1])
        check_ints_length_and_value("strides", strides, 2)
        attrs["stride"] = strides[0]

        check_unsupported_attr(onnx_op.attribute, "auto_pad", "string", "NOTSET")
        check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])
        check_unsupported_attr(onnx_op.attribute, "group", "int", 1)
        check_unsupported_attr(onnx_op.attribute, "output_padding", "ints", [0, 0, 0, 0])
        check_unsupported_attr(onnx_op.attribute, "pads", "ints", [0, 0, 0, 0])

    elif onnx_op.op_type == "Gather":
        op_type = "Gather"

        attrs["axis"] = get_attr(onnx_op.attribute, "axis", "int", 0)

    elif onnx_op.op_type == "Gemm":
        op_type = "Gemm"

        attrs["alpha"] = get_attr(onnx_op.attribute, "alpha", "float", 1.0)
        attrs["beta"] = get_attr(onnx_op.attribute, "beta", "float", 1.0)
        attrs["transpose_a"] = bool(get_attr(onnx_op.attribute, "transA", "int", 0))
        attrs["transpose_b"] = bool(get_attr(onnx_op.attribute, "transB", "int", 0))

    elif onnx_op.op_type == "GlobalAveragePool":
        op_type = "GlobalAveragePool"

    elif onnx_op.op_type == "MatMul":
        op_type = "MatMul"

    elif onnx_op.op_type == "MaxPool":
        op_type = "MaxPool2d"

        kernel_shape = require_attr(onnx_op.attribute, "kernel_shape", "ints")
        check_ints_length_and_value("kernel_shape", kernel_shape, 2)
        attrs["kernel_size"] = kernel_shape[0]

        check_unsupported_attr(onnx_op.attribute, "auto_pad", "string", "NOTSET")
        check_unsupported_attr(onnx_op.attribute, "ceil_mode", "int", 0)
        check_unsupported_attr(onnx_op.attribute, "dilations", "ints", [1, 1])
        check_unsupported_attr(onnx_op.attribute, "pads", "ints", [0, 0, 0, 0])
        check_unsupported_attr(onnx_op.attribute, "storage_order", "int", 0)
        check_unsupported_attr(onnx_op.attribute, "strides", "ints", kernel_shape)

    elif onnx_op.op_type == "Relu":
        op_type = "ReLU"

    elif onnx_op.op_type == "Reshape":
        op_type = "Reshape"

        check_unsupported_attr(onnx_op.attribute, "allowzero", "int", 0)

    elif onnx_op.op_type == "Shape":
        op_type = "Shape"

        check_unsupported_attr(onnx_op.attribute, "end", "int", 0)
        check_unsupported_attr(onnx_op.attribute, "start", "int", 0)

    elif onnx_op.op_type == "Slice":
        op_type = "Slice"

    elif onnx_op.op_type == "Sigmoid":
        op_type = "Sigmoid"

    elif onnx_op.op_type == "Unsqueeze":
        op_type = "Unsqueeze"

        axes = get_attr(onnx_op.attribute, "axes", "ints", [])
        attrs["axes"] = axes

    else:
        raise Exception(f"Unsupported operation {onnx_op.op_type}")

    return OperatorNode(
        name=onnx_op_output_name(onnx_op), op_type=op_type, attrs=attrs, inputs=input_indexes
    )


def graph_from_onnx_graph(onnx_graph: onnx.GraphProto) -> list[Node]:
    """
    Parse an ONNX model into a graph representation compatible with this library.
    """
    nodes: list[Node] = []

    # Map from tensor ID to node index
    tensor_map: dict[str, int] = {}

    def add_node(node: Node):
        nodes.append(node)
        tensor_map[node.name] = len(nodes) - 1

    for tensor in onnx_graph.initializer:
        node = constant_node_from_onnx_initializer(tensor)
        add_node(node)
    for operator in onnx_graph.node:
        if operator.op_type != "Constant":
            continue
        node = constant_node_from_onnx_constant_op(operator)
        add_node(node)

    for value in onnx_graph.input:
        node = value_node_from_onnx_value(value)
        add_node(node)

    for operator in onnx_graph.node:
        if operator.op_type == "Constant":
            continue
        node = op_node_from_onnx_operator(operator, tensor_map)
        add_node(node)

    return nodes


def build_constant_node(builder: flatbuffers.Builder, constant: ConstantNode):
    sg.ConstantNodeStartShapeVector(builder, len(constant.shape))
    for item in reversed(constant.shape):
        builder.PrependUint32(item)
    shape_vec = builder.EndVector()

    if constant.data.typecode == "f":
        sg.FloatDataStartDataVector(builder, len(constant.data))
        for item in reversed(constant.data):
            builder.PrependFloat32(item)
        data_vec = builder.EndVector()

        sg.FloatDataStart(builder)
        sg.FloatDataAddData(builder, data_vec)
        const_data = sg.FloatDataEnd(builder)
        const_data_type = sg.ConstantData.FloatData
    elif constant.data.typecode == "i":
        sg.IntDataStartDataVector(builder, len(constant.data))
        for item in reversed(constant.data):
            builder.PrependInt32(item)
        data_vec = builder.EndVector()

        sg.IntDataStart(builder)
        sg.IntDataAddData(builder, data_vec)
        const_data = sg.IntDataEnd(builder)
        const_data_type = sg.ConstantData.IntData
    else:
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
        case "Conv2d":
            op_type_code = sg.OperatorType.Conv2d
            attrs_type = sg.OperatorAttrs.Conv2dAttrs

            if operator.attrs["pad_mode"] == "same":
                pad_mode = sg.PadMode.Same
            else:
                pad_mode = sg.PadMode.Fixed

            sg.Conv2dAttrsStart(builder)
            sg.Conv2dAttrsAddGroups(builder, operator.attrs["groups"])
            sg.Conv2dAttrsAddPadMode(builder, pad_mode)
            sg.Conv2dAttrsAddPadHorizontal(builder, operator.attrs["pad_horizontal"])
            sg.Conv2dAttrsAddPadVertical(builder, operator.attrs["pad_vertical"])
            sg.Conv2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.Conv2dAttrsEnd(builder)
        case "ConvTranspose2d":
            op_type_code = sg.OperatorType.ConvTranspose2d
            attrs_type = sg.OperatorAttrs.ConvTranspose2dAttrs
            sg.ConvTranspose2dAttrsStart(builder)
            sg.ConvTranspose2dAttrsAddStride(builder, operator.attrs["stride"])
            attrs = sg.ConvTranspose2dAttrsEnd(builder)
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
        case "MatMul":
            op_type_code = sg.OperatorType.MatMul
        case "MaxPool2d":
            op_type_code = sg.OperatorType.MaxPool2d
            attrs_type = sg.OperatorAttrs.MaxPool2dAttrs
            sg.MaxPool2dAttrsStart(builder)
            sg.MaxPool2dAttrsAddKernelSize(builder, operator.attrs["kernel_size"])
            attrs = sg.MaxPool2dAttrsEnd(builder)
        case "ReLU":
            op_type_code = sg.OperatorType.ReLU
        case "Reshape":
            op_type_code = sg.OperatorType.Reshape
        case "Pad2d":
            op_type_code = sg.OperatorType.Pad2d
            attrs_type = sg.OperatorAttrs.Pad2dAttrs
            sg.Pad2dAttrsStart(builder)
            sg.Pad2dAttrsAddPadLeft(builder, operator.attrs["pad_left"])
            sg.Pad2dAttrsAddPadRight(builder, operator.attrs["pad_right"])
            sg.Pad2dAttrsAddPadTop(builder, operator.attrs["pad_top"])
            sg.Pad2dAttrsAddPadBottom(builder, operator.attrs["pad_bottom"])
            attrs = sg.Pad2dAttrsEnd(builder)
        case "Shape":
            op_type_code = sg.OperatorType.Shape
        case "Sigmoid":
            op_type_code = sg.OperatorType.Sigmoid
        case "Slice":
            op_type_code = sg.OperatorType.Slice
        case "Unsqueeze":
            op_type_code = sg.OperatorType.Unsqueeze
            attrs_type = sg.OperatorAttrs.UnsqueezeAttrs

            axes = operator.attrs["axes"]
            sg.UnsqueezeAttrsStartAxesVector(builder, len(axes))
            for item in axes:
                builder.PrependUint32(item)
            axes_vec = builder.EndVector()

            sg.UnsqueezeAttrsStart(builder)
            sg.UnsqueezeAttrsAddAxes(builder, axes_vec)
            attrs = sg.UnsqueezeAttrsEnd(builder)

        case _:
            raise Exception(f"Unsupported operator type {operator.op_type}")

    sg.OperatorNodeStartInputsVector(builder, len(operator.inputs))
    for input_index in reversed(operator.inputs):
        builder.PrependUint32(input_index)
    inputs_vec = builder.EndVector()

    sg.OperatorNodeStart(builder)
    sg.OperatorNodeAddType(builder, op_type_code)
    sg.OperatorNodeAddAttrsType(builder, attrs_type)
    if attrs:
        sg.OperatorNodeAddAttrs(builder, attrs)
    sg.OperatorNodeAddInputs(builder, inputs_vec)
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
        if isinstance(node, ConstantNode):
            data_type = sg.NodeKind.ConstantNode
            data = build_constant_node(builder, node)
        elif isinstance(node, OperatorNode):
            data_type = sg.NodeKind.OperatorNode
            data = build_operator_node(builder, node)
        elif isinstance(node, ValueNode):
            data_type = sg.NodeKind.ValueNode
            data = build_value_node(builder, node)
        else:
            raise Exception("Unsupported node type")

        id_str = builder.CreateString(node.name)
        sg.NodeStart(builder)
        sg.NodeAddId(builder, id_str)
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

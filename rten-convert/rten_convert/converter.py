#!/usr/bin/env python

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from functools import reduce
import hashlib
import json
from operator import mul
from os.path import splitext
import sys
import struct
from typing import BinaryIO, Callable, Literal, Optional, Protocol, cast

import flatbuffers
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto, ValueInfoProto

import rten_convert.schema_generated as sg
from rten_convert.attr_reader import AttributeReader
from rten_convert.errors import ConversionError, UnsupportedOperatorError
from rten_convert.graph import Node, ConstantNode, OperatorNode, ValueNode, Graph
from rten_convert.tensor_data import TensorDataBuilder
from rten_convert.util import round_up, warn_once, write_padding

AttributeValue = int | float | str | list[int]


@dataclass
class Metadata:
    """
    Model metadata.

    This corresponds to the `ModelMetadata` struct in RTen. See its docs for
    details of the individual fields.

    When adding new fields here, they also need to be added to
    `METADATA_BUILDER_FNS`.
    """

    code_repository: Optional[str] = None
    commit: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    model_repository: Optional[str] = None
    onnx_hash: Optional[str] = None
    run_id: Optional[str] = None
    run_url: Optional[str] = None


def check_ints_length(name: str, ints: list[int], allowed_lengths: list[int]):
    """
    Check that an ints attribute has a fixed length.

    Various ONNX operators allow for a wider range of dimensions and per-axis
    values (eg. for strides, dilations, padding...) than this library currently
    supports.
    """
    if len(ints) not in allowed_lengths:
        raise ConversionError(
            f'Attribute "{name}" length must be one of {allowed_lengths}'
        )


def constant_node_from_onnx_initializer(
    tensor: onnx.TensorProto, op_name: Optional[str]
) -> ConstantNode:
    dims = list(tensor.dims)
    data = numpy_helper.to_array(tensor)
    dtype_name = data.dtype.name

    match dtype_name:
        # Types that don't need to change
        case "float32" | "int8" | "int32" | "uint8":
            pass

        # Int types that are not supported natively, but can be widened to
        # int32.
        case "bool" | "int16":
            data = data.astype(np.int32)

        # Float types that are not supported natively, but can be widened to
        # float32.
        case "float16":
            warn_once(
                f"Converting {dtype_name} weights to float32 because {dtype_name} is not supported natively yet. This will increase model size."
            )
            data = data.astype(np.float32)

        # Types that need to be narrowed
        case "int64":
            # In the case where the value is an `int64` and we are converting
            # this to an `int32` in the model, this will cause an overflow. To
            # resolve this, clamp the value to the min/max values for the
            # smaller integer type we are using.
            i32 = np.iinfo(np.int32)
            i64 = np.iinfo(np.int64)

            out_of_range_mask = np.logical_or(data > i32.max, data < i32.min)
            for val in data[out_of_range_mask]:
                neg_inf_threshold = -i64.max  # i64.min + 1
                pos_inf_threshold = i64.max
                if val <= neg_inf_threshold or val >= pos_inf_threshold:
                    # Some ONNX exporters use `i64::MIN` (or `-i64::MAX`) and
                    # `i64::MAX` to represent infinity when slicing to the end
                    # of a dimension with unknown size (see
                    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice
                    # and https://github.com/pytorch/pytorch/issues/17606).
                    #
                    # Avoid warning about this common usage of specific i64
                    # values outside the i32 range.
                    continue

                warn_once(
                    f"Clamping out-of-range tensor value {val} to [{i32.min}, {i32.max}]"
                )
            data = data.clip(i32.min, i32.max).astype(np.int32)

        case _:
            raise ConversionError(
                f"Unsupported tensor data type {data.dtype.name} for operator {op_name}"
            )

    return ConstantNode(name=tensor.name, shape=dims, data=data)


def constant_node_from_onnx_constant_op(onnx_op: onnx.OperatorProto) -> ConstantNode:
    def noop_add_node(node: Node) -> int:
        raise ValueError("Not implemented")

    if not len(onnx_op.output):
        raise ConversionError(f'Operator "{onnx_op.name}" has no outputs')

    output_name = onnx_op.output[0]

    attrs = AttributeReader(onnx_op, input_indexes=[], add_node=noop_add_node)
    if (tensor := attrs.get_attr("value", "tensor", None)) is not None:
        const_node = constant_node_from_onnx_initializer(tensor, output_name)
    else:
        if (int_ := attrs.get_attr("value_int", "int", None)) is not None:
            shape = []
            data = np.array(int_).astype(np.int32)
        elif (ints := attrs.get_attr("value_ints", "ints", None)) is not None:
            shape = [len(ints)]
            data = np.array(ints).astype(np.int32)
        elif (float_ := attrs.get_attr("value_float", "float", None)) is not None:
            shape = []
            data = np.array(float_).astype(np.float32)
        elif (floats := attrs.get_attr("value_floats", "floats", None)) is not None:
            shape = [len(floats)]
            data = np.array(floats).astype(np.float32)
        else:
            # Unsupported attributes: value_string, value_strings
            raise ConversionError(
                f'Unable to get value from "Constant" operator "{onnx_op.name}"'
            )
        const_node = ConstantNode("dummy_name", shape, data)

    const_node.name = output_name

    return const_node


def value_node_from_onnx_value(value: onnx.ValueInfoProto) -> ValueNode:
    if value.type.tensor_type.HasField("elem_type"):
        dtype = convert_data_type(value.type.tensor_type.elem_type)
    else:
        dtype = None

    if value.type.tensor_type.HasField("shape"):
        dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
    else:
        dims = None

    return ValueNode(name=value.name, shape=dims, dtype=dtype)


class PadAttrs(Protocol):
    """Common fields for RTen operator attributes which support padding."""

    autoPad: int  # sg.AutoPad
    pads: list[int]


def read_pads(attr_reader: AttributeReader, attrs: PadAttrs) -> None:
    """
    Update the padding attributes for an operator.

    Reads padding attributes from an ONNX operator and updates the attributes
    for an RTen operator.
    """

    auto_pad_attr = attr_reader.get_attr("auto_pad", "string", "NOTSET")
    pads: list[int]

    match auto_pad_attr:
        case "SAME_UPPER" | "SAME_LOWER":
            auto_pad = sg.AutoPad.Same
            pads = []
        case "NOTSET":
            auto_pad = sg.AutoPad.NotSet
            pads = attr_reader.get_attr("pads", "ints", [0, 0, 0, 0])
            if len(pads) not in [2, 4]:
                raise ConversionError('"padding" attribute must have 2 or 4 values')
        case "VALID":
            # "VALID" means no padding. Map this to fixed padding of zero,
            # using `kernel_shape` to infer the number of dimensions.
            auto_pad = sg.AutoPad.NotSet
            kernel_shape = attr_reader.require_attr("kernel_shape", "ints")
            pads = [0, 0] * len(kernel_shape)

        case other:
            raise ConversionError(f"Unsupported auto_pad value {other}")

    attrs.autoPad = auto_pad
    if auto_pad == sg.AutoPad.NotSet:
        attrs.pads = pads


def read_strides(
    attr_reader: AttributeReader,
):
    """
    Read a stride specification from an ONNX operator.
    """
    strides = attr_reader.get_attr("strides", "ints", [1, 1])
    if len(strides) not in [1, 2]:
        raise ConversionError('"strides" attribute must have 1 or 2 values')
    return strides


def read_dilations(
    attr_reader: AttributeReader,
):
    """
    Read a dilation specification from an ONNX operator.
    """
    dilations = attr_reader.get_attr("dilations", "ints", [1, 1])
    if len(dilations) not in [1, 2]:
        raise ConversionError('"dilations" attribute must have 1 or 2 values')
    return dilations


def convert_data_type(onnx_dtype: int) -> int:
    """
    Convert a data type enum value from ONNX to RTen.

    :param onnx_dtype: Data type from `TensorProto.DataType`.
    :return: Value from `sg.DataType`
    """
    match onnx_dtype:
        case TensorProto.DataType.FLOAT:  # type:ignore[attr-defined]
            return sg.DataType.Float
        case (
            TensorProto.DataType.BOOL  # type:ignore[attr-defined]
            | TensorProto.DataType.INT32  # type:ignore[attr-defined]
            | TensorProto.DataType.INT64  # type:ignore[attr-defined]
        ):
            return sg.DataType.Int32
        case TensorProto.DataType.INT8:  # type:ignore[attr-defined]
            return sg.DataType.Int8
        case TensorProto.DataType.UINT8:  # type:ignore[attr-defined]
            return sg.DataType.UInt8
        case _:
            raise ConversionError(f"Unsupported data type {onnx_dtype}")


def op_node_from_onnx_operator(
    onnx_op: onnx.OperatorProto,
    node_index_from_name: dict[str, int],
    constant_nodes: dict[str, ConstantNode],
    add_node: Callable[[Node], int],
) -> OperatorNode:
    """
    Map an ONNX operator to the equivalent operator in this library.

    See https://github.com/onnx/onnx/blob/main/docs/Operators.md for list of
    available ONNX operators and attributes for each.

    :param onnx_op: ONNX operator to convert
    :param node_index_from_name: Mapping of constant and value tensor node names
      in the graph to corresponding input names
    :param constant_nodes: Map of constant value tensor node names
    :param add_node: Function that adds a new node to the graph and returns its
      node ID. This is called if an operator attribute needs to be converted
      to a constant input.
    """
    input_indexes = []
    for input_name in onnx_op.input:
        if input_name:
            index = node_index_from_name.get(input_name)
            if index is None:
                raise ConversionError(
                    f'Unable to find input "{input_name}" for operator {onnx_op.name}'
                )
        else:
            # An empty input name indicates an omitted optional input. This is
            # only required in cases where at least one subsequent optional
            # input is provided. All trailing optional inputs can simply be
            # omitted.
            index = None
        input_indexes.append(index)

    output_indexes = []
    for output_name in onnx_op.output:
        index = node_index_from_name.get(output_name)
        if index is None:
            raise ConversionError(
                f'Unable to find output "{output_name}" for operator {onnx_op.name}'
            )
        output_indexes.append(index)

    # Operator attributes. This will be `None` for operators with no attributes,
    # or one of the `OperatorNameAttrsT` classes generated by flatc.
    attrs: object | None = None

    # Operator type name in RTen models. By default assume this is the same as
    # the ONNX type.
    op_type = onnx_op.op_type

    # Check / convert operator attributes and operator name, if different than
    # ONNX.
    attr_reader = AttributeReader(onnx_op, input_indexes, add_node)
    match op_type:
        case "ArgMax" | "ArgMin":
            attrs = sg.ArgMaxAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", None)
            attrs.keepDims = bool(attr_reader.get_attr("keepdims", "int", 1))
            attr_reader.check_attr("select_last_index", "int", 0)

        case "AveragePool":
            kernel_shape = attr_reader.require_attr("kernel_shape", "ints")
            check_ints_length("kernel_shape", kernel_shape, [1, 2])
            attr_reader.check_attr("ceil_mode", "int", 0)

            attrs = sg.AveragePoolAttrsT()
            attrs.kernelSize = kernel_shape
            read_pads(attr_reader, attrs)
            attrs.strides = read_strides(attr_reader)
            attrs.countIncludePad = attr_reader.get_bool_attr(
                "count_include_pad", False
            )

        case "BatchNormalization":
            attrs = sg.BatchNormalizationAttrsT()
            attrs.epsilon = attr_reader.get_attr("epsilon", "float", 1e-5)
            attr_reader.check_attr("training_mode", "int", 0)

            # Ignore attributes which are valid only if training_mode=1, which
            # is unsupported.
            attr_reader.ignore_attr("momentum")

        case "Cast":
            attrs = sg.CastAttrsT()
            to = attr_reader.get_attr(
                "to",
                "int",
                TensorProto.DataType.FLOAT,  # type:ignore[attr-defined]
            )
            attrs.to = convert_data_type(to)

        case "CastLike":
            attrs = sg.CastLikeAttrsT()

        case "Clip":
            attr_reader.generate_input_from_attr(1, "min", "float")
            attr_reader.generate_input_from_attr(2, "max", "float")

        case "Concat":
            attrs = sg.ConcatAttrsT()
            attrs.axis = attr_reader.require_attr("axis", "int")

        case "ConstantOfShape":
            tensor = attr_reader.require_attr("value", "tensor")
            const_node = constant_node_from_onnx_initializer(tensor, onnx_op.name)

            if len(const_node.data) != 1:
                raise ConversionError(
                    "Expected ConstantOfShape value to be a 1-element tensor"
                )

            scalar: sg.FloatScalarT | sg.IntScalarT
            if const_node.data.dtype == np.float32:
                scalar_type = sg.Scalar.FloatScalar
                scalar = sg.FloatScalarT()
                scalar.value = const_node.data.item()  # type:ignore[assignment]
            elif const_node.data.dtype == np.int32:
                scalar_type = sg.Scalar.IntScalar
                scalar = sg.IntScalarT()
                scalar.value = const_node.data.item()  # type:ignore[assignment]
            else:
                raise ConversionError(
                    f"Unsupported value type {const_node.data.dtype.name} for ConstantOfShape"
                )

            attrs = sg.ConstantOfShapeAttrsT()
            attrs.valueType = scalar_type
            attrs.value = scalar

        case "Conv" | "ConvInteger":
            attrs = sg.ConvAttrsT()
            attrs.dilations = read_dilations(attr_reader)
            attrs.groups = attr_reader.get_attr("group", "int", 1)
            read_pads(attr_reader, attrs)
            attrs.strides = read_strides(attr_reader)

            # The kernel shape is inferred at runtime from the input weight tensor.
            attr_reader.ignore_attr("kernel_shape")

        case "ConvTranspose":
            attrs = sg.ConvTransposeAttrsT()
            attrs.strides = read_strides(attr_reader)

            attr_reader.check_attr("dilations", "ints", ([1], [1, 1]))
            attr_reader.check_attr("group", "int", 1)

            # The kernel shape is inferred at runtime from the input weight tensor.
            attr_reader.ignore_attr("kernel_shape")

            attr_reader.check_attr("output_padding", "ints", [0, 0, 0, 0])
            read_pads(attr_reader, attrs)

        case "CumSum":
            attr_reader.check_attr("exclusive", "int", 0)
            attr_reader.check_attr("reverse", "int", 0)

        case "DequantizeLinear":
            attrs = sg.DequantizeLinearAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 1)

        case "DepthToSpace":
            attrs = sg.DepthToSpaceAttrsT()
            attrs.blockSize = attr_reader.require_attr("blocksize", "int")
            attrs.mode = attr_reader.get_enum_attr("mode", sg.DepthToSpaceMode, "dcr")

        case "Dropout":
            attrs = sg.DropoutAttrsT()
            attrs.seed = attr_reader.get_attr("seed", "int", None)

        case "Einsum":
            attrs = sg.EinsumAttrsT()
            attrs.equation = attr_reader.require_attr("equation", "string")

        case "Elu":
            attrs = sg.EluAttrsT()
            attrs.alpha = attr_reader.get_attr("alpha", "float", 1.0)

        case "EyeLike":
            attrs = sg.EyeLikeAttrsT()
            dtype = attr_reader.get_attr("dtype", "int", None)
            if dtype is not None:
                attrs.dtype = convert_data_type(dtype)
            attrs.k = attr_reader.get_attr("k", "int", 0)

        case "Flatten":
            attrs = sg.FlattenAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 1)

        case "Gather" | "GatherElements":
            attrs = sg.GatherAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 0)

        case "GatherND":
            attrs = sg.GatherNDAttrsT()
            attrs.batchDims = attr_reader.get_attr("batch_dims", "int", 0)

        case "Gelu":
            attrs = sg.GeluAttrsT()
            attrs.approximate = attr_reader.get_enum_attr(
                "approximate", sg.GeluApproximation, "none"
            )

        case "Gemm":
            attrs = sg.GemmAttrsT()
            attrs.alpha = attr_reader.get_attr("alpha", "float", 1.0)
            attrs.beta = attr_reader.get_attr("beta", "float", 1.0)
            attrs.transposeA = bool(attr_reader.get_attr("transA", "int", 0))
            attrs.transposeB = bool(attr_reader.get_attr("transB", "int", 0))

        case "GRU":
            attrs = sg.GRUAttrsT()
            attrs.direction = attr_reader.get_enum_attr(
                "direction", sg.RNNDirection, "forward"
            )
            attrs.hiddenSize = attr_reader.require_attr("hidden_size", "int")
            attrs.linearBeforeReset = bool(
                attr_reader.get_attr("linear_before_reset", "int", 0)
            )

        case "HardSigmoid":
            attrs = sg.HardSigmoidAttrsT()
            attrs.alpha = attr_reader.get_attr("alpha", "float", 0.2)
            attrs.beta = attr_reader.get_attr("beta", "float", 0.5)

        case "If":
            attrs = sg.IfAttrsT()

            then_branch = graph_from_onnx_graph(
                attr_reader.get_attr("then_branch", "graph", None), allow_captures=True
            )
            attrs.thenBranch = DummyGraphT(then_branch, None)

            else_branch = graph_from_onnx_graph(
                attr_reader.get_attr("else_branch", "graph", None), allow_captures=True
            )
            attrs.elseBranch = DummyGraphT(else_branch, None)

        case "InstanceNormalization":
            attrs = sg.BatchNormalizationAttrsT()
            attrs.epsilon = attr_reader.get_attr("epsilon", "float", 1e-5)

        case "IsInf":
            attrs = sg.IsInfAttrsT()
            attr_reader.check_attr("detect_positive", "int", 1)
            attr_reader.check_attr("detect_negative", "int", 1)

        case "LayerNormalization":
            attrs = sg.LayerNormalizationAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", -1)
            attrs.epsilon = attr_reader.get_attr("epsilon", "float", 1e-5)

        case "LeakyRelu":
            attrs = sg.LeakyReluAttrsT()
            attrs.alpha = attr_reader.get_attr("alpha", "float", 0.01)

        case "LogSoftmax":
            attrs = sg.SoftmaxAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 0)

        case "LSTM":
            attrs = sg.LSTMAttrsT()
            attrs.direction = attr_reader.get_enum_attr(
                "direction", sg.RNNDirection, "forward"
            )
            attrs.hiddenSize = attr_reader.require_attr("hidden_size", "int")

            attr_reader.check_attr("activation_alpha", "floats", [])
            attr_reader.check_attr("activation_beta", "floats", [])
            attr_reader.check_attr("activations", "strings", [])
            attr_reader.check_attr("clip", "float", 0.0)
            attr_reader.check_attr("input_forget", "int", 0)
            attr_reader.check_attr("layout", "int", 0)

        case "MaxPool":
            attrs = sg.MaxPoolAttrsT()
            kernel_shape = attr_reader.require_attr("kernel_shape", "ints")
            check_ints_length("kernel_shape", kernel_shape, [1, 2])
            attrs.kernelSize = kernel_shape
            read_pads(attr_reader, attrs)
            attrs.strides = read_strides(attr_reader)

            attr_reader.check_attr("ceil_mode", "int", 0)
            attr_reader.check_attr("dilations", "ints", ([1], [1, 1]))
            attr_reader.check_attr("storage_order", "int", 0)

        case "Mod":
            attrs = sg.ModAttrsT()
            attrs.fmod = bool(attr_reader.get_attr("fmod", "int", 0))

        case "NonMaxSuppression":
            attrs = sg.NonMaxSuppressionAttrsT()
            center_point_box = attr_reader.get_attr("center_point_box", "int", 0)
            attrs.boxOrder = {
                0: sg.NMSBoxOrder.TopLeftBottomRight,
                1: sg.NMSBoxOrder.CenterWidthHeight,
            }[center_point_box]

        case "OneHot":
            attrs = sg.OneHotAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", -1)

        case "RandomNormal" | "RandomNormalLike":
            match op_type:
                case "RandomNormal":
                    attrs = sg.RandomNormalAttrsT()
                    attrs.shape = attr_reader.require_attr("shape", "ints")
                case "RandomNormalLike":
                    attrs = sg.RandomNormalLikeAttrsT()

            attr_reader.check_attr("dtype", "int", 1)
            attrs.seed = attr_reader.get_attr("seed", "float", None)
            attrs.mean = attr_reader.get_attr("mean", "float", 0.0)
            attrs.scale = attr_reader.get_attr("scale", "float", 1.0)

        case "RandomUniform" | "RandomUniformLike":
            match op_type:
                case "RandomUniform":
                    attrs = sg.RandomUniformAttrsT()
                    attrs.shape = attr_reader.require_attr("shape", "ints")
                case "RandomUniformLike":
                    attrs = sg.RandomUniformLikeAttrsT()

            attr_reader.check_attr("dtype", "int", 1)
            attrs.seed = attr_reader.get_attr("seed", "float", None)
            attrs.low = attr_reader.get_attr("low", "float", 0.0)
            attrs.high = attr_reader.get_attr("high", "float", 1.0)

        case (
            "ReduceL2"
            | "ReduceMax"
            | "ReduceMean"
            | "ReduceMin"
            | "ReduceProd"
            | "ReduceSum"
            | "ReduceSumSquare"
        ):
            attrs = sg.ReduceMeanAttrsT()
            attrs.axes = attr_reader.get_attr("axes", "ints", None)
            attrs.keepDims = bool(attr_reader.get_attr("keepdims", "int", 1))

            attr_reader.check_attr("noop_with_empty_axes", "int", 0)

        case "Reshape":
            attrs = sg.ReshapeAttrsT()
            attrs.allowZero = bool(attr_reader.get_attr("allowzero", "int", 0))

        case "Resize":
            attrs = sg.ResizeAttrsT()
            attrs.mode = attr_reader.get_enum_attr(
                "mode", sg.ResizeMode, "nearest", fallback="linear"
            )

            attr_reader.check_attr("antialias", "int", 0)

            # We only support resizing HW dimensions of NCHW tensor
            attr_reader.check_attr("axes", "ints", [2, 3])

            attrs.coordMode = attr_reader.get_enum_attr(
                "coordinate_transformation_mode", sg.CoordTransformMode, "half_pixel"
            )

            attr_reader.check_attr("cubic_coeff_a", "float", -0.75, on_mismatch="warn")
            attr_reader.check_attr("exclude_outside", "int", 0)
            attr_reader.check_attr("extrapolation_value", "float", 0.0)
            attr_reader.check_attr("keep_aspect_ratio_policy", "string", "stretch")

            attrs.nearestMode = attr_reader.get_enum_attr(
                "nearest_mode", sg.NearestMode, "round_prefer_floor"
            )

        case "Pad":
            attrs = sg.PadAttrsT()
            attrs.mode = attr_reader.get_enum_attr("mode", sg.PadMode, "constant")

        case "QuantizeLinear":
            attrs = sg.QuantizeLinearAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 1)

            output_dtype = attr_reader.get_attr("output_dtype", "int", None)
            if output_dtype is not None:
                attrs.outputDtype = convert_data_type(output_dtype)

        case "ScatterElements":
            attrs = sg.ScatterElementsAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 0)
            attrs.reduction = attr_reader.get_enum_attr(
                "reduction", sg.ScatterReduction, "none"
            )

        case "ScatterND":
            attrs = sg.ScatterNDAttrsT()
            attrs.reduction = attr_reader.get_enum_attr(
                "reduction", sg.ScatterReduction, "none"
            )

        case "Shape":
            attrs = sg.ShapeAttrsT()
            start = attr_reader.get_attr("start", "int", None)
            if start is not None:
                attrs.start = start
            end = attr_reader.get_attr("end", "int", None)
            if end is not None:
                attrs.end = end

        case "Softmax":
            attrs = sg.SoftmaxAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 0)

        case "Split":
            attrs = sg.SplitAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", 0)
            attrs.numOutputs = attr_reader.get_attr("num_outputs", "int", None)
            attr_reader.generate_input_from_attr(1, "split", "ints")

        case "Squeeze":
            attr_reader.generate_input_from_attr(1, "axes", "ints")

        case "TopK":
            attrs = sg.TopKAttrsT()
            attrs.axis = attr_reader.get_attr("axis", "int", -1)
            attrs.largest = bool(attr_reader.get_attr("largest", "int", 1))
            attrs.sorted = bool(attr_reader.get_attr("sorted", "int", 1))

        case "Transpose":
            attrs = sg.TransposeAttrsT()
            attrs.perm = attr_reader.get_attr("perm", "ints", None)

        case "Trilu":
            attrs = sg.TriluAttrsT()
            attrs.upper = bool(attr_reader.get_attr("upper", "int", 1))

        case "Unsqueeze":
            attr_reader.generate_input_from_attr(1, "axes", "ints")

    if not hasattr(sg.OperatorType, op_type):
        raise UnsupportedOperatorError(op_type)

    # Display a warning for any attributes that were not handled above.
    for attr in attr_reader.unhandled_attrs():
        warn_once(
            f"WARNING: Unsupported attribute {attr.name} for operator {onnx_op.op_type}"
        )

    return OperatorNode(
        name=onnx_op.name,
        op_type=op_type,
        attrs=attrs,
        inputs=attr_reader.input_indexes,
        outputs=cast(list[int | None], output_indexes),
    )


def duplicate_node_names(nodes: list[onnx.ValueInfoProto]) -> list[str]:
    """
    Check for node names which are duplicated in `nodes` and return the
    duplicates.
    """
    dupes = []
    names = list(n.name for n in nodes)
    for name in set(names):
        if names.count(name) > 1:
            dupes.append(name)
    return dupes


def graph_from_onnx_graph(onnx_graph: onnx.GraphProto, allow_captures=False) -> Graph:
    """
    Parse an ONNX model into a graph representation compatible with this library.

    :param allow_captures:
        Whether operator inputs are allowed to reference value names that do
        not appear in the graph. If true, such inputs are captured from the
        parent scope at runtime.
    """

    nodes: list[Node] = []

    # Map from name of constant or value node to index in `nodes`.
    value_name_to_index: dict[str, int] = {}

    # Map of constant/initializer name to node.
    constant_map: dict[str, ConstantNode] = {}

    # IDs of value nodes that are not listed in the graph's inputs or computed
    # by operator outputs. These are resolved at runtime by looking up the name
    # in parent scopes.
    capture_ids: list[int] | None
    if allow_captures:
        capture_ids = []
    else:
        capture_ids = None

    def add_node(node: Node) -> int:
        nodes.append(node)
        node_index = len(nodes) - 1

        if not isinstance(node, OperatorNode) and node.name:
            if node.name in value_name_to_index:
                raise ConversionError(
                    f'Node name "{node.name}" conflicts with another node'
                )
            value_name_to_index[node.name] = node_index

        if isinstance(node, ConstantNode):
            constant_map[node.name] = node

        return node_index

    conversion_errors = 0

    for tensor in onnx_graph.initializer:
        try:
            const_node = constant_node_from_onnx_initializer(tensor, None)
            add_node(const_node)
        except Exception as ex:
            warn_once(f"Error converting initializer: {ex}")
            conversion_errors += 1

    for operator in onnx_graph.node:
        if operator.op_type != "Constant":
            continue
        try:
            const_node = constant_node_from_onnx_constant_op(operator)
            add_node(const_node)
        except Exception as ex:
            warn_once(f'Error converting "Constant" operator: {ex}')
            conversion_errors += 1

    # If conversion of any tensors failed, then conversion of any operators
    # which use those tensors will also fail, so we bail early.
    if conversion_errors > 0:
        raise ConversionError(
            f"Errors occurred when converting {conversion_errors} constants"
        )

    def add_value_node(value: ValueInfoProto):
        # If the same node is referenced in at 2 or more of:
        #
        # - The initializer list
        # - The input list
        # - The output list
        #
        # Then we only keep the first definition seen.
        if value.name in value_name_to_index:
            return
        value_node = value_node_from_onnx_value(value)
        add_node(value_node)

    # Create value nodes for inputs, outputs and internal values for which the
    # ONNX model contains dtype or shape information.
    for value_info in onnx_graph.input:
        add_value_node(value_info)

    for value_info in onnx_graph.output:
        add_value_node(value_info)

    for value_info in onnx_graph.value_info:
        add_value_node(value_info)

    # Names of unsupported operators that have been encountered.
    unsupported_op_types: set[str] = set()

    for operator in onnx_graph.node:
        if operator.op_type == "Constant":
            continue

        # If converting a subgraph, create capture nodes for any input names
        # which don't appear in the graph.
        if capture_ids is not None:
            for input_name in operator.input:
                if input_name not in value_name_to_index:
                    capture_id = add_node(
                        ValueNode(name=input_name, shape=None, dtype=None)
                    )
                    capture_ids.append(capture_id)

        for output_name in operator.output:
            # If this output value hasn't been registered already, create a
            # value node without shape or dtype info.
            if output_name in value_name_to_index:
                continue
            value_node = ValueNode(output_name, shape=None, dtype=None)
            add_node(value_node)

        try:
            op_node = op_node_from_onnx_operator(
                operator, value_name_to_index, constant_map, add_node=add_node
            )
            add_node(op_node)
        except Exception as ex:
            skip_warning = False
            if isinstance(ex, UnsupportedOperatorError):
                if ex.op_type in unsupported_op_types:
                    skip_warning = True
                else:
                    unsupported_op_types.add(ex.op_type)

            if not skip_warning:
                print(
                    f'Error converting operator "{operator.name}": {ex}',
                    file=sys.stderr,
                )
            conversion_errors += 1

    if conversion_errors > 0:
        raise ConversionError(
            f"Errors occurred when converting {conversion_errors} operators"
        )

    dup_inputs = duplicate_node_names(list(onnx_graph.input))
    if dup_inputs:
        raise ConversionError(
            f"ONNX graph contains duplicate input names: {', '.join(dup_inputs)}"
        )

    dup_outputs = duplicate_node_names(list(onnx_graph.output))
    if dup_outputs:
        raise ConversionError(
            f"ONNX graph contains duplicate output names: {', '.join(dup_outputs)}"
        )

    inputs = [value_name_to_index[info.name] for info in onnx_graph.input]
    outputs = [value_name_to_index[info.name] for info in onnx_graph.output]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs, captures=capture_ids)


def build_constant_node(
    builder: flatbuffers.Builder,
    constant: ConstantNode,
    tensor_data: TensorDataBuilder | None,
):
    """
    Serialize a constant tensor value (eg. model weights) into a FlatBuffers model.
    """
    shape_vec = write_vec(
        builder, sg.ConstantNodeStartShapeVector, constant.shape, "u32"
    )
    n_elems = reduce(mul, constant.shape, 1)
    assert n_elems == constant.data.size, "constant shape does not match element count"

    match constant.data.dtype:
        case np.float32:
            inline_data_type = sg.ConstantData.FloatData
            dtype = sg.ConstantDataType.Float32
        case np.int32:
            inline_data_type = sg.ConstantData.Int32Data
            dtype = sg.ConstantDataType.Int32
        case np.int8:
            inline_data_type = sg.ConstantData.Int8Data
            dtype = sg.ConstantDataType.Int8
        case np.uint8:
            inline_data_type = sg.ConstantData.UInt8Data
            dtype = sg.ConstantDataType.UInt8
        case _:
            raise ConversionError(
                f"Unsupported data array type {constant.data.dtype.name}"  # type:ignore[union-attr]
            )

    # Store inline if we're generating the V1 format, or the tensor is small.
    # Small values are mostly parameters such as axes, slice ranges etc.
    store_inline = (
        tensor_data is None or n_elems <= 16
    ) and inline_data_type is not None
    inline_data = None
    data_offset = None

    if store_inline:
        inline_data_vec = builder.CreateNumpyVector(constant.data.flatten())
        match constant.data.dtype:
            case np.float32:
                sg.FloatDataStart(builder)
                sg.FloatDataAddData(builder, inline_data_vec)
                inline_data = sg.FloatDataEnd(builder)
            case np.int32:
                sg.Int32DataStart(builder)
                sg.Int32DataAddData(builder, inline_data_vec)
                inline_data = sg.Int32DataEnd(builder)
            case np.int8:
                sg.Int8DataStart(builder)
                sg.Int8DataAddData(builder, inline_data_vec)
                inline_data = sg.Int8DataEnd(builder)
            case np.uint8:
                sg.UInt8DataStart(builder)
                sg.UInt8DataAddData(builder, inline_data_vec)
                inline_data = sg.UInt8DataEnd(builder)
            case _:
                raise ConversionError(
                    f"Unsupported data type for inline storage {constant.data.dtype.name}"  # type:ignore
                )
    else:
        assert tensor_data
        data_offset = tensor_data.add_tensor(constant.data)

    sg.ConstantNodeStart(builder)
    sg.ConstantNodeAddShape(builder, shape_vec)
    sg.ConstantNodeAddDtype(builder, dtype)

    if inline_data:
        sg.ConstantNodeAddDataType(builder, inline_data_type)
        sg.ConstantNodeAddData(builder, inline_data)
    else:
        assert data_offset is not None
        sg.ConstantNodeAddDataOffset(builder, data_offset)

    return sg.ConstantNodeEnd(builder)


def write_vec(
    builder: flatbuffers.Builder,
    start_vec,
    data: list[int],
    dtype: Literal["u32", "i32", "offset"],
):
    """
    Serialize a list into a vector in a FlatBuffers buffer.

    `start_vec` is the generated function that starts the vector.
    """
    start_vec(builder, len(data))
    for item in reversed(data):
        match dtype:
            case "u32":
                builder.PrependUint32(item)
            case "i32":
                builder.PrependInt32(item)
            case "offset":
                builder.PrependUOffsetTRelative(item)
            case _:
                raise ConversionError("Unsupported data type")
    return builder.EndVector()


class DummyGraphT(sg.GraphT):
    """
    Replacement for `sg.GraphT` whose `Pack` method serializes a `Graph` object.
    """

    def __init__(self, graph: Graph, tensor_data: TensorDataBuilder | None):
        self.graph = graph
        self.tensor_data = tensor_data

    def Pack(self, builder):
        return build_graph(builder, self.graph, self.tensor_data)


def build_operator_node(
    builder: flatbuffers.Builder,
    operator: OperatorNode,
    tensor_data: TensorDataBuilder | None,
):
    """
    Serialize an operator into a FlatBuffers model.
    """

    if operator.attrs:
        # Given an `operator.attrs` which is an instance of `SomeOpAttrsT`,
        # find the `sg.OperatorAttrs.SomeOpAttrs` constant.
        attrs_const_name = operator.attrs.__class__.__name__[:-1]
        attrs_type = getattr(sg.OperatorAttrs, attrs_const_name)
    else:
        attrs_type = sg.OperatorAttrs.NONE

    operator_table = sg.OperatorNodeT()
    operator_table.type = getattr(sg.OperatorType, operator.op_type)

    operator_table.attrsType = attrs_type
    operator_table.attrs = operator.attrs

    # If any attributes are graphs, update the `tensor_data` reference used when
    # serializing.
    if operator.attrs:
        for attr, val in operator.attrs.__dict__.items():
            if isinstance(val, DummyGraphT):
                val.tensor_data = tensor_data

    def node_id(maybe_id: int | None) -> int:
        if maybe_id is None:
            return -1
        return maybe_id

    operator_table.inputs = [node_id(id_) for id_ in operator.inputs]
    operator_table.outputs = [node_id(id_) for id_ in operator.outputs]
    return operator_table.Pack(builder)


def build_value_node(builder: flatbuffers.Builder, value: ValueNode):
    """
    Serialize a placeholder for an input/output value into a FlatBuffers model.
    """

    def write_dim(builder, dim: str | int) -> int:
        if isinstance(dim, str):
            name = builder.CreateString(dim)
            sg.DimStart(builder)
            sg.DimAddName(builder, name)
        else:
            sg.DimStart(builder)
            sg.DimAddValue(builder, dim)
        return sg.DimEnd(builder)

    if value.shape is not None:
        dims = [write_dim(builder, dim) for dim in value.shape]
        shape_vec = write_vec(builder, sg.ValueNodeStartShapeVector, dims, "offset")
    else:
        shape_vec = None

    sg.ValueNodeStart(builder)
    if shape_vec:
        sg.ValueNodeAddShape(builder, shape_vec)
    if value.dtype is not None:
        sg.ValueNodeAddDtype(builder, value.dtype)
    return sg.ValueNodeEnd(builder)


METADATA_BUILDER_FNS = {
    "code_repository": sg.MetadataAddCodeRepository,
    "commit": sg.MetadataAddCommit,
    "description": sg.MetadataAddDescription,
    "license": sg.MetadataAddLicense,
    "model_repository": sg.MetadataAddModelRepository,
    "onnx_hash": sg.MetadataAddOnnxHash,
    "run_id": sg.MetadataAddRunId,
    "run_url": sg.MetadataAddRunUrl,
}
"""
Map of metadata field to function that serializes this field.
"""


def build_metadata(builder: flatbuffers.Builder, metadata: Metadata):
    """
    Serialize model metadata into a flatbuffers model.
    """

    # Map of field name to flatbuffer string offset.
    field_values = {}

    for field in METADATA_BUILDER_FNS.keys():
        if val := getattr(metadata, field):
            field_values[field] = builder.CreateString(val)

    sg.MetadataStart(builder)
    for field, builder_fn in METADATA_BUILDER_FNS.items():
        if val := field_values.get(field):
            builder_fn(builder, val)
    return sg.MetadataEnd(builder)


def build_graph(
    builder: flatbuffers.Builder, graph: Graph, tensor_data: TensorDataBuilder | None
):
    """
    Serialize a computation graph into a flatbuffers model.
    """
    node_offsets = []
    for node in graph.nodes:
        match node:
            case ConstantNode():
                data_type = sg.NodeKind.ConstantNode
                data = build_constant_node(builder, node, tensor_data)
            case OperatorNode():
                data_type = sg.NodeKind.OperatorNode
                data = build_operator_node(builder, node, tensor_data)
            case ValueNode():
                data_type = sg.NodeKind.ValueNode
                data = build_value_node(builder, node)
            case _:
                raise ConversionError("Unsupported node type")

        name_str = builder.CreateString(node.name)
        sg.NodeStart(builder)
        sg.NodeAddName(builder, name_str)
        sg.NodeAddDataType(builder, data_type)
        sg.NodeAddData(builder, data)
        node_offset = sg.NodeEnd(builder)
        node_offsets.append(node_offset)

    graph_nodes = write_vec(builder, sg.GraphStartNodesVector, node_offsets, "offset")
    inputs = write_vec(builder, sg.GraphStartInputsVector, graph.inputs, "u32")
    outputs = write_vec(builder, sg.GraphStartOutputsVector, graph.outputs, "u32")

    if graph.captures is not None:
        captures = write_vec(
            builder, sg.GraphStartCapturesVector, graph.captures, "u32"
        )
    else:
        captures = None

    sg.GraphStart(builder)
    sg.GraphAddNodes(builder, graph_nodes)
    sg.GraphAddInputs(builder, inputs)
    sg.GraphAddOutputs(builder, outputs)

    if captures is not None:
        sg.GraphAddCaptures(builder, captures)

    return sg.GraphEnd(builder)


def serialize_model(
    graph: Graph, metadata: Metadata, tensor_data: TensorDataBuilder | None
) -> bytes:
    """
    Serialize a model into a flatbuffers model.

    This serializes the parsed graph representation into the flatbuffers-based
    model format that this library uses.

    :param graph: The main graph for the model
    :param metadata: Model metadata
    :param tensor_data:
        Object that will be used to write the tensor data in a separate segment
        of the file that follows the model buffer.
    """

    builder = flatbuffers.Builder(initialSize=1024)

    graph = build_graph(builder, graph, tensor_data)
    metadata = build_metadata(builder, metadata)

    sg.ModelStart(builder)
    sg.ModelAddSchemaVersion(builder, 1)
    sg.ModelAddGraph(builder, graph)
    sg.ModelAddMetadata(builder, metadata)
    model = sg.ModelEnd(builder)

    builder.Finish(model)
    return builder.Output()


def write_header(
    fp: BinaryIO, model_data_offset: int, model_data_len: int, tensor_data_offset: int
):
    """
    Write the model file header.

    :param model_data_offset:
        Offset of the FlatBuffers data for the model, relative to the start of the file.
    :param model_data_len:
        Length of the FlatBuffers data for the model.
    :param tensor_data_offset:
        Offset of the tensor data segment, relative to the start of the file.
    """
    fp.write(b"RTEN")

    version = 2
    version_bytes = struct.pack("<I", version)
    fp.write(version_bytes)

    md_offset_bytes = struct.pack("<Q", model_data_offset)
    fp.write(md_offset_bytes)

    md_len_bytes = struct.pack("<Q", model_data_len)
    fp.write(md_len_bytes)

    td_offset_bytes = struct.pack("<Q", tensor_data_offset)
    fp.write(td_offset_bytes)


def sha256(filename: str) -> str:
    """Generate SHA-256 hash of a file as a hex string."""
    hasher = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_metadata(onnx_path: str, metadata_path: Optional[str] = None) -> Metadata:
    """
    Generate metadata to embed into RTen model.

    :param onnx_path: Path to .onnx file
    :param metadata_path: Path to JSON file containing additional metadata
    """
    onnx_hash = sha256(onnx_path)

    fields = {"onnx_hash": onnx_hash}
    if metadata_path:
        with open(metadata_path) as fp:
            metadata_dict = json.load(fp)

        for field in METADATA_BUILDER_FNS.keys():
            if field == "onnx_hash":
                # This is handled separately.
                continue
            fields[field] = metadata_dict.get(field)

    return Metadata(**fields)


def main():
    parser = ArgumentParser(description="Convert ONNX models to .rten format.")
    parser.add_argument("model", help="Input ONNX model")
    parser.add_argument(
        "-m", "--metadata", help="Path to JSON file containing model metadata."
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Generate version 1 .rten models. These are limited to files < 2GB",
    )
    parser.add_argument(
        "--infer-shapes",
        action=BooleanOptionalAction,
        default=True,
        help="Perform shape inference before converting model (default: true)",
    )
    parser.add_argument("out_name", help="Output model file name", nargs="?")
    args = parser.parse_args()

    output_path = args.out_name
    if output_path is None:
        model_basename = splitext(args.model)[0]
        output_path = f"{model_basename}.rten"

    use_v2_format = not args.v1
    if use_v2_format:
        tensor_data = TensorDataBuilder()
    else:
        # Version 1 format stores all tensor data inline.
        tensor_data = None

    model = onnx.load(args.model)

    # Run shape inference. This is recommended as some runtime graph
    # optimizations depend on it.
    if args.infer_shapes:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)

    graph = graph_from_onnx_graph(model.graph)
    metadata = generate_metadata(args.model, args.metadata)

    try:
        model_data = serialize_model(graph, metadata, tensor_data)
    except flatbuffers.builder.BuilderSizeError:
        print("Model buffer exceeded maximum size (2GB)", file=sys.stderr)
        if not use_v2_format:
            print(
                "To serialize models > 2GB, the V2 format must be used.",
                file=sys.stderr,
            )
        sys.exit(1)

    with open(output_path, "wb") as output:
        if use_v2_format:
            header_size = 32

            # The model data needs to start at an offset that is a multiple of
            # the largest tensor element alignment, so inline tensor data is
            # correctly aligned. Fortunately the header size is already a
            # suitable offset.
            model_data_offset = header_size
            model_data_len = len(model_data)

            tensor_data_align = 64
            tensor_data_offset = round_up(
                header_size + len(model_data), tensor_data_align
            )

            write_header(output, model_data_offset, model_data_len, tensor_data_offset)

            assert output.tell() == model_data_offset
            output.write(model_data)
            assert output.tell() <= tensor_data_offset

            if tensor_data:
                write_padding(output, tensor_data_offset - output.tell())
                tensor_data.write(output)
        else:
            # The Version 1 format is just the FlatBuffers model
            output.write(model_data)


if __name__ == "__main__":
    main()

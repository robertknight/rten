"""
Convert parsed models into ONNX format.

This is the inverse of the conversion performed by `rten_convert.converter`.

Some information is lost when models are converted to `.rten` format, so the
generated ONNX model will not be identical to the model the `.rten` file was
produced from. The main differences are:

- rten models store `int64` and `bool` tensors as `int32`. Since most integer
  tensors in ONNX models are `int64`, they are converted back to `int64`, and
  `Cast` operations are inserted where an operator requires `bool` inputs.
- `float16` and `float64` tensors were widened to `float32` during the original
  conversion. This cannot be undone.
- `SAME_LOWER` auto-padding is reported as `SAME_UPPER`, as rten has a single
  "same" padding mode.
- Operators which were deprecated in newer ONNX opsets (`Upsample`, `Scatter`)
  are replaced by their modern equivalents.
"""

from dataclasses import fields
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import onnx
from onnx import (
    GraphProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    helper,
    numpy_helper,
)

import rten_convert.schema_generated as sg
from rten_convert.errors import ConversionError
from rten_convert.graph import ConstantNode, Graph, OperatorNode, ValueNode
from rten_convert.metadata import Metadata
from rten_convert.util import warn_once

DEFAULT_OPSET = 23
"""
ONNX opset version that generated models target.

Operators are converted using the semantics that rten implements, which
requires a recent opset. Notable examples are `Reduce*` operators taking axes
as an input (opset 18), the interaction between `ceil_mode` and
`count_include_pad` in `AveragePool` (opset 19), `Gelu` (opset 20), the
`output_dtype` attribute of `QuantizeLinear` (opset 21) and `Attention`
(opset 23).
"""

DTYPE_FROM_RTEN = {
    # rten uses int32 to represent ONNX's int64 and bool types. int64 is by far
    # the more common of the two in ONNX models, so it is used as the default.
    # Values used in a boolean context are handled separately.
    sg.DataType.Int32: TensorProto.INT64,
    sg.DataType.Float: TensorProto.FLOAT,
    sg.DataType.Int8: TensorProto.INT8,
    sg.DataType.UInt8: TensorProto.UINT8,
}
"""Map of `sg.DataType` to ONNX tensor element type."""

DTYPE_FROM_NUMPY = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.int32): TensorProto.INT64,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.uint8): TensorProto.UINT8,
}
"""Map of constant data type to ONNX tensor element type."""

REDUCE_OPS = {
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
}

BOOL_INPUTS: dict[str, tuple[int, ...]] = {
    "And": (0, 1),
    "If": (0,),
    "Loop": (1,),
    "Not": (0,),
    "Or": (0, 1),
    "Where": (0,),
    "Xor": (0, 1),
}
"""Map of operator name to indexes of inputs which ONNX requires to be `bool`."""

INT64_INPUTS: dict[str, tuple[int, ...]] = {
    "ConstantOfShape": (0,),
    "DFT": (2,),
    "Expand": (1,),
    "GatherND": (1,),
    "Loop": (0,),
    "NonMaxSuppression": (2,),
    "Pad": (1,),
    "Reshape": (1,),
    "Resize": (3,),
    "ReverseSequence": (1,),
    "RotaryEmbedding": (3,),
    "ScatterND": (1,),
    "Split": (1,),
    "Squeeze": (1,),
    "Tile": (1,),
    "TopK": (1,),
    "Trilu": (1,),
    "Unsqueeze": (1,),
}
"""Map of operator name to indexes of inputs which ONNX requires to be `int64`."""

BOOL_OUTPUT_OPS = {
    "And",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "IsInf",
    "IsNaN",
    "Less",
    "LessOrEqual",
    "Not",
    "Or",
    "Xor",
}
"""Operators whose output is always `bool`."""

INT64_OUTPUT_OPS = {"ArgMax", "ArgMin", "NonMaxSuppression", "NonZero", "Shape", "Size"}
"""Operators whose output is always `int64`."""

FLOAT_OUTPUT_OPS = {
    "DequantizeLinear",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
}
"""Operators whose output is always `float32` in converted models."""

SAME_DTYPE_AS_INPUT = {
    "Where": 1,
    "CastLike": 1,
    "OneHot": 2,
}
"""
Operators whose output data type matches an input other than the first.
"""

SAME_DTYPE_AS_FIRST_INPUT = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AveragePool",
    "BatchNormalization",
    "Ceil",
    "Clip",
    "Concat",
    "ConcatFromSequence",
    "Conv",
    "ConvTranspose",
    "Cos",
    "Cosh",
    "CumSum",
    "DFT",
    "DepthToSpace",
    "Div",
    "Einsum",
    "Elu",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "GRU",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gelu",
    "Gemm",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "GridSample",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "InstanceNormalization",
    "LSTM",
    "LayerNormalization",
    "LeakyRelu",
    "Log",
    "LogSoftmax",
    "LpNormalization",
    "MatMul",
    "Max",
    "MaxPool",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Neg",
    "PRelu",
    "Pad",
    "Pow",
    "Range",
    "Reciprocal",
    "Relu",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "Round",
    "STFT",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Slice",
    "Softmax",
    "Softplus",
    "Split",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "Tile",
    "Transpose",
    "Trilu",
    "Unsqueeze",
    "Upsample",
} | REDUCE_OPS
"""Operators whose output data type matches their first input."""

AUTO_PAD_NAMES = {sg.AutoPad.Same: "SAME_UPPER", sg.AutoPad.NotSet: "NOTSET"}
COORD_TRANSFORM_MODES = {
    sg.CoordTransformMode.HalfPixel: "half_pixel",
    sg.CoordTransformMode.Asymmetric: "asymmetric",
    sg.CoordTransformMode.AlignCorners: "align_corners",
    sg.CoordTransformMode.PytorchHalfPixel: "pytorch_half_pixel",
}
DEPTH_TO_SPACE_MODES = {sg.DepthToSpaceMode.DCR: "DCR", sg.DepthToSpaceMode.CRD: "CRD"}
GELU_APPROXIMATIONS = {
    sg.GeluApproximation.None_: "none",
    sg.GeluApproximation.Tanh: "tanh",
}
NEAREST_MODES = {
    sg.NearestMode.Floor: "floor",
    sg.NearestMode.Ceil: "ceil",
    sg.NearestMode.RoundPreferFloor: "round_prefer_floor",
    sg.NearestMode.RoundPreferCeil: "round_prefer_ceil",
}
PAD_MODES = {
    sg.PadMode.Constant: "constant",
    sg.PadMode.Reflect: "reflect",
    sg.PadMode.Edge: "edge",
    sg.PadMode.Wrap: "wrap",
}
RESIZE_MODES = {sg.ResizeMode.Nearest: "nearest", sg.ResizeMode.Linear: "linear"}
RNN_DIRECTIONS = {
    sg.RNNDirection.Forward: "forward",
    sg.RNNDirection.Reverse: "reverse",
    sg.RNNDirection.Bidirectional: "bidirectional",
}
SCATTER_REDUCTIONS = {
    sg.ScatterReduction.None_: "none",
    sg.ScatterReduction.Add: "add",
    sg.ScatterReduction.Mul: "mul",
    sg.ScatterReduction.Min: "min",
    sg.ScatterReduction.Max: "max",
}


def _enum_value(name: str, value: int, names: dict[int, str]) -> str:
    """Convert an enum value read from a model to the ONNX attribute value."""
    if value not in names:
        raise ConversionError(f'Unsupported value {value} for "{name}" attribute')
    return names[value]


def _ints(values: Iterable[Any]) -> list[int]:
    """Convert a vector attribute read from a model to a list of ints."""
    return [int(value) for value in values]


def _required_dtype(op_type: str, input_index: int) -> Optional[int]:
    """
    Return the element type that ONNX requires for an operator input.

    Returns None if the input can have any type, or a type that rten models
    preserve.
    """
    if input_index in BOOL_INPUTS.get(op_type, ()):
        return TensorProto.BOOL
    if input_index in INT64_INPUTS.get(op_type, ()):
        return TensorProto.INT64
    # `Reduce*` operators take the axes to reduce as their second input.
    if input_index == 1 and op_type in REDUCE_OPS:
        return TensorProto.INT64
    return None


class GraphConverter:
    """
    Converts a parsed graph into an ONNX graph.

    Each graph in a model is converted by a separate instance. Subgraphs (eg.
    the branches of an `If` operator) reference values from enclosing graphs by
    name, so no state is shared between instances.
    """

    def __init__(
        self,
        graph: Graph,
        name: str,
        input_dtypes: Optional[dict[int, int]] = None,
        output_dtypes: Optional[dict[int, int]] = None,
    ):
        """
        :param graph: The graph to convert
        :param name: Name of the generated ONNX graph
        :param input_dtypes:
            Element types for graph inputs at given positions, overriding the
            types recorded in the model. This is used for subgraphs whose
            signature is defined by the parent operator.
        :param output_dtypes: Element types for graph outputs at given positions.
        """
        self.graph = graph
        self.name = name
        self.input_dtypes = input_dtypes or {}
        self.output_dtypes = output_dtypes or {}

        self.nodes: list[NodeProto] = []
        self.initializers: list[TensorProto] = []

        # Element type of the value produced by each node, where known.
        self.elem_types: dict[int, int] = {}

        # Names of values that were renamed because they are constants which
        # are also graph outputs. See `_rename_constant_outputs`.
        self.constant_outputs: dict[int, str] = {}

        # Cache of inserted `Cast` operations, keyed by (input name, type).
        self._casts: dict[tuple[str, int], str] = {}

        self._assign_names()
        self._rename_constant_outputs()
        self._resolve_dtypes()

    def build(self) -> GraphProto:
        """Convert the graph into an ONNX graph."""

        for index, node in enumerate(self.graph.nodes):
            if isinstance(node, ConstantNode):
                self._add_initializer(index, node)

        for index, node in enumerate(self.graph.nodes):
            if isinstance(node, OperatorNode):
                self._add_operator(index, node)

        inputs = [self._value_info(id_, "input") for id_ in self.graph.inputs]
        outputs = [
            self._graph_output(position, id_)
            for position, id_ in enumerate(self.graph.outputs)
        ]

        # Constants that are graph outputs cannot be returned directly, as ONNX
        # requires each output to be produced by an operator.
        for id_, output_name in self.constant_outputs.items():
            self.nodes.append(
                helper.make_node("Identity", [self.names[id_]], [output_name])
            )

        return helper.make_graph(
            self.nodes,
            self.name,
            inputs,
            outputs,
            initializer=self.initializers,
        )

    def _assign_names(self) -> None:
        """
        Assign a unique name to each node.

        Nodes in rten models are referenced by ID rather than name, so names
        are not guaranteed to be present or unique.
        """
        kinds = {ConstantNode: "constant", OperatorNode: "operator", ValueNode: "value"}
        self.used_names: set[str] = set()
        self.names: list[str] = []

        for index, node in enumerate(self.graph.nodes):
            name = node.name
            if not name or name in self.used_names:
                name = self._unique_name(f"{kinds[type(node)]}_{index}")
            self.used_names.add(name)
            self.names.append(name)

    def _unique_name(self, base: str) -> str:
        """Generate a name that is not used by any node in this graph."""
        name = base
        suffix = 0
        while name in self.used_names:
            suffix += 1
            name = f"{base}_{suffix}"
        self.used_names.add(name)
        return name

    def _rename_constant_outputs(self) -> None:
        """
        Rename constants which are graph outputs.

        The original name is used for the output of an `Identity` operation
        added in `build`, and the constant itself is renamed.
        """
        for id_ in self.graph.outputs:
            if not isinstance(self.graph.nodes[id_], ConstantNode):
                continue
            output_name = self.names[id_]
            self.names[id_] = self._unique_name(f"{output_name}_value")
            self.constant_outputs[id_] = output_name

    def _resolve_dtypes(self) -> None:
        """Determine the element type of the value produced by each node."""

        # Values which must have a boolean type. Constants used in a boolean
        # context are converted, rather than being cast at runtime.
        bool_values: set[int] = set()
        for node in self.graph.nodes:
            if not isinstance(node, OperatorNode):
                continue
            for index in BOOL_INPUTS.get(node.op_type, ()):
                input_id = node.inputs[index] if index < len(node.inputs) else None
                if input_id is not None:
                    bool_values.add(input_id)

        for position, dtype in self.output_dtypes.items():
            if dtype == TensorProto.BOOL:
                bool_values.add(self.graph.outputs[position])

        for index, node in enumerate(self.graph.nodes):
            match node:
                case ConstantNode():
                    constant_dtype = DTYPE_FROM_NUMPY[node.data.dtype]
                    if index in bool_values and constant_dtype == TensorProto.INT64:
                        constant_dtype = TensorProto.BOOL
                    self.elem_types[index] = constant_dtype

                case ValueNode():
                    if node.dtype is not None:
                        self.elem_types[index] = DTYPE_FROM_RTEN[node.dtype]

                case OperatorNode():
                    # Operator outputs take precedence over the data type
                    # recorded for the value, as rten cannot represent all ONNX
                    # types.
                    for position, output_id in enumerate(node.outputs):
                        if output_id is None:
                            continue
                        output_dtype = self._output_dtype(node, position)
                        if output_dtype is not None:
                            self.elem_types[output_id] = output_dtype

        for position, dtype in self.input_dtypes.items():
            self.elem_types[self.graph.inputs[position]] = dtype

    def _input_dtype(self, op: OperatorNode, index: int) -> Optional[int]:
        """Return the element type of an operator input, if known."""
        input_id = op.inputs[index] if index < len(op.inputs) else None
        if input_id is None:
            return None
        return self.elem_types.get(input_id)

    def _output_dtype(self, op: OperatorNode, position: int) -> Optional[int]:
        """
        Return the element type of an operator output, if known.

        This only needs to handle cases where the type differs from the type
        recorded for the output value in the model, which is either because
        rten cannot represent the type (`bool`, `int64`) or because the model
        has no type information.
        """
        op_type = op.op_type

        if op_type in BOOL_OUTPUT_OPS:
            return TensorProto.BOOL
        if op_type in INT64_OUTPUT_OPS:
            return TensorProto.INT64
        if op_type in FLOAT_OUTPUT_OPS:
            return TensorProto.FLOAT

        match op_type:
            case "Cast":
                return DTYPE_FROM_RTEN[op.attrs.to]
            case "ConstantOfShape":
                if op.attrs.valueType == sg.Scalar.IntScalar:
                    return TensorProto.INT64
                return TensorProto.FLOAT
            case "ConvInteger" | "MatMulInteger":
                return TensorProto.INT32
            case "DynamicQuantizeLinear":
                return [TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8][
                    position
                ]
            case "Dropout":
                return TensorProto.BOOL if position == 1 else self._input_dtype(op, 0)
            case "EyeLike":
                if op.attrs.dtype is not None:
                    return DTYPE_FROM_RTEN[op.attrs.dtype]
                return self._input_dtype(op, 0)
            case "Multinomial":
                # rten does not record the output type, and ONNX defaults to
                # int32.
                return TensorProto.INT32
            case "QuantizeLinear":
                if op.attrs.outputDtype is not None:
                    return DTYPE_FROM_RTEN[op.attrs.outputDtype]
                zero_point = self._input_dtype(op, 2)
                return zero_point if zero_point is not None else TensorProto.UINT8
            case "TopK":
                return self._input_dtype(op, 0) if position == 0 else TensorProto.INT64

        if op_type in SAME_DTYPE_AS_INPUT:
            return self._input_dtype(op, SAME_DTYPE_AS_INPUT[op_type])
        if op_type in SAME_DTYPE_AS_FIRST_INPUT:
            return self._input_dtype(op, 0)

        return None

    def _add_initializer(self, index: int, constant: ConstantNode) -> None:
        """Add a constant value to the graph as an initializer."""
        data = constant.data
        match self.elem_types[index]:
            case TensorProto.INT64:
                data = data.astype(np.int64)
            case TensorProto.BOOL:
                data = data.astype(np.bool_)
        self.initializers.append(numpy_helper.from_array(data, self.names[index]))

    def _add_operator(self, index: int, op: OperatorNode) -> None:
        """Convert an operator and add it to the graph."""

        input_names = []
        for position, input_id in enumerate(op.inputs):
            if input_id is None:
                input_names.append("")
                continue
            name = self.names[input_id]
            dtype = _required_dtype(op.op_type, position)
            if dtype is not None:
                name = self._cast(name, input_id, dtype)
            input_names.append(name)

        output_names = [
            self.names[output_id] if output_id is not None else ""
            for output_id in op.outputs
        ]

        try:
            op_type, input_names, attrs = self._convert_operator(op, input_names)
        except ConversionError as ex:
            raise ConversionError(f'Error converting operator "{op.name}": {ex}')

        self.nodes.append(
            helper.make_node(
                op_type,
                _trim_empty(input_names),
                _trim_empty(output_names),
                name=self.names[index],
                **attrs,
            )
        )

    def _cast(self, name: str, node_id: int, dtype: int) -> str:
        """
        Return a value with element type `dtype`, casting `name` if needed.

        This is used for operator inputs where ONNX requires a type that rten
        does not preserve.
        """
        have = self.elem_types.get(node_id)
        if have == dtype:
            return name

        # Values with an unknown type are assumed to have the required type,
        # except for booleans, which rten never records.
        if have is None and dtype != TensorProto.BOOL:
            return name

        key = (name, dtype)
        if key not in self._casts:
            output = self._unique_name(
                f"{name}_{onnx.TensorProto.DataType.Name(dtype).lower()}"
            )
            self.nodes.append(helper.make_node("Cast", [name], [output], to=dtype))
            self._casts[key] = output
        return self._casts[key]

    def _kernel_shape(self, op: OperatorNode) -> Optional[list[int]]:
        """
        Infer the kernel shape of a convolution from its weights.

        ONNX allows the `kernel_shape` attribute to be omitted, and `.rten`
        models do not store it, but readers may use it to determine the number
        of spatial dimensions.
        """
        weights_id = op.inputs[1] if len(op.inputs) > 1 else None
        if weights_id is None:
            return None
        weights = self.graph.nodes[weights_id]
        if not isinstance(weights, ConstantNode) or len(weights.shape) < 3:
            return None
        return weights.shape[2:]

    def _add_constant(self, base_name: str, data: np.ndarray) -> str:
        """Add a generated constant to the graph and return its name."""
        name = self._unique_name(base_name)
        self.initializers.append(numpy_helper.from_array(data, name))
        return name

    def _value_info(self, node_id: int, role: str) -> ValueInfoProto:
        """Create the type declaration for a graph input or output."""
        node = self.graph.nodes[node_id]
        name = self.constant_outputs.get(node_id, self.names[node_id])

        dtype = self.elem_types.get(node_id)
        if dtype is None:
            warn_once(
                f'Model does not specify a data type for {role} "{name}". Assuming float32.'
            )
            dtype = TensorProto.FLOAT

        shape: Optional[Sequence[int | str]] = None
        if isinstance(node, (ConstantNode, ValueNode)):
            shape = node.shape

        return helper.make_tensor_value_info(name, dtype, shape)

    def _graph_output(self, position: int, node_id: int) -> ValueInfoProto:
        """Create the type declaration for a graph output."""
        dtype = self.output_dtypes.get(position)
        if dtype is not None and self.elem_types.get(node_id) != dtype:
            # The output type is dictated by the parent operator, so insert a
            # conversion. This only happens for subgraphs, where the name of
            # the output is not significant.
            name = self._cast(self.names[node_id], node_id, dtype)
            return helper.make_tensor_value_info(name, dtype, None)
        return self._value_info(node_id, "output")

    def _convert_subgraph(
        self,
        graph: Any,
        name: str,
        input_dtypes: Optional[dict[int, int]] = None,
        output_dtypes: Optional[dict[int, int]] = None,
    ) -> GraphProto:
        if not isinstance(graph, Graph):
            raise ConversionError(f'Missing subgraph "{name}"')
        return GraphConverter(graph, name, input_dtypes, output_dtypes).build()

    def _convert_operator(
        self, op: OperatorNode, input_names: list[str]
    ) -> tuple[str, list[str], dict[str, Any]]:
        """
        Convert an operator's type, inputs and attributes to ONNX.

        :return: Tuple of (operator type, input names, attributes)
        """
        attrs = op.attrs
        onnx_attrs: dict[str, Any] = {}
        op_type = op.op_type

        if attrs is None:
            # Attributes were added to the model schema for some operators
            # after support for them was added, so older models can have none.
            # The FlatBuffers defaults, which rten uses in this case, match the
            # ONNX defaults.
            match op_type:
                case "Scatter":
                    return ("ScatterElements", input_names, onnx_attrs)
                case "Upsample":
                    return _upsample_to_resize(input_names, "nearest")
            return (op_type, input_names, onnx_attrs)

        def kernel_shape():
            """Set the kernel shape for a convolution from its weights."""
            shape = self._kernel_shape(op)
            if shape is not None:
                onnx_attrs["kernel_shape"] = shape

        def pad_attrs():
            """Convert padding attributes shared by convolutions and pooling."""
            if attrs.autoPad == sg.AutoPad.Same:
                onnx_attrs["auto_pad"] = "SAME_UPPER"
            elif attrs.pads is not None:
                onnx_attrs["pads"] = _ints(attrs.pads)

        def axes_input():
            """
            Convert an `axes` attribute to an input.

            `Reduce*` operators took `axes` as an attribute until opset 18,
            when it became an input.
            """
            if attrs.axes is None or len(input_names) > 1:
                return
            axes = np.array(_ints(attrs.axes), dtype=np.int64)
            input_names.append(self._add_constant(f"{op.name}_axes", axes))

        match op_type:
            case "ArgMax" | "ArgMin":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["keepdims"] = int(attrs.keepDims)

            case "Attention":
                onnx_attrs["is_causal"] = int(attrs.isCausal)
                onnx_attrs["softcap"] = attrs.softcap
                if attrs.qNumHeads is not None:
                    onnx_attrs["q_num_heads"] = attrs.qNumHeads
                if attrs.kvNumHeads is not None:
                    onnx_attrs["kv_num_heads"] = attrs.kvNumHeads
                if attrs.scale is not None:
                    onnx_attrs["scale"] = attrs.scale

            case "AveragePool":
                onnx_attrs["kernel_shape"] = _ints(attrs.kernelSize)
                onnx_attrs["ceil_mode"] = int(attrs.ceilMode)
                onnx_attrs["count_include_pad"] = int(attrs.countIncludePad)
                pad_attrs()
                if attrs.strides is not None:
                    onnx_attrs["strides"] = _ints(attrs.strides)

            case "BatchNormalization" | "InstanceNormalization":
                onnx_attrs["epsilon"] = attrs.epsilon

            case "Cast":
                onnx_attrs["to"] = DTYPE_FROM_RTEN[attrs.to]

            case (
                "Concat"
                | "Flatten"
                | "Gather"
                | "GatherElements"
                | "LogSoftmax"
                | "OneHot"
                | "Softmax"
            ):
                onnx_attrs["axis"] = attrs.axis

            case "ConcatFromSequence":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["new_axis"] = int(attrs.newAxis)

            case "ConstantOfShape":
                if attrs.valueType == sg.Scalar.IntScalar:
                    value = np.array([attrs.value.value], dtype=np.int64)
                else:
                    value = np.array([attrs.value.value], dtype=np.float32)
                onnx_attrs["value"] = numpy_helper.from_array(value)

            case "Conv" | "ConvInteger":
                onnx_attrs["group"] = attrs.groups
                kernel_shape()
                pad_attrs()
                if attrs.dilations is not None:
                    onnx_attrs["dilations"] = _ints(attrs.dilations)
                if attrs.strides is not None:
                    onnx_attrs["strides"] = _ints(attrs.strides)

            case "ConvTranspose":
                onnx_attrs["group"] = attrs.groups
                kernel_shape()
                pad_attrs()
                if attrs.dilations is not None:
                    onnx_attrs["dilations"] = _ints(attrs.dilations)
                if attrs.outputPadding is not None:
                    onnx_attrs["output_padding"] = _ints(attrs.outputPadding)
                if attrs.strides is not None:
                    onnx_attrs["strides"] = _ints(attrs.strides)

            case "CumSum":
                onnx_attrs["exclusive"] = int(attrs.exclusive)
                onnx_attrs["reverse"] = int(attrs.reverse)

            case "DFT":
                onnx_attrs["inverse"] = int(attrs.inverse)
                onnx_attrs["onesided"] = int(attrs.onesided)

            case "DepthToSpace":
                onnx_attrs["blocksize"] = attrs.blockSize
                onnx_attrs["mode"] = _enum_value(
                    "mode", attrs.mode, DEPTH_TO_SPACE_MODES
                )

            case "DequantizeLinear" | "QuantizeLinear":
                onnx_attrs["axis"] = attrs.axis
                if op_type == "QuantizeLinear" and attrs.outputDtype is not None:
                    onnx_attrs["output_dtype"] = DTYPE_FROM_RTEN[attrs.outputDtype]

            case "Dropout":
                if attrs.seed is not None:
                    onnx_attrs["seed"] = attrs.seed

            case "Einsum":
                onnx_attrs["equation"] = attrs.equation.decode()

            case "Elu":
                onnx_attrs["alpha"] = attrs.alpha

            case "EyeLike":
                onnx_attrs["k"] = attrs.k
                if attrs.dtype is not None:
                    onnx_attrs["dtype"] = DTYPE_FROM_RTEN[attrs.dtype]

            case "GRU":
                onnx_attrs["direction"] = _enum_value(
                    "direction", attrs.direction, RNN_DIRECTIONS
                )
                onnx_attrs["hidden_size"] = attrs.hiddenSize
                onnx_attrs["linear_before_reset"] = int(attrs.linearBeforeReset)

            case "GatherND":
                onnx_attrs["batch_dims"] = attrs.batchDims

            case "Gelu":
                onnx_attrs["approximate"] = _enum_value(
                    "approximate", attrs.approximate, GELU_APPROXIMATIONS
                )

            case "Gemm":
                onnx_attrs["alpha"] = attrs.alpha
                onnx_attrs["beta"] = attrs.beta
                onnx_attrs["transA"] = int(attrs.transposeA)
                onnx_attrs["transB"] = int(attrs.transposeB)

            case "GridSample":
                onnx_attrs["align_corners"] = int(attrs.alignCorners)

            case "HardSigmoid":
                onnx_attrs["alpha"] = attrs.alpha
                onnx_attrs["beta"] = attrs.beta

            case "If":
                onnx_attrs["then_branch"] = self._convert_subgraph(
                    attrs.thenBranch, f"{op.name}_then"
                )
                onnx_attrs["else_branch"] = self._convert_subgraph(
                    attrs.elseBranch, f"{op.name}_else"
                )

            case "LSTM":
                onnx_attrs["direction"] = _enum_value(
                    "direction", attrs.direction, RNN_DIRECTIONS
                )
                onnx_attrs["hidden_size"] = attrs.hiddenSize

            case "LayerNormalization":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["epsilon"] = attrs.epsilon

            case "LeakyRelu":
                onnx_attrs["alpha"] = attrs.alpha

            case "Loop":
                # The body of a Loop takes the iteration number and a condition
                # as its first two inputs, and returns the condition for the
                # next iteration as its first output.
                onnx_attrs["body"] = self._convert_subgraph(
                    attrs.body,
                    f"{op.name}_body",
                    input_dtypes={0: TensorProto.INT64, 1: TensorProto.BOOL},
                    output_dtypes={0: TensorProto.BOOL},
                )

            case "LpNormalization":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["p"] = attrs.p

            case "MaxPool":
                onnx_attrs["kernel_shape"] = _ints(attrs.kernelSize)
                onnx_attrs["ceil_mode"] = int(attrs.ceilMode)
                pad_attrs()
                if attrs.strides is not None:
                    onnx_attrs["strides"] = _ints(attrs.strides)

            case "Mod":
                onnx_attrs["fmod"] = int(attrs.fmod)

            case "Multinomial":
                onnx_attrs["sample_size"] = attrs.sampleSize
                if attrs.seed is not None:
                    onnx_attrs["seed"] = attrs.seed

            case "NonMaxSuppression":
                onnx_attrs["center_point_box"] = int(
                    attrs.boxOrder == sg.NMSBoxOrder.CenterWidthHeight
                )

            case "Pad":
                onnx_attrs["mode"] = _enum_value("mode", attrs.mode, PAD_MODES)

            case "RandomNormal" | "RandomNormalLike":
                onnx_attrs["mean"] = attrs.mean
                onnx_attrs["scale"] = attrs.scale
                if attrs.seed is not None:
                    onnx_attrs["seed"] = attrs.seed
                if op_type == "RandomNormal":
                    onnx_attrs["shape"] = _ints(attrs.shape)

            case "RandomUniform" | "RandomUniformLike":
                onnx_attrs["high"] = attrs.high
                onnx_attrs["low"] = attrs.low
                if attrs.seed is not None:
                    onnx_attrs["seed"] = attrs.seed
                if op_type == "RandomUniform":
                    onnx_attrs["shape"] = _ints(attrs.shape)

            case _ if op_type in REDUCE_OPS:
                onnx_attrs["keepdims"] = int(attrs.keepDims)
                onnx_attrs["noop_with_empty_axes"] = int(attrs.noopWithEmptyAxes)
                axes_input()

            case "Reshape":
                onnx_attrs["allowzero"] = int(attrs.allowZero)

            case "Resize":
                onnx_attrs["mode"] = _enum_value("mode", attrs.mode, RESIZE_MODES)
                onnx_attrs["coordinate_transformation_mode"] = _enum_value(
                    "coord_mode", attrs.coordMode, COORD_TRANSFORM_MODES
                )
                onnx_attrs["nearest_mode"] = _enum_value(
                    "nearest_mode", attrs.nearestMode, NEAREST_MODES
                )

            case "ReverseSequence":
                onnx_attrs["batch_axis"] = attrs.batchAxis
                onnx_attrs["time_axis"] = attrs.timeAxis

            case "RotaryEmbedding":
                onnx_attrs["interleaved"] = int(attrs.interleaved)
                onnx_attrs["num_heads"] = attrs.numHeads
                onnx_attrs["rotary_embedding_dim"] = attrs.rotaryEmbeddingDim

            case "STFT":
                onnx_attrs["onesided"] = int(attrs.onesided)

            case "Scatter" | "ScatterElements":
                # `Scatter` was deprecated in favor of `ScatterElements`.
                op_type = "ScatterElements"
                onnx_attrs["axis"] = attrs.axis
                if op.op_type == "ScatterElements":
                    onnx_attrs["reduction"] = _enum_value(
                        "reduction", attrs.reduction, SCATTER_REDUCTIONS
                    )

            case "ScatterND":
                onnx_attrs["reduction"] = _enum_value(
                    "reduction", attrs.reduction, SCATTER_REDUCTIONS
                )

            case "SequenceEmpty":
                if attrs.dtype is not None:
                    onnx_attrs["dtype"] = DTYPE_FROM_RTEN[attrs.dtype]

            case "Shape":
                if attrs.start is not None:
                    onnx_attrs["start"] = attrs.start
                if attrs.end is not None:
                    onnx_attrs["end"] = attrs.end

            case "Split":
                onnx_attrs["axis"] = attrs.axis
                if attrs.numOutputs is not None:
                    onnx_attrs["num_outputs"] = attrs.numOutputs

            case "SplitToSequence":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["keepdims"] = int(attrs.keepDims)

            case "TopK":
                onnx_attrs["axis"] = attrs.axis
                onnx_attrs["largest"] = int(attrs.largest)
                onnx_attrs["sorted"] = int(attrs.sorted)

            case "Transpose":
                if attrs.perm is not None:
                    onnx_attrs["perm"] = _ints(attrs.perm)

            case "Trilu":
                onnx_attrs["upper"] = int(attrs.upper)

            case "Upsample":
                return _upsample_to_resize(
                    input_names, _enum_value("mode", attrs.mode, RESIZE_MODES)
                )

        return (op_type, input_names, onnx_attrs)


def _upsample_to_resize(
    input_names: list[str], mode: str
) -> tuple[str, list[str], dict[str, Any]]:
    """
    Convert an `Upsample` operator to the `Resize` operator which replaced it.

    `Resize` takes an additional "roi" input before the scales, and needs
    explicit coordinate transformation attributes to match `Upsample`.
    """
    return (
        "Resize",
        [input_names[0], ""] + input_names[1:],
        {
            "mode": mode,
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        },
    )


def _trim_empty(names: list[str]) -> list[str]:
    """Remove trailing omitted inputs or outputs from an operator."""
    while names and not names[-1]:
        names.pop()
    return names


def _has_shape(value: ValueInfoProto) -> bool:
    return value.type.tensor_type.HasField("shape")


def _infer_output_shapes(model: onnx.ModelProto) -> None:
    """
    Fill in shapes for graph outputs which the model does not specify.

    ONNX requires a shape for each of a model's outputs, even if all of the
    sizes are unknown. rten models can omit this information entirely.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as ex:
        warn_once(f"Unable to infer shapes of model outputs: {ex}")
        return

    for output, inferred_output in zip(model.graph.output, inferred.graph.output):
        if not _has_shape(output) and _has_shape(inferred_output):
            output.type.tensor_type.shape.CopyFrom(
                inferred_output.type.tensor_type.shape
            )


def onnx_model_from_graph(
    graph: Graph, metadata: Metadata, opset: int = DEFAULT_OPSET
) -> onnx.ModelProto:
    """
    Convert a parsed model into an ONNX model.

    :param graph: The model's main graph
    :param metadata: Model metadata
    :param opset: ONNX opset version to target
    """

    onnx_graph = GraphConverter(graph, "main").build()
    opset_id = helper.make_opsetid("", opset)
    model = helper.make_model(
        onnx_graph,
        producer_name="rten-convert",
        opset_imports=[opset_id],
        # Use the oldest IR version that supports the target opset, for
        # compatibility with older runtimes.
        ir_version=helper.find_min_ir_version_for([opset_id]),
    )

    if not all(_has_shape(output) for output in model.graph.output):
        _infer_output_shapes(model)

    props = {
        field.name: value
        for field in fields(Metadata)
        if (value := getattr(metadata, field.name)) is not None
    }
    if props:
        helper.set_model_props(model, props)

    return model

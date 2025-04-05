from typing import Any, Callable, Literal

import numpy as np
import onnx

from rten_convert.errors import ConversionError
from rten_convert.graph import ConstantNode, Node
from rten_convert.util import warn_once


class AttributeReader:
    """
    Utility for extracting attribute and input values from an ONNX operator.

    This keeps track of which attributes have been read so that we can warn about
    any unhandled ones.
    """

    onnx_op: onnx.OperatorProto

    add_node: Callable[[Node], int]
    """
    Function that adds a new node to the graph and returns its ID.

    This is used if a new constant node has to be generated to replace an
    operator attribute.
    """

    input_indexes: list[int | None]
    """
    IDs of the operator's input nodes.

    New inputs may be generated while reading an operator if it has an attribute
    that needs to be converted to a dynamic input.
    """

    _handled_attrs: set[str]
    """Names of attributes that have been handled."""

    def __init__(
        self,
        onnx_op: onnx.OperatorProto,
        input_indexes: list[int | None],
        add_node: Callable[[Node], int],
    ):
        self.onnx_op = onnx_op

        self.add_node = add_node
        self.input_indexes = input_indexes.copy()

        self._handled_attrs = set()

    def get_attr(self, name: str, expected_type: str, default):
        """Get the value of an optional operator attribute."""

        self._handled_attrs.add(name)

        type_code = getattr(onnx.AttributeProto, expected_type.upper())
        for attr in self.onnx_op.attribute:
            if attr.name == name:
                if attr.type != type_code:
                    raise ConversionError(
                        f"Attribute {name} type does not match {expected_type}"
                    )
                val = getattr(attr, _value_fields[type_code])

                # String attribute values are stored as bytes, so we have to decode
                # them.
                if expected_type == "string":
                    val = val.decode()

                return val
        return default

    def get_bool_attr(self, name: str, default: bool) -> bool:
        """
        Get the value of an optional boolean operator attribute.

        ONNX represents boolean attributes as "int" fields with values 0 or 1
        rather than a dedicated boolean type. This method converts these
        attributes to Python booleans.
        """
        return bool(self.get_attr(name, "int", int(default)))

    def get_enum_attr(self, name: str, enum: Any, default: str, fallback: Any = None):
        """
        Get an optional attribute whose value is an enum variant.

        The variant name is Pascal-Cased and looked up on the enum object.
        eg. `round_prefer_floor` => `RoundPreferFloor`. If the Pascal-Cased
        name matches a Python keyword, it is expected to be escaped, eg.
        `none` => `None_`.

        If the attribute value does not match any enum value, this will raise if
        `fallback` is not specified, or emit a warning and use the value
        `fallback` otherwise. Use of `fallback` is appropriate if the
        substitution is unlikely to affect the resulting model's ability to run,
        but might impact accuracy modestly.
        """

        def convert_attr(val: str):
            pascal_case = _snake_case_to_pascal_case(val)

            # Enum values that match Python keywords have a trailing underscore appended.
            escaped_pascal_case = pascal_case + "_"

            try:
                return getattr(enum, pascal_case)
            except AttributeError:
                return getattr(enum, escaped_pascal_case)

        val = self.get_attr(name, "string", default)
        try:
            return convert_attr(val)
        except AttributeError:
            if fallback:
                op = self.onnx_op.op_type
                warn_once(
                    f'Replacing unsupported value "{val}" for "{name}" attr in {op} op with "{fallback}"'
                )
                return convert_attr(fallback)
            raise ConversionError(f'Unsupported value "{val}" for "{name}" attr')

    def ignore_attr(self, name: str):
        """
        Mark an attribute as ignored.

        This is useful in cases where an attribute contains redundant information.
        """
        self._handled_attrs.add(name)

    def require_attr(self, name: str, expected_type: str):
        """Get the value of a required operator attribute."""
        val = self.get_attr(name, expected_type, default=None)
        if val is None:
            raise ConversionError(f"Missing required attribute {name}")
        return val

    def generate_input_from_attr(
        self, input_index: int, attr_name: str, attr_type: str
    ):
        """
        Generate a constant operator input from an attribute, if it exists.

        Some operator inputs changed from attributes to inputs in different ONNX
        releases. This function checks to see if an operator has an attribute
        and synthesizes a constant input.

        :param input_index: Index of the input that the attribute corresponds to
        :param attr_name: Name of the attribute
        :param attr_type: Expected type of the attribute
        """

        attr_val = self.get_attr(attr_name, attr_type, default=None)
        if attr_val is None:
            return

        if input_index < len(self.input_indexes):
            raise ConversionError(
                f'Operator has both an attribute "{attr_name}" and corresponding input at index {input_index}'
            )

        shape: list[int]
        match attr_type:
            case "int":
                shape = []
                data = np.array(attr_val).astype(np.int32)

            case "float":
                shape = []
                data = np.array(attr_val).astype(np.float32)

            case "ints":
                shape = [len(attr_val)]
                data = np.array([attr_val]).astype(np.int32)
            case _:
                raise ConversionError(
                    f'Unable to generate input from "{attr_name}" attribute of type "{attr_type}"'
                )

        generated_name = self.onnx_op.name + ":rten-" + attr_name
        const_node = ConstantNode(generated_name, shape, data)
        input_id = self.add_node(const_node)

        while len(self.input_indexes) < input_index + 1:
            self.input_indexes.append(None)
        self.input_indexes[input_index] = input_id

    def check_attr(
        self,
        name: str,
        expected_type,
        default,
        on_mismatch: Literal["raise", "warn"] = "raise",
    ):
        """
        Check if an operator has an unsupported non-default value for an attribute.

        If `default` is a tuple, it specifies a set of acceptable defaults.

        :param name: The name of the operator attribute
        :param default: The value which is equivalent to the default behavior
        :param on_mismatch:
            Whether a mismatch should be treated as a fatal error in model
            conversion or merely warn that this might cause a problem.
        """

        val = self.get_attr(name, expected_type, None)
        if val is None:
            return

        if not isinstance(default, tuple):
            default = (default,)
        if val not in default:
            msg = f"Unsupported value {val} for attribute {name}. Default is {default}"
            if on_mismatch == "raise":
                raise ConversionError(msg)
            else:
                warn_once(msg)

    def unhandled_attrs(self) -> list[onnx.AttributeProto]:
        """Return a list of attributes which have not been read."""
        return [
            attr
            for attr in self.onnx_op.attribute
            if attr.name not in self._handled_attrs
        ]


def _snake_case_to_pascal_case(s: str) -> str:
    """Transform a snake_case string to PascalCase."""
    return "".join([word[0].upper() + word[1:] for word in s.split("_")])


# Mapping of ONNX attribute types to the field on an AttributeProto which
# contains the value. Note that if you try to access the wrong field on an
# AttributeProto, you get a default value instead of an exception.
_value_fields = {
    onnx.AttributeProto.FLOAT: "f",
    onnx.AttributeProto.GRAPH: "g",
    onnx.AttributeProto.INT: "i",
    onnx.AttributeProto.INTS: "ints",
    onnx.AttributeProto.STRING: "s",
    onnx.AttributeProto.TENSOR: "t",
}

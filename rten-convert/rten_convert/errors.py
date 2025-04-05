"""Errors reported during model conversion."""


class ConversionError(Exception):
    """Errors when converting ONNX models to .rten format."""

    def __init__(self, message: str):
        super().__init__(message)


class UnsupportedOperatorError(ConversionError):
    """Conversion failed because an operator is unsupported."""

    op_type: str
    """The name of the unsupported operator, eg. `Conv`"""

    def __init__(self, op_type: str):
        self.op_type = op_type
        super().__init__(f'Unsupported operator "{op_type}"')

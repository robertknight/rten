from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import rten


def save_model(model, path):
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def add_model(path):
    """Model computing `c = a + b` for float32 tensors."""
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["c"])],
        "add",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return save_model(model, path)


def int64_model(path):
    """Model adding one to each element of an int64 tensor."""
    one = helper.make_tensor("one", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(
        [helper.make_node("Add", ["ids", "one"], ["out"])],
        "increment",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["batch", 3])],
        [helper.make_tensor_value_info("out", TensorProto.INT64, ["batch", 3])],
        initializer=[one],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return save_model(model, path)


@pytest.fixture
def add_path(tmp_path):
    return add_model(tmp_path / "add.onnx")


@pytest.fixture
def int64_path(tmp_path):
    return int64_model(tmp_path / "int64.onnx")


def test_run_returns_all_outputs_by_default(add_path):
    model = rten.Model(add_path)
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((2, 3), dtype=np.float32)

    outputs = model.run(None, {"a": a, "b": b})

    assert len(outputs) == 1
    assert outputs[0].dtype == np.float32
    np.testing.assert_array_equal(outputs[0], a + b)


def test_run_with_named_outputs(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float32)

    (output,) = model.run(["c"], {"a": a, "b": a})

    np.testing.assert_array_equal(output, a)


def test_load_from_path_object(add_path):
    model = rten.Model(Path(add_path))
    assert [node.name for node in model.get_inputs()] == ["a", "b"]


def test_load_from_bytes(add_path):
    with open(add_path, "rb") as file:
        model = rten.Model(file.read())
    assert [node.name for node in model.get_inputs()] == ["a", "b"]


def test_load_invalid_model():
    with pytest.raises(rten.RtenError):
        rten.Model(b"not a model")


def test_get_inputs_and_outputs(int64_path):
    model = rten.Model(int64_path)

    (input_node,) = model.get_inputs()
    assert input_node.name == "ids"
    # RTen converts int64 tensors to int32 when a model is loaded.
    assert input_node.type == "tensor(i32)"
    assert input_node.shape == ["batch", 3]

    (output_node,) = model.get_outputs()
    assert output_node.name == "out"
    assert output_node.type == "tensor(i32)"


def test_int64_input_is_converted(int64_path):
    model = rten.Model(int64_path)
    ids = np.array([[1, 2, 3]], dtype=np.int64)

    (output,) = model.run(None, {"ids": ids})

    assert output.dtype == np.int32
    np.testing.assert_array_equal(output, ids + 1)


def test_int64_input_out_of_range(int64_path):
    model = rten.Model(int64_path)
    ids = np.array([[1, 2, 2**40]], dtype=np.int64)

    with pytest.raises(OverflowError, match="does not fit in an int32"):
        model.run(None, {"ids": ids})


def test_bool_input_is_converted(int64_path):
    model = rten.Model(int64_path)
    mask = np.array([[True, False, True]])

    (output,) = model.run(None, {"ids": mask})

    np.testing.assert_array_equal(output, [[2, 1, 2]])


def test_float64_input_is_converted(add_path):
    model = rten.Model(add_path)
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.ones((2, 3), dtype=np.float32)

    (output,) = model.run(None, {"a": a, "b": b})

    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, a + b)


def test_scalar_input(tmp_path):
    """A 0-dimensional input is passed through as a scalar tensor."""
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "scale"], ["y"])],
        "scale",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )
    path = save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        tmp_path / "scale.onnx",
    )
    model = rten.Model(path)

    (output,) = model.run(
        None,
        {
            "x": np.array([1, 2, 3], dtype=np.float32),
            "scale": np.array(10, dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(output, [11, 12, 13])


def test_non_contiguous_input(add_path):
    model = rten.Model(add_path)
    a = np.arange(6, dtype=np.float32).reshape(3, 2).transpose()
    b = np.ones((2, 3), dtype=np.float32)

    (output,) = model.run(None, {"a": a, "b": b})
    (expected,) = model.run(None, {"a": np.ascontiguousarray(a), "b": b})

    np.testing.assert_array_equal(output, expected)


def test_unknown_input_name(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no node named"):
        model.run(None, {"a": a, "nonexistent": a})


def test_unknown_output_name(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="no node named"):
        model.run(["nonexistent"], {"a": a, "b": a})


def test_unsupported_dtype(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float16)

    with pytest.raises(TypeError, match="unsupported dtype"):
        model.run(None, {"a": a, "b": a})


def test_input_must_be_an_array(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="must be a numpy array"):
        model.run(None, {"a": a, "b": [[1, 2, 3], [4, 5, 6]]})


def test_input_with_wrong_shape(add_path):
    model = rten.Model(add_path)
    a = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(rten.RtenError):
        model.run(None, {"a": a, "b": np.zeros((5, 5), dtype=np.float32)})


def test_repr(add_path):
    model = rten.Model(add_path)

    assert repr(model) == "Model(inputs=[a, b], outputs=[c])"
    expected = 'NodeInfo(name="a", type="tensor(f32)", shape=[2, 3])'
    assert repr(model.get_inputs()[0]) == expected

import json
import os

from torch import Tensor
import torch
import torch.nn as nn

from .common import tensor_json, params_json


def gen_test_case(
    module: nn.Module, inputs: Tensor, initial: tuple[Tensor, Tensor] | None = None
) -> dict:
    output, (last_hidden, last_cell) = module(x, initial)
    case = {
        "input": tensor_json(x),
        "output": tensor_json(output),
        "params": params_json(module),
    }
    if initial:
        case["initial_hidden"] = tensor_json(initial[0])
        case["initial_cell"] = tensor_json(initial[1])

    return case


# Ensure we get the same output on every run.
torch.manual_seed(1234)

input_features = 10
hidden_size = 5
seq_len = 7

lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size)
x = torch.rand((seq_len, input_features))
initial_hidden = torch.rand((1, hidden_size))
initial_cell = torch.rand((1, hidden_size))

lstm_bidirectional = nn.LSTM(
    input_size=input_features, hidden_size=hidden_size, bidirectional=True
)

test_cases = {
    "__comment__": f"Generated with {os.path.basename(__file__)}",
    "lstm_forwards": gen_test_case(lstm, x),
    "lstm_bidirectional": gen_test_case(lstm_bidirectional, x),
    "lstm_initial": gen_test_case(lstm, x, (initial_hidden, initial_cell)),
}

script_dir = os.path.dirname(__file__)
with open(f"{script_dir}/lstm.json", "w") as f:
    json.dump(test_cases, f, indent=2)

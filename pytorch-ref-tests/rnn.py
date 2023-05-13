import json
import os

from torch import Tensor
import torch
import torch.nn as nn

from .common import tensor_json, params_json


def gen_lstm_test_case(
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


def gen_gru_test_case(
    module: nn.Module, inputs: Tensor, initial: Tensor | None = None
) -> dict:
    (
        output,
        last_hidden,
    ) = module(x, initial)
    case = {
        "input": tensor_json(x),
        "output": tensor_json(output),
        "params": params_json(module),
    }
    if initial is not None:
        case["initial_hidden"] = tensor_json(initial)
    return case


# Ensure we get the same output on every run.
torch.manual_seed(1234)

input_features = 10
hidden_size = 5
seq_len = 7

x = torch.rand((seq_len, input_features))
initial_hidden = torch.rand((1, hidden_size))
initial_cell = torch.rand((1, hidden_size))

lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size)
lstm_bidirectional = nn.LSTM(
    input_size=input_features, hidden_size=hidden_size, bidirectional=True
)

gru = nn.GRU(input_size=input_features, hidden_size=hidden_size)
gru_bidirectional = nn.GRU(
    input_size=input_features, hidden_size=hidden_size, bidirectional=True
)

test_cases = {
    "__comment__": f"Generated with {os.path.basename(__file__)}",
    "lstm_forwards": gen_lstm_test_case(lstm, x),
    "lstm_bidirectional": gen_lstm_test_case(lstm_bidirectional, x),
    "lstm_initial": gen_lstm_test_case(lstm, x, (initial_hidden, initial_cell)),
    "gru_forwards": gen_gru_test_case(gru, x),
    "gru_bidirectional": gen_gru_test_case(gru_bidirectional, x),
    "gru_initial": gen_gru_test_case(gru, x, initial_hidden),
}

script_dir = os.path.dirname(__file__)
with open(f"{script_dir}/rnn.json", "w") as f:
    json.dump(test_cases, f, indent=2)

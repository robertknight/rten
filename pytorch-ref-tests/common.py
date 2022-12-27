from torch import Tensor
import torch.nn as nn

def tensor_json(x: Tensor):
    """
    Convert a tensor to a JSON-serializable representation.
    """
    return [list(x.shape), x.flatten().tolist()]


def params_json(m: nn.Module):
    """
    Convert a PyTorch module's parameters to a JSON-serializable dict.
    """
    params_dict = {}
    for name, tensor in m.state_dict().items():
        params_dict[name] = tensor_json(tensor)
    return params_dict



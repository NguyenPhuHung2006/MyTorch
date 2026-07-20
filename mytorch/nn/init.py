from ..tensor import Tensor
import numpy as np
import math

def _calculate_fan_in_and_fan_out(tensor: Tensor):
    if tensor.ndim < 2:
        raise ValueError(
            "Fan in and fan out can only be computed for tensors with at least 2 dimensions."
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def calculate_gain(nonlinearity: str, param=None):
    nonlinearity = nonlinearity.lower()

    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
        return 1.0

    if nonlinearity == "tanh":
        return 5.0 / 3

    if nonlinearity == "relu":
        return math.sqrt(2.0)

    if nonlinearity == "leaky_relu":
        negative_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

def zeros_(tensor: Tensor):
    tensor.data[...] = np.zeros_like(tensor.data)
    return tensor

def ones_(tensor: Tensor):
    tensor.data[...] = np.ones_like(tensor.data)
    return tensor

def constant_(tensor: Tensor, val):
    tensor.data[...] = np.full_like(tensor.data, val)
    return tensor

def uniform_(tensor: Tensor, a=0.0, b=1.0):
    tensor.data[...] = np.random.uniform(a, b, tensor.shape)
    return tensor

def normal_(tensor: Tensor, mean=0.0, std=1.0):
    tensor.data[...] = np.random.normal(mean, std, tensor.shape)
    return tensor

def xavier_uniform_(tensor: Tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    return uniform_(tensor, -bound, bound)

def xavier_normal_(tensor: Tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    return normal_(tensor, 0.0, std)

def _calculate_correct_fan(tensor: Tensor, mode: str):
    mode = mode.lower()

    if mode not in ("fan_in", "fan_out"):
        raise ValueError(
            f"Mode {mode} not supported, please use 'fan_in' or 'fan_out'"
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    return fan_in if mode == "fan_in" else fan_out

def kaiming_uniform_(
    tensor: Tensor,
    a=0.0,
    mode="fan_in",
    nonlinearity="leaky_relu",
):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    return uniform_(tensor, -bound, bound)

def kaiming_normal_(
    tensor: Tensor,
    a=0.0,
    mode="fan_in",
    nonlinearity="leaky_relu",
):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    return normal_(tensor, 0.0, std)

def uniform_bias_(bias: Tensor, weight: Tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(weight)
    bound = 1.0 / math.sqrt(fan_in)
    return uniform_(bias, -bound, bound)
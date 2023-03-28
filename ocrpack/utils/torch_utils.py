import torch
from torch import Tensor, nn

activation_layers = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "none": nn.Identity
}


normalization_layers = {
    "batchnorm": nn.BatchNorm2d,
    "layernorm": nn.LayerNorm,
    "none": None,
}


def get_act_layer(name: str):
    name: str = name.lower()
    assert name in activation_layers, f"Activation Layer: {name} is not implemented!"

    return activation_layers[name]


def get_norm_layer(name: str):
    name = name.lower()
    assert name in normalization_layers, f"Normalization Layer: {name} is not implemented!"

    return normalization_layers[name]


def auto_pad(k: int, p: int = None, d: int = 1) -> str:
    if p is None:
        if isinstance(k, int):
            p: int = (k - 1) // 2 * d
        else:
            p: list = [(x - 1) // 2 * d[i] for i, x in enumerate(k)]
    return p

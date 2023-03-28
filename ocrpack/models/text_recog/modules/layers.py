import math

import torch
from torch import Tensor, nn

from ocrpack.models.text_recog.modules import ConvNormAct
from ocrpack.utils import get_norm_layer, to_tuple


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class ImagePatch(nn.Module):
    """
    Embedding Image to Patches Layer
    """

    def __init__(
        self,
        embed_dim: int = 768,
        img_sz: int = 224,
        patch_sz: int = 16,
        kernel: int = None,
        stride: int = None,
        padding: int = None,
        norm: str = None,
        flatten: bool = True,
        bias: bool = True,
        multi_conv: bool = False,
    ) -> None:
        super().__init__()

        self.img_size = to_tuple(img_sz)
        self.patch_size = to_tuple(patch_sz)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        in_chs = 3
        kernel = self.patch_size if kernel is None else kernel
        stride = self.patch_size if stride is None else stride
        padding = 0 if padding is None else padding

        if multi_conv:
            if self.patch_size[0] == 12:
                self.proj = nn.Sequential(
                    ConvNormAct(in_chs, embed_dim // 4, kernel=7, stride=4, padding=3),
                    ConvNormAct(embed_dim // 4, embed_dim // 2, kernel=3, stride=3, padding=0),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1)
                )
            elif self.patch_size == 16:
                self.proj = nn.Sequential(
                    ConvNormAct(in_chs, embed_dim // 4, kernel=7, stride=4, padding=3),
                    ConvNormAct(embed_dim // 4, embed_dim // 2, kernel=3, stride=2, padding=1),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
                )
        else:
            self.proj = nn.Conv2d(in_chs, embed_dim, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
        self.norm = get_norm_layer(norm)(embed_dim) if norm else None

    def forward(self, x: Tensor, return_shape: bool = False) -> Tensor:
        b, c, h, w = x.shape

        assert h == self.img_size[0], f"Wrong image height! Expected {self.img_size[0]} but got {h}!"
        assert w == self.img_size[1], f"Wrong image woidth! Expected {self.img_size[1]} but got {w}!"

        x = self.proj(x)
        if self.flatten:  # B C H W -> B HW C
            x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)

        if return_shape:
            return x, x.shape[-2:]
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        chs: int,
        init_values: float = 1e-5,
        inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(chs))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """
    Drop Paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    """

    def __init__(
        self, drop_prob: float = 0.0, scale_by_keep: bool = True
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"

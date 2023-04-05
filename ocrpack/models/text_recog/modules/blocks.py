from typing import Any
import torch
from torch import Tensor, nn

from ocrpack.utils.torch_utils import auto_pad, get_act_layer, get_norm_layer


class ConvNormAct(nn.Module):
    """
    Standard (Convolution + Batch Norm + Activation) Module
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel: int,
        stride: int,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        norm: str = None,
        act: str = "relu"
    ) -> None:
        super().__init__()

        if bias is None:
            bias = norm is None

        padding = auto_pad(kernel, padding, dilation)

        self.conv = nn.Conv2d(
            in_chs, out_chs,
            kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.bn = get_norm_layer(norm)(out_chs) if norm else None

        self.act = get_act_layer(act)(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    def fuse_forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv(x))
        return x


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        HW: list[int, int] = [8, 25],
        local_k: list[int, int] = [3, 3]
    ) -> None:
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=local_k, stride=1,
            padding=[local_k[0] // 2, local_k[1] // 2],
            groups=num_heads
        )

    def forward(self, x: Tensor) -> Any | Tensor:
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron Module
    """

    def __init__(
        self,
        in_chs: int,
        hidden_chs: int = None,
        out_chs: int = None,
        act: str = "gelu",
        drop_prob: float = 0.0,
        inplace: bool = None,
        bias: bool = True
    ) -> None:
        super().__init__()

        out_chs = out_chs or in_chs
        hidden_chs = hidden_chs or in_chs

        param = {} if inplace is None else {"inplace": inplace}

        self.fc1 = nn.Linear(in_chs, hidden_chs, bias=bias)
        self.act1 = get_act_layer(act)(**param)
        self.drop1 = nn.Dropout(drop_prob, **param)
        self.fc2 = nn.Linear(hidden_chs, out_chs, bias=bias)
        self.drop2 = nn.Dropout(drop_prob, **param)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        chs: int,
        num_heads: int = 8,
        mixer: str = 'Global',
        HW: list[int, int] = None,
        local_k: list[int, int] = [7, 11],
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ) -> None:
        super().__init__()
        assert chs % num_heads == 0, "Channels should be divisible by num_heads"

        self.chs = chs
        self.num_heads = num_heads
        self.scale = (chs // num_heads) ** -0.5

        self.qkv = nn.Linear(chs, chs * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(chs, chs)
        self.proj_drop = nn.Dropout(proj_drop)

        self.HW = HW
        if mixer == 'Local' and HW is not None:
            H = HW[0]
            W = HW[1]
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(
                [H * W, H + hk - 1, W + wk - 1], dtype=torch.float32
            )
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w+wk] = 0.
            mask_flatten = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], -(torch.inf), dtype=torch.float32)
            mask = torch.where(mask_flatten < 1, mask_flatten, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x: Tensor) -> Tensor:
        if self.HW is not None:
            H = self.HW[0]
            W = self.HW[1]
            N = H * W
            C = self.chs
        else:
            B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.mixer == 'Local':
            attn += self.mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

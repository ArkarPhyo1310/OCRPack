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
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ) -> None:
        super().__init__()
        assert chs % num_heads == 0, "Channels should be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = (chs // num_heads) ** -0.5

        self.qkv = nn.Linear(chs, chs * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(chs, chs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

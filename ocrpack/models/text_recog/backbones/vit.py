import torch
from torch import Tensor, nn

from ocrpack.models.text_recog.modules.blocks import AttentionBlock, MLPBlock
from ocrpack.models.text_recog.modules.layers import (DropPath, ImagePatch,
                                                       LayerScale)
from ocrpack.utils import get_norm_layer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        chs: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_out: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = None,
        drop_path: float = 0.0,
        act: str = "gelu",
        norm: str = "layernorm"
    ) -> None:
        super().__init__()

        self.norm1 = get_norm_layer(norm)(chs)
        self.attn = AttentionBlock(chs, num_heads, qkv_bias, attn_drop, proj_drop=drop_out)
        self.ls1 = LayerScale(chs, init_values) if init_values else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else None

        self.norm2 = get_norm_layer(norm)(chs)
        self.mlp = MLPBlock(chs, hidden_chs=int(chs * mlp_ratio), act=act, drop_prob=drop_out)
        self.ls2 = LayerScale(chs, init_values) if init_values else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = self.ls1(x1) if self.ls1 else x1
        x1 = self.drop_path1(x1) if self.drop_path1 else x1

        x2 = x + x1

        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        x3 = self.ls2(x3) if self.ls2 else x3
        x3 = self.drop_path2(x3) if self.drop_path2 else x3

        x = x2 + x3

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer

    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
     - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_sz: int = 224,
        patch_sz: int = 16,
        num_classes: int = 1000,
        num_heads: int = 12,
        depth: int = 12,
        embed_dim: int = 768,
        mlp_ratio: int = 4,
        init_values: float = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        qkv_bias: bool = True,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: bool = None,
        global_pool: str = "token",
        norm: str = None,
        act: str = None
    ) -> None:
        """

        Args:
            img_sz (int, optional): Input Image Size. Defaults to 224.
            patch_sz (int, optional): Patch Size. Defaults to 16.
            num_classes (int, optional): Num of classes for classification head. Defaults to 1000.
            num_heads (int, optional): Num of Attention head. Defaults to 12.
            depth (int, optional): Depth of transformer. Defaults to 12.
            embed_dim (int, optional): Embedding Dimension. Defaults to 768.
            mlp_ratio (int, optional): Ratio of MLP hidden dim to embedding dim. Defaults to 4.
            init_values (float, optional): Layer-scale init values. Defaults to None.
            drop_rate (float, optional): Dropout Rate. Defaults to 0..
            attn_drop_rate (float, optional): Attention Dropout Rate. Defaults to 0..
            drop_path_rate (float, optional): Stochastic Depth Rate. Defaults to 0..
            qkv_bias (bool, optional): Enable bias for QKV if True. Defaults to True.
            class_token (bool, optional): Use class Token. Defaults to True.
            no_embed_class (bool, optional): _description_. Defaults to False.
            pre_norm (bool, optional): Perform pre-normalization. Defaults to False.
            fc_norm (bool, optional): Pre-FC norm after pool. Defaults to None.
            global_pool (str, optional): Type of global pooling for final sequence. Defaults to "token".
            act (str, optional): Activation Layer Name. Defaults to None.
            act (str, optional): Activation Layer Name. Defaults to None.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm = "layernorm" if norm is None else norm
        act = "gelu" if act is None else act
        norm_layer = get_norm_layer(norm)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class

        self.patch_embedding = ImagePatch(
            embed_dim=embed_dim,
            img_sz=img_sz,
            patch_sz=patch_sz,
            bias=not pre_norm
        )
        num_patches = self.patch_embedding.num_patches
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_dpout = nn.Dropout(drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else None

        # stochasitc depth decay rule
        dppath_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.encoder_block = nn.Sequential(
            *[EncoderBlock(
                chs=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_out=drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                drop_path=dppath_rate[i],
                norm=norm,
                act=act,
            ) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else None

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else None
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else None

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_dpout(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x) if self.norm_pre else x
        x = self.encoder_block(x)
        x = self.norm(x) if self.norm else x
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x) if self.fc_norm else x
        x = self.head(x) if self.head else x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


if __name__ == "__main__":
    import timm
    vit = VisionTransformer()
    v_sd = vit.state_dict().copy()

    timm_vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    t_sd = timm_vit.state_dict()

    for v_key, t_key in zip(v_sd.keys(), t_sd.keys()):
        v_sd[v_key] = t_sd[t_key]

    vit.load_state_dict(v_sd)
    torch.save(vit.half().state_dict(), "vit_base_16_224.pt")

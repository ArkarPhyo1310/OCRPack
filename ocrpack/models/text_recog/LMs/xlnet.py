from torch import Tensor, nn

from ocrpack.utils.torch_utils import get_act_layer


class XLNetDecoder(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        out_dim: int = 2048,
        drop_rate: float = 0.1,
        act: str = "gelu",
        layer_norm_eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)

        self.linear1 = nn.Linear(embed_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)

        self.act_layer = get_act_layer(act)()

    def forward_stream(
        self,
        tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor,
        tgt_mask: Tensor, tgt_key_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv,
            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.act_layer(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query: Tensor, content: Tensor, memory: Tensor,
        query_mask: Tensor = None, content_mask: Tensor = None, content_key_padding_mask: Tensor = None,
        update_content: Tensor = True
    ) -> tuple[Tensor, Tensor]:
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query, query_norm, content_norm, memory,
            query_mask, content_key_padding_mask
        )[0]
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory,
                content_mask, content_key_padding_mask
            )[0]

        return query, content

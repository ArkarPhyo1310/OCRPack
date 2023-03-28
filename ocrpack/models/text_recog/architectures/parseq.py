import math
from itertools import permutations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ocrpack.data.tokenzier import Tokenizer
from ocrpack.models.text_recog.backbones import VisionTransformer
from ocrpack.models.text_recog.LMs import XLNetDecoder
from ocrpack.models.text_recog.modules import TokenEmbedding
from ocrpack.utils.torch_utils import get_norm_layer


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        qkv_bias: bool = True
    ) -> None:
        super().__init__()
        self.backbone = VisionTransformer(
            img_sz=img_size,
            patch_sz=patch_size,
            num_classes=0,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            qkv_bias=qkv_bias,
            global_pool="",
            class_token=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone.forward_features(x)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float = 0.,
        norm: str = "layernorm",
        act: str = "gelu"
    ) -> None:
        super().__init__()
        self.decoder_block = nn.ModuleList([
            XLNetDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                out_dim=int(embed_dim * mlp_ratio),
                drop_rate=drop_rate,
                act=act
            ) for _ in range(depth)
        ])
        self.norm_layer = get_norm_layer(norm)(embed_dim)

    def forward(
        self,
        query: Tensor, content: Tensor, memory: Tensor,
        query_mask: Tensor = None, content_mask: Tensor = None, content_key_padding_mask: Tensor = None,
    ) -> Tensor:
        for i, module in enumerate(self.decoder_block):
            last = i == len(self.decoder_block) - 1
            query, content = module(
                query, content, memory,
                query_mask, content_mask, content_key_padding_mask,
                update_content=not last
            )
        query = self.norm_layer(query)
        return query


class PARSeq(nn.Module):
    def __init__(self, data_cfg, model_cfg, hyp_cfg) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer(data_cfg.charset_train)
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id

        self.model_cfg = model_cfg
        encoder_cfg = model_cfg.encoder
        decoder_cfg = model_cfg.decoder
        embed_dim = model_cfg.embed_dim

        self.encoder = Encoder(
            img_size=model_cfg.image_size,
            patch_size=model_cfg.patch_size,
            embed_dim=embed_dim,
            depth=encoder_cfg.depth,
            num_heads=encoder_cfg.num_heads,
            mlp_ratio=encoder_cfg.mlp_ratio,
            drop_rate=encoder_cfg.drop_rate,
            attn_drop_rate=encoder_cfg.attn_drop_rate,
            drop_path_rate=encoder_cfg.drop_path_rate,
            qkv_bias=encoder_cfg.qkv_bias
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            depth=decoder_cfg.depth,
            num_heads=decoder_cfg.num_heads,
            mlp_ratio=decoder_cfg.mlp_ratio,
            drop_rate=decoder_cfg.drop_rate,
            norm=decoder_cfg.norm,
            act=decoder_cfg.activation
        )

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(Tensor(1, model_cfg.max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(hyp_cfg.drop_rate)

        # Permutation/Attention Mask
        self.rng = np.random.default_rng()
        self.max_gen_perms = model_cfg.perm_num // 2 if model_cfg.perm_mirror else model_cfg.perm_num
        self.perm_forward = model_cfg.perm_forward
        self.perm_mirrored = model_cfg.perm_mirror

        nn.init.trunc_normal_(self.pos_queries, std=.02)

    def encode(self, img: Tensor) -> Tensor:
        return self.encoder(img)

    def decode(
        self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None,
        tgt_padding_mask: Tensor = None, tgt_query: Tensor = None,
        tgt_query_mask: Tensor = None
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_embed = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_embed = self.dropout(torch.cat([null_ctx, tgt_embed], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_embed, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: int = None):
        testing = max_length is None
        max_length = self.model_cfg.max_label_length if max_length is None else min(max_length, self.model_cfg.max_label_length)
        bs = images.shape[0]

        # +1 for <eos> at end of sequence
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to "num_steps"
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `gen_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self.device), 1)

        if self.model_cfg.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self.device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient Decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # The next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # Greedy decode. Add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch Decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self.device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.model_cfg.refine_iters:
            # For iterative refinement we always use a "cloze" mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self.device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self.device)
            for i in range(self.model_cfg.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

    def forward_train(self, images, tgt):
        loss = 0
        loss_numel = 0
        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.gen_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel
        return loss

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
            This works because the same attention mask can be used for the shorter sequences 
            because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self.device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=self.device)] if self.perm_forward else []
        # Additional Permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the label in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)), device=self.device
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perm_pool), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self.device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)

        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS is T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self.device)

        return perms

    def gen_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)

        Args:
            perm: The permutation sequence. i = 0 is always the BOS

        Returns:
            lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self.device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float("-inf")
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self.device)] = float("-inf")
        query_mask = mask[1:, :-1]
        return content_mask, query_mask


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from PIL import Image
    from strhub.data.module import SceneTextDataModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load("./configs/parseq.yaml")
    parseq = PARSeq(cfg.data, cfg.model, cfg.hypermeters)
    parseq_sd = parseq.state_dict()

    parseq_hub = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
    parseq_hub_sd = parseq_hub.state_dict()

    for n, o in zip(parseq_sd.keys(), parseq_hub_sd.keys()):
        parseq_sd[n] = parseq_hub_sd[o]

    parseq.load_state_dict(parseq_sd)
    parseq.eval()
    parseq.to(device)

    print(f"Image Size: \t{parseq_hub.hparams.img_size}")
    img_transform = SceneTextDataModule.get_transform(parseq_hub.hparams.img_size)

    image = Image.open("./assets/test.jpg").convert("RGB")
    image = img_transform(image).unsqueeze(0)

    logits = parseq(image.to(device))
    print(logits.shape)

    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    print(f"Decoded Label: \t{label[0]}")

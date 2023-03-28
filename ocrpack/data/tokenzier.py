from abc import ABC, abstractmethod
from itertools import groupby

import torch
from torch.nn.utils.rnn import pad_sequence


class BaseTokenizer(ABC):
    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self) -> int:
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: list[int], join: bool = True):
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: list[str], device: torch.device = None) -> torch.Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels (list[str]): List of labels. Each can be of arbitrary length.
            device (torch.device, optional): Create Tensor on this device. Defaults to None.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: torch.Tensor, ids: torch.Tensor) -> tuple[list[str], list[torch.Tensor]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: torch.Tensor, raw: bool = False) -> tuple[list[str], list[torch.Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists (torch.Tensor): softmax probabilities over the token distribution. Shape: NxLxC
            raw (bool, optional): Return unprocessed labels (will return list of list of strings). Defaults to False.

        Returns:
            list: list of string labels (arbitrary length)
            list: their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)

        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: list[str], device: torch.device = None) -> torch.Tensor:
        batch = [
            torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: torch.Tensor, ids: torch.Tensor) -> tuple[list[str], list[torch.Tensor]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except:
            eos_idx = len(ids)

        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]
        return probs, ids


class CTCTokenizer(BaseTokenizer):
    BLANK = "[B]"

    def __init__(self, charset: str) -> None:
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: list[str], device: torch.device = None) -> torch.Tensor:
        batch = [
            torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: torch.Tensor, ids: torch.Tensor) -> tuple[list[str], list[torch.Tensor]]:
        ids = list(zip(*groupby(ids.tolist())))[0]      # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]    # Remove Blanks

        return probs, ids

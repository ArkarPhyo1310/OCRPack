import re
import unicodedata

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        max_label_length: int, charset: str,
        normalize_unicode: bool = True, remove_whitespace: bool = True
    ) -> None:
        super().__init__()

        self.max_label_length = max_label_length
        self.normalize_unicode = normalize_unicode
        self.remove_whitespace = remove_whitespace
        self.unsupported = f'[^{re.escape(charset)}]'

    def _process_label(self, label: str):
        if self.remove_whitespace:
            label = "".join(label.split())

        if self.normalize_unicode:
            label = unicodedata.normalize('NFKD', label).encode("utf-8", "ignore").decode()

        filtered_label = re.sub(self.unsupported, "", label)

        if len(label) > self.max_label_length or not filtered_label:
            return None

        return label
    
    
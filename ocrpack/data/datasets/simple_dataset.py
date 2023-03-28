import csv
import unicodedata

import torch
from PIL import Image

from ocr_pack.data.datasets.base import BaseDataset


class SimpleDataset(BaseDataset):
    def __init__(
        self,
        data_file: str, charset: str, max_label_length: int = 25, delimiter: str = "\t",
        normalize_unicode: bool = True, remove_whitespace: bool = True, transform=None
    ) -> None:
        super().__init__(max_label_length=max_label_length, charset=charset,
                         normalize_unicode=normalize_unicode, remove_whitespace=remove_whitespace
                         )

        self.normalize_unicode = normalize_unicode
        self.remove_whitespace = remove_whitespace
        self.transform = transform if isinstance(transform, list) else [transform] if transform else transform

        with open(data_file, encoding="utf-8") as file:
            file_content = csv.reader(file, delimiter=delimiter)

        self._content = list(file_content)
        self._preprocess_data()

    def _preprocess_data(self):
        self._image_list = []
        self._label_list = []
        for data in self._content:
            label = self._process_label(data[1])
            if not label:
                continue
            self._image_list.append(data[0])
            self._label_list.append(label)

    def __len__(self):
        return len(self._label_list)

    def __getitem__(self, index: int):
        image = self._image_list[index]
        label = self._label_list[index]

        img = Image.open(image).convert("RGB")
        if self.transform:
            for t in self.transform:
                img = t(img)

        return img, label

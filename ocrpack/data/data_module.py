from typing import Optional
import pytorch_lightning as pl
from ocrpack.data.data_augment import BackgroundEffect, DefaultTransform, OCRAugmentPipeline

from ocrpack.data.datasets.simple_dataset import SimpleDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.cfg = data_cfg

        # Data Transformation
        bg_effect = BackgroundEffect()
        ocr_augment = OCRAugmentPipeline()
        default_augment = DefaultTransform(self.cfg.image_size)

        self.train_transform = [bg_effect, ocr_augment, default_augment]
        self.val_transform = self.test_tranform = default_augment

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = SimpleDataset(self.cfg.train_txt, self.cfg.charset_train, transform=self.train_transform)
        self.val_set = SimpleDataset(self.cfg.val_txt, self.cfg.charset_test, transform=self.val_transform)
        self.test_set = self.val_set
        if self.cfg.test_txt:
            self.test_set = SimpleDataset(self.cfg.test_txt, self.cfg.charset_test, transform=self.test_tranform)

    def train_dataloader(self):
        return 

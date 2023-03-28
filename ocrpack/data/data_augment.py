import os
import random
from typing import Any

import albumentations as A
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageOps


class DefaultTransform:
    def __init__(self, img_sz=[32, 128], mean=0.5, std=0.5) -> None:
        self.transform = T.Compose([
            T.Resize(img_sz, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, img) -> Any:
        img = self.transform(img)
        return img


class OCRAugmentPipeline:
    def __init__(self, pad_width: int = 50, pad_height: int = 50) -> None:
        self.transform = A.Compose([
            A.PadIfNeeded(pad_width, pad_height),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.GaussNoise(),
            A.PiecewiseAffine(),
            A.Rotate((0, 30)),
            A.CLAHE(),
            A.ColorJitter(),
            A.ChannelDropout(),
            A.ChannelShuffle(),
            A.FancyPCA()
        ])

    def __call__(self, image):
        img = self.transform(image=image)["image"]
        return img


class BackgroundEffect:
    def __init__(self, noise_prob: float = 0.5, color_prob: float = 0.5, image_prob: float = 0.5) -> None:
        self._noise_prob = noise_prob
        self._color_prob = color_prob
        self._image_prob = image_prob
        self._effects = [
            self._add_gaussian_noise(),
            self._add_color(),
            self._add_image()
        ]

    def __call__(self, image) -> Image:
        self.foreground = Image.open(image).convert("RGB")
        self.height = self.foreground.height
        self.width = self.foreground.width
        self.inverted_img = ImageOps.invert(self.foreground)
        self.bg_img = self._one_of_effects()
        self.bg_img.paste(self.inverted_img, (0, 0), self.foreground)

        return self.bg_img

    def _one_of_effects(self) -> Image:
        effect_ps = [self._noise_prob, self._color_prob, self._image_prob]
        sum_ps = sum(effect_ps)
        effect_ps = [p / sum_ps for p in effect_ps]
        idx = np.random.choice(len(effect_ps), None, True, effect_ps)
        return self._effects[idx]

    def _add_gaussian_noise(self, mean=235, std=10) -> Image:
        """
        Create a background with Gaussian noise (to mimic paper)
        """
        back_img = np.random.normal(mean, std, size=(self.height, self.width))
        return Image.fromarray(back_img).convert("RGBA")

    def _add_color(self, color: tuple = (242, 238, 203)) -> Image:
        return Image.new("RGB", (self.width, self.height), color).convert("RGBA")

    def _add_image(self, path: str = "./data/bg_effect"):
        images = os.listdir(path) if os.path.isdir(path) else [path] if os.path.isfile(path) else []

        if len(images) > 0:
            back_img = Image.open(
                os.path.join(path, images[random.randint(0, len(images) - 1)])
            )

            if back_img.size[0] < self.width or back_img.size[1] < self.height:
                back_img = back_img.resize((self.width, self.height), Image.ANTIALIAS)

            if back_img.size[0] == self.width:
                x = 0
            else:
                x = random.randint(0, back_img.size[0] - self.width)

            if back_img.size[1] == self.height:
                y = 0
            else:
                y = random.randint(0, back_img.size[1] - self.height)

            return back_img.crop((x, y, x + self.width, y + self.height))
        else:
            return self._add_color(color=(255, 255, 255))

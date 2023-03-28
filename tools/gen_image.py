#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob
import os
import random
import re

from rich.progress import track
from fontTools import ttLib

if os.name == "posix":
    import pyvips
else:
    vipshome = 'D:\\Dev_pkgs\\vips-dev\\vips-dev-8.13\\bin'
    os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
    import pyvips


def gen_image(file: str, from_wordlist: bool = True):
    gt = open("./data/gt.txt", "a", encoding="utf-8")
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()

    if from_wordlist:
        word_list = data.split("\n")
        text = "Converting text-to-image(from wordlist) : "
    else:  # paragraph
        re.sub("\s+", " ", data)
        word_list = data.split()
        text = "Converting text-to-image(from paragraph) : "

    font_list = glob.glob("assets/fonts/*.ttf")
    os.makedirs("./data/my_dataset/", exist_ok=True)

    for i, word in enumerate(track(word_list, description=text)):
        i = str(i).zfill(5)
        filename = f"./data/my_dataset/word_{i}.png"

        try:
            random_font = random.choice(font_list)
            font = ttLib.TTFont(random_font)
            font_name = font['name'].getDebugName(4)
            image = pyvips.Image.text(word, font=font_name, fontfile=random_font, dpi=300, align=pyvips.enums.Align.CENTRE)
            image.write_to_file(filename)
        except pyvips.error.Error:
            continue

        gt.write(f"{filename}\t{word}\n")

    gt.close()


if __name__ == "__main__":
    gen_image(file="./data/wordlist.txt")
    gen_image(file="./data/myan_data.txt", from_wordlist=False)

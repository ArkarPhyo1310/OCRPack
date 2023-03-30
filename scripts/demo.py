import argparse
import warnings

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms as T

from ocrpack.models.text_recog.architectures import PARSeq

warnings.filterwarnings(action="ignore")

def get_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description="OCR")
    arg_parser.add_argument(
        "-c", "--config", 
        type=str, default="./configs/parseq.yaml", 
        help="Path to the model config file"
    )
    arg_parser.add_argument(
        "-w", "--weight", 
        type=str, default="./weights/parseq.pt",
        help="Path to the model weight file"
    )
    arg_parser.add_argument(
        "-i", "--image",
        type=str, default="./assets/test.jpg",
        help="Path to the image file"
    )

    args = arg_parser.parse_args()
    return args

def main():
    opts = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(opts.config)
    
    # Model Initialization
    parseq = PARSeq(cfg.data, cfg.model, cfg.hypermeters)
    parseq.load_state_dict(torch.load(opts.weight))
    parseq.eval()
    parseq.to(device)

    # Image PreProcessing
    transforms = []
    transforms.extend([
        T.Resize(cfg.model.image_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    img_transform = T.Compose(transforms)

    # Image Transformation
    image = Image.open(opts.image).convert("RGB")
    image = img_transform(image).unsqueeze(0)

    # Inferencing
    logits = parseq(image.to(device))
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    word_score = float(confidence[0].cumprod(dim=0)[-1].detach().numpy())

    # Result Generation
    chars_print = "Characters: "
    score_print = "Score     : "
    
    for letter, score in zip(label[0], confidence[0]):
        chars_print += f"\t{letter}"
        score_print += f"\t{round(float(score.detach().numpy()), 2)}"

    total_print_len = (len(score_print) + (len(label[0]) * 4))

    print("=" * total_print_len)
    print(f"Decoded Label(Score): \t{label[0]}({round(word_score, 2)})")
    print("=" * total_print_len)
    print(chars_print)
    print(score_print)
    print("=" * total_print_len)


if __name__ == "__main__":
    main()
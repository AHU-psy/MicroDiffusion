import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from models.microdiffusion import MicroDiffusion
from train import load_config


def image_tensor(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def main():
    parser = argparse.ArgumentParser(description="Enhance one low-light image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="enhanced.png")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "default.yml"))
    args = parser.parse_args()

    config = load_config(args.config)
    model_args = SimpleNamespace(image_folder="outputs", resume="", mode="testing", seed=0)
    trainer = MicroDiffusion(model_args, config)
    trainer.load_ddm_ckpt(args.checkpoint, ema=True)
    trainer.model.eval()
    with torch.no_grad():
        prediction = trainer.model(image_tensor(args.input).to(config.device))["pred_x"][0]
    result = (prediction.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(target)
    print("Saved " + str(target))


if __name__ == "__main__":
    main()

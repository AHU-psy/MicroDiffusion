import argparse
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from datasets import PairDataset
from models.microdiffusion import MicroDiffusion


ROOT = Path(__file__).resolve().parent


def namespace(values):
    return SimpleNamespace(**{key: namespace(value) if isinstance(value, dict) else value
                              for key, value in values.items()})


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = namespace(yaml.safe_load(stream))
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config


def save_checkpoint(trainer, epoch, target=None):
    target = Path(target) if target else ROOT / "checkpoints" / "model_latest.pth.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"epoch": epoch, "epoch_complete": True, "step": trainer.step,
                "config": trainer.config, "params": trainer.args,
                "state_dict": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "ema_helper": trainer.ema_helper.state_dict()}, temporary)
    os.replace(temporary, target)


@torch.no_grad()
def write_validation(trainer, loader):
    folder = ROOT / "outputs" / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    for index, (images, names) in enumerate(loader):
        prediction = trainer.model(images[:, :3].to(trainer.device))["pred_x"][0]
        array = (prediction.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        target = folder / "{:08d}_{}_{}.png".format(trainer.step, index, Path(names[0]).stem)
        Image.fromarray(array).save(target)


def main():
    parser = argparse.ArgumentParser(description="Train MicroDiffusion")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yml"))
    parser.add_argument("--initialize-from", help="Compatible LightenDiffusion MicroDiffusionV4 checkpoint")
    parser.add_argument("--resume", help="MicroDiffusion checkpoint to resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=230)
    args = parser.parse_args()
    if args.initialize_from and args.resume:
        parser.error("--initialize-from and --resume cannot be used together")

    config = load_config(args.config)
    if args.epochs is not None:
        config.training.n_epochs = args.epochs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(config.data.root)
    if not root.is_absolute():
        root = ROOT / root
    train_set = PairDataset(root, config.data.train_manifest, training=True)
    validation_set = PairDataset(root, config.data.validation_manifest)
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True,
                              num_workers=config.data.num_workers, pin_memory=config.device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False,
                                   num_workers=config.data.num_workers, pin_memory=config.device.type == "cuda")
    model_args = SimpleNamespace(image_folder=str(ROOT / "outputs"), resume="", mode="training", seed=args.seed)
    trainer = MicroDiffusion(model_args, config)
    if args.resume:
        trainer.load_ddm_ckpt(args.resume, load_optimizer=True)
    elif args.initialize_from:
        trainer.initialize_from_v4(args.initialize_from)

    for epoch in range(trainer.start_epoch, config.training.n_epochs):
        trainer.model.train()
        for images, _ in train_loader:
            trainer.step += 1
            images = images.to(config.device, non_blocking=True)
            losses = trainer.noise_estimation_loss(trainer.model(images))
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss at step {}".format(trainer.step))
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.optimizer.step()
            trainer.ema_helper.update(trainer.model)
        trainer.model.eval()
        with trainer.ema_helper.average_parameters(trainer.model):
            write_validation(trainer, validation_loader)
        save_checkpoint(trainer, epoch + 1)
        print("epoch={} step={} loss={:.6f}".format(epoch + 1, trainer.step, float(loss.detach())))


if __name__ == "__main__":
    main()

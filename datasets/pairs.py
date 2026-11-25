from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, root, manifest, training=False):
        self.root = Path(root)
        self.training = training
        manifest = Path(manifest)
        if not manifest.is_absolute():
            manifest = self.root / manifest
        if not manifest.is_file():
            raise FileNotFoundError("Pair manifest not found: " + str(manifest))
        self.pairs = []
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths = line.split("\t") if "\t" in line else line.split(maxsplit=1)
            if len(paths) != 2:
                raise ValueError("Expected low and normal image paths on line {}".format(line_number))
            paths = [path.strip() for path in paths]
            self.pairs.append(tuple(self._resolve(path) for path in paths))
        if not self.pairs:
            raise ValueError("Pair manifest contains no image pairs: " + str(manifest))

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _tensor(path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index):
        low_path, normal_path = self.pairs[index]
        low, normal = self._tensor(low_path), self._tensor(normal_path)
        if low.shape != normal.shape:
            raise ValueError("Pair images must have matching dimensions: {} and {}".format(low_path, normal_path))
        if self.training and torch.rand(()) < 0.5:
            low, normal = low.flip(-1), normal.flip(-1)
        return torch.cat((low, normal)), low_path.name

    def __len__(self):
        return len(self.pairs)

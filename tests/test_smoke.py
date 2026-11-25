from pathlib import Path
import tempfile
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train
import test as inference


class ModelSmokeTest(unittest.TestCase):
    def test_train_checkpoint_and_inference(self):
        root = ROOT
        config_path = root / "configs" / "default.yml"
        config = train.load_config(config_path)
        config.data.patch_size = 32
        config.model.illumination_size = 32
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            data = temporary / "data"
            (data / "low").mkdir(parents=True)
            (data / "normal").mkdir()
            low_path, normal_path = data / "low" / "scene.png", data / "normal" / "scene.png"
            low = (torch.rand(32, 32, 3).numpy() * 255).round().astype("uint8")
            normal = (low.astype("float32") * 1.4).clip(0, 255).astype("uint8")
            from PIL import Image
            Image.fromarray(low).save(low_path)
            Image.fromarray(normal).save(normal_path)
            line = "low/scene.png\tnormal/scene.png\n"
            (data / "train_pairs.txt").write_text(line, encoding="utf-8")
            (data / "validation_pairs.txt").write_text(line, encoding="utf-8")

            config.data.root = str(data)
            config.training.n_epochs = 1
            config.training.batch_size = 1
            temporary_config = temporary / "config.yml"
            import yaml
            temporary_config.write_text(yaml.safe_dump({
                "data": {"root": str(data), "train_manifest": "train_pairs.txt",
                        "validation_manifest": "validation_pairs.txt", "patch_size": 32, "num_workers": 0},
                "model": vars(config.model), "prior": vars(config.prior), "noise": vars(config.noise),
                "degradation": vars(config.degradation), "diffusion": vars(config.diffusion),
                "loss": vars(config.loss), "training": vars(config.training),
                "optim": vars(config.optim),
                "ablation": vars(config.ablation)}, sort_keys=False), encoding="utf-8")

            previous_root, previous_argv = train.ROOT, __import__("sys").argv
            try:
                train.ROOT = temporary
                __import__("sys").argv = ["train.py", "--config", str(temporary_config)]
                train.main()
                checkpoint = temporary / "checkpoints" / "model_latest.pth.tar"
                output_path = temporary / "enhanced.png"
                __import__("sys").argv = ["test.py", "--config", str(temporary_config),
                                         "--checkpoint", str(checkpoint), "--input", str(low_path),
                                         "--output", str(output_path)]
                inference.main()
                self.assertTrue(output_path.is_file())
                self.assertGreater(output_path.stat().st_size, 0)
            finally:
                train.ROOT, __import__("sys").argv = previous_root, previous_argv


if __name__ == "__main__":
    unittest.main()

"""Load the official LightenDiffusion U-Net without copying upstream code."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "LightenDiffusion-main" / "models" / "unet.py"
if not _path.is_file():
    raise FileNotFoundError("Place the official LightenDiffusion-main beside MicroDiffusion; see README.md")
_spec = importlib.util.spec_from_file_location("_lighten_unet", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

DiffusionUNet = _module.DiffusionUNet
get_timestep_embedding = _module.get_timestep_embedding

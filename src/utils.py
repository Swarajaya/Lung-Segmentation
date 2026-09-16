"""General utilities: reproducibility, device selection, small helpers."""
import os
import random
import json
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic where practical. Full determinism on all ops can hurt
    # performance / is not always available on CPU-only builds, so we do not
    # force torch.use_deterministic_algorithms(True), but we do disable the
    # cudnn benchmark heuristic which introduces non-determinism on GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device, honoring an explicit preference."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_hardware() -> dict:
    """Return a small dict describing the hardware currently available."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

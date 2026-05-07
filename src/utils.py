import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
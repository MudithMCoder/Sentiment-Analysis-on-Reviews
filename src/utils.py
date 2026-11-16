import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG in play and make torch ops deterministic where it can."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by cuBLAS determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

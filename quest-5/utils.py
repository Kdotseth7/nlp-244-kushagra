import torch
import random
import numpy as np
import os


def get_device() -> torch.device:
    """Returns the device to be used for model training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For new Mac M1 or M2 chips
    else:
        device = torch.device("cpu")
    return device


def make_reproducible(seed: int=42) -> None:
    """Set seed to make the training reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    print(f"Device: {get_device()}")
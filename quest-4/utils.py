import torch


def get_device():
    """Returns the device to be used for model training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For new Mac M1 or M2 chips
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    print(f"Device: {get_device()}")
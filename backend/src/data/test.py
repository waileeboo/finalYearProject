import torch
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
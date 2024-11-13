import torch

# Generate random data (e.g., 1000 samples, 100-dimensional)
activation_data = torch.rand(1000, 100)
torch.save(activation_data, "data/sample_data.pt")

import torch
import os

def load_data(filepath="data/sample_data.pt"):
    """Loads data from the specified path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at path: {filepath}")
    return torch.load(filepath)

def extract_activations(model, data):
    """
    Passes data through the multitask model and captures activations from the shared layer
    for each task.
    """
    model.eval()
    with torch.no_grad():
        shared_representation = model.shared_layer(data)
    # Return the shared 50-dimensional representation for both tasks
    return shared_representation, shared_representation

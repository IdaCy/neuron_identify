import torch
from torch.optim import Adam
from multitask_model import MultitaskModel
from autoencoder import SparseAutoencoder, sparse_loss
from utils import load_data, extract_activations

def train_autoencoder(task_activations, input_dim=50, hidden_dim=20, epochs=50, l1_lambda=1e-3):
    model = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = Adam(model.parameters(), lr=0.001)
    task_activations = task_activations.to(torch.float32)

    #print("Task A Activations Shape:", task_a_activations.shape)
    #print("Task B Activations Shape:", task_b_activations.shape)

    for epoch in range(epochs):
        model.train()
        encoded, reconstructed = model(task_activations)
        loss = sparse_loss(reconstructed, task_activations, encoded, l1_lambda)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

if __name__ == "__main__":
    data = load_data()
    multitask_model = MultitaskModel()

    # Extract task-specific activations
    task_a_activations, task_b_activations = extract_activations(multitask_model, data)

    # Train a sparse autoencoder for each task
    task_a_autoencoder = train_autoencoder(task_a_activations)
    task_b_autoencoder = train_autoencoder(task_b_activations)

    # Save the trained models
    torch.save(task_a_autoencoder, "models/task_a_autoencoder.pth")
    torch.save(task_b_autoencoder, "models/task_b_autoencoder.pth")
    torch.save(task_a_activations, "data/task_a_activations.pt")
    torch.save(task_b_activations, "data/task_b_activations.pt")

# src/multitask_model.py
import torch
import torch.nn as nn

class MultitaskModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=50):
        super(MultitaskModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Task-specific output heads
        self.task_a_head = nn.Linear(hidden_dim, 1)  # Binary classification
        self.task_b_head = nn.Linear(hidden_dim, 1)  # Regression

    def forward(self, x):
        shared_representation = self.shared_layer(x)
        task_a_output = torch.sigmoid(self.task_a_head(shared_representation))
        task_b_output = self.task_b_head(shared_representation)
        return task_a_output, task_b_output

import torch
import numpy as np
from torch import nn
from .input_encoder import InputEncoder
from .vqc_layer import VQCLayer

class HybridQNN(nn.Module):
    """
    Hybrid quantum-classical neural network.

    Args:
        num_features (int): Number of classical input features.
        num_classes (int): Number of output classes.
        n_vqc_layers (int): Depth of the variational quantum circuit.
        embedding_type (str): 'angle' or 'amplitude'.
        gate_type (str): Rotation gate type ('X', 'Y', or 'Z').

    Returns:
        torch.Tensor: Logits of shape (batch_size, num_classes).
    """

    def __init__(self, num_features, num_classes,
                 n_vqc_layers=2, embedding_type="angle", gate_type="Y"):
        super().__init__()

        self.embedding_type = embedding_type
        self.gate_type = gate_type

        # Determine output size from quantum layer
        if embedding_type == "amplitude":
            q_output_dim = int(np.log2(VQCLayer._next_power_of_two_int(num_features)))
        else:
            q_output_dim = num_features

        # Quantum layer
        self.quantum_layer = VQCLayer(
            encoder=InputEncoder(),
            n_layers=n_vqc_layers
        )

        # Classical output layer
        self.linear_head = nn.Linear(q_output_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features)

        Returns:
            torch.Tensor: Logits after classical linear layer
        """
        q_out = self.quantum_layer(x,
                                   embedding_type=self.embedding_type,
                                   gate_type=self.gate_type)
        return self.linear_head(q_out)

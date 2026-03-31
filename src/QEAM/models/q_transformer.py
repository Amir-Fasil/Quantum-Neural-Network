import torch
import torch.nn as nn
from ..layers.transformer_block import QuantumTransformerBlock

class QTransformer(nn.Module):
    """
    Full Quantum Transformer model for sequence classification.

    Args:
        vocab_size (int): Size of the token vocabulary.
        embed_dim (int): Embedding dimension for tokens.
        n_qubits (int): Number of qubits in quantum attention.
        n_layers (int): Number of layers in the quantum ansatz.
        device (qml.Device): PennyLane quantum device.
        num_blocks (int): Number of QuantumTransformerBlocks to stack.
        num_classes (int): Number of output classes for classification.
        ff_hidden_dim (int): Hidden dimension in feed-forward layers.
        max_seq_len (int): Maximum sequence length (for positional encoding & interference term).
        dropout (float): Dropout probability.
    """

    def __init__(
            self, 
            vocab_size, 
            embed_dim, 
            n_qubits, 
            n_layers, 
            device, 
            q_device,
            num_blocks=2, 
            num_classes=2, 
            ff_hidden_dim=128, 
            max_seq_len=128, 
            dropout=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.q_device = q_device

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.01)

        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                embed_dim=embed_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                device=self.q_device,  
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the QTransformer.

        Args:
            x (torch.Tensor): Input token IDs, shape (batch_size, seq_len)

        Returns:
            logits (torch.Tensor): Output logits, shape (batch_size, num_classes)
        """

        batch_size, seq_len = x.shape
        x = self.token_embed(x) + self.pos_embed[:, :seq_len, :]
        attn_weights_all = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_weights_all.append(attn_weights)
        x_pooled = x.mean(dim=1)
        logits = self.classifier(x_pooled)
        return logits, attn_weights_all
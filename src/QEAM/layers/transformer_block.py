import torch
import torch.nn as nn
from ..attention.quantum_self_attention import QuantumSelfAttention

class QuantumTransformerBlock(nn.Module):
    """
    Single Transformer block using Quantum Self-Attention (QSA).

    Args:
        embed_dim (int): Dimension of input token embeddings.
        n_qubits (int): Number of qubits for quantum self-attention.
        n_layers (int): Number of layers in the quantum ansatz.
        device (qml.Device): PennyLane device for quantum circuits.
        ff_hidden_dim (int): Hidden dimension of feed-forward network.
        dropout (float): Dropout probability for attention and feed-forward layers.
        max_seq_len (int): Maximum sequence length (used for interference parameter shape).
     """


    def __init__(
            self, 
            embed_dim, 
            n_qubits, 
            n_layers, 
            device,
            ff_hidden_dim=128, 
            dropout=0.1, 
            max_seq_len=128
    ):
        super().__init__()
        self.attention = QuantumSelfAttention(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device,  
            embed_dim=embed_dim,
            max_seq_len=max_seq_len
        )
        
        # LayerNorm for residual connections
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            Forward pass through the Quantum Transformer Block.

            Args:
            x (torch.Tensor): Input embeddings (batch_size, seq_len, embed_dim)

            Returns:
                out (torch.Tensor): Output embeddings after QSA + feed-forward, same shape as input
                attn_weights (torch.Tensor): Quantum attention weights (batch_size, seq_len, seq_len)
        """
        # Quantum Self-Attention + residual + layer norm
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + self.attn_dropout(attn_out))

        ff_out = self.ff(x)
        out = self.norm2(x + ff_out)
        
        return out, attn_weights
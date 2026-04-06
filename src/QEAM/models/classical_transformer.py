import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """
    Standard single-head self-attention module.
    
    Args:
        embed_dim (int): Dimension of input embeddings.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of self-attention.

        Args:
            x (torch.Tensor): Input embeddings (batch_size, seq_len, embed_dim)
        
        Returns:
            out (torch.Tensor): Output embeddings (batch_size, seq_len, embed_dim)
            attn_weights (torch.Tensor): Attention weights (batch_size, seq_len, seq_len)
        """
        Q = self.q_proj(x)  
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = self.out_proj(out)

        return out, attn_weights


class ClassicalTransformerBlock(nn.Module):
    """
    A single transformer block with classical self-attention + feed-forward.
    
    Args:
        embed_dim (int): Embedding dimension
        ff_hidden_dim (int): Hidden dimension in feed-forward network
        dropout (float): Dropout rate
    """
    def __init__(self, embed_dim: int, ff_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attn = ClassicalSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input embeddings (batch_size, seq_len, embed_dim)
        
        Returns:
            out (torch.Tensor): Output embeddings
            attn_weights (torch.Tensor): Attention matrix
        """
        # Self-Attention with residual
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class ClassicalTransformer(nn.Module):
    """
    Full classical transformer for sequence classification.

    Args:
        vocab_size (int): Vocabulary size
        embed_dim (int): Embedding dimension
        num_blocks (int): Number of transformer blocks to stack
        num_classes (int): Number of output classes
        ff_hidden_dim (int): Hidden dimension in feed-forward
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int = 2,
        num_classes: int = 2,
        ff_hidden_dim: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.01)

        # Stack of classical transformer blocks
        self.blocks = nn.ModuleList([
            ClassicalTransformerBlock(embed_dim, ff_hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Pooling (mean over tokens)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input token IDs (batch_size, seq_len)
        
        Returns:
            logits (torch.Tensor): Output logits (batch_size, num_classes)
            attn_weights_all (List[Tensor]): Attention matrices from each block
        """
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # Embed tokens + positional embedding
        x = self.token_embed(x) + self.pos_embed[:, :seq_len, :]

        attn_weights_all = []

        # Pass through transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_weights_all.append(attn_weights)

        # Pooling (mean over sequence)
        x_pooled = x.mean(dim=1)

        # Classification
        logits = self.classifier(x_pooled)

        return logits, attn_weights_all
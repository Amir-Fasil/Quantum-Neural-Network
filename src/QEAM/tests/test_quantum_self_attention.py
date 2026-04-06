import torch
import pennylane as qml
from ..attention.quantum_self_attention import QuantumSelfAttention

def test_quantum_self_attention():
    """
    End-to-end test for QuantumSelfAttention.
    """

    torch.manual_seed(42)
    seq_len = 3
    embedding_dim = 4
    n_qubits = embedding_dim  
    n_layers = 2
    max_seq_len = 5

    # Define quantum device
    device = qml.device('default.qubit', wires=n_qubits)

    # Initialize Quantum Self-Attention block
    qsa = QuantumSelfAttention(
        embed_dim=embedding_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=device,
        max_seq_len=max_seq_len
    )

    # Test multiple batch sizes
    for batch_size in [1, 2, 5]:
        
        x = torch.randn(batch_size, seq_len, embedding_dim)

        output, attn_weights = qsa(x)

      
        print(f"Batch size {batch_size} | Output shape: {output.shape} | Attention shape: {attn_weights.shape}")

      
        assert output.shape == (batch_size, seq_len, embedding_dim), \
            f"Output shape mismatch: {output.shape}"
        assert attn_weights.shape == (batch_size, seq_len, seq_len), \
            f"Attention weights shape mismatch: {attn_weights.shape}"

     
        assert torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1), \
            "Attention weights should be probabilities (0 <= attn <= 1)"
        assert torch.all(output.abs() < 1e3), "Output values out of expected range"

    print("QuantumSelfAttention test passed for all batch sizes!")

if __name__ == "__main__":
    test_quantum_self_attention()
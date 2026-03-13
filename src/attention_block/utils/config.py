import torch
import pennylane as qml

class Config:
    """
    General configuration for training and experiments.
    """
    # Training hyperparameters
    
    batch_size = 4
    learning_rate = 3e-5
    weight_decay = 0.01
    epochs = 5
    dropout = 0.1
    subset=200

    # Model parameters
    embed_dim = 64          
    vocab_size = 30522      

  
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Quantum-specific parameters
  
    n_qubits = 4             
    n_layers = 1             
    
    q_device = qml.device("default.qubit", wires=n_qubits)

    
    # Transformer parameters
    max_seq_len = 64       

 
    # Dataset
    dataset_name = "AGNEWS"  

 
    # Random seed for reproducibility
    seed = 42


def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across random, numpy, torch.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
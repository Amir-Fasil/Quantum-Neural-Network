import pennylane as qml
import torch
from .feature_map import AngleEmbeddingFeatureMap
from .ansatz import StronglyEntanglingAnsatz


class QuantumKernel:
    """
    Args:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Number of ansatz layers.
        device (qml.Device): PennyLane quantum device (simulator or hardware).
    """

    def __init__(self, n_qubits: int, n_layers: int, device):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = device

      
        self.feature_map = AngleEmbeddingFeatureMap(n_qubits)
       
        self.ansatz = StronglyEntanglingAnsatz(n_qubits, n_layers)

        @qml.qnode(self.dev, interface="torch",diff_method="parameter-shift")
        def expval_circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """

            Args:
                x (torch.Tensor): Input feature vector (length n_qubits)
                weights (torch.Tensor): Trainable parameters for ansatz (n_layers, n_qubits, 2)

            Returns:
                torch.Tensor: Expectation values <Z_i> for all qubits
            """
            
            self.feature_map(x)
            
            self.ansatz(x, weights)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.expval_circuit = expval_circuit

    def compute_kernel_matrix(self, X: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): Input embeddings of shape (batch_size, n_qubits)
            weights (torch.Tensor): Ansatz parameters of shape (n_layers, n_qubits, 2)

        Returns:
            torch.Tensor: Quantum attention matrix of shape (batch_size, batch_size)
                          where K[i, j] = <Z(x_i), Z(x_j)> / n_qubits
        """
        batch_size = X.shape[0]

       
        expvals = []
        for i in range(batch_size):
            expval = self.expval_circuit(X[i], weights)  
            expval = torch.stack(expval)  
            
            expvals.append(expval)

        expvals = torch.stack(expvals)  

        kernel_matrix = torch.matmul(expvals, expvals.T) / self.n_qubits

        return kernel_matrix
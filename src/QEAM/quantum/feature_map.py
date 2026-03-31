import pennylane as qml
import torch


class AngleEmbeddingFeatureMap:
    """
    Args:
        n_qubits (int):
            Number of qubits used in the quantum circuit.

        rotation (str, optional):
            Type of rotation gate used for encoding.
            Supported values:
                - "RY"
                - "RZ"
                - "RX"   
    """

    def __init__(self, n_qubits: int, rotation: str = "RY"):
        self.n_qubits = n_qubits
        self.rotation = rotation

    def __call__(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor):
                Input feature vector.

                Shape:
                    (n_qubits,)

        """

        for i in range(self.n_qubits):

            angle = x[i]

            if self.rotation == "RY":
                qml.RY(angle, wires=i)

            elif self.rotation == "RZ":
                qml.RZ(angle, wires=i)

            elif self.rotation == "RX":
                qml.RX(angle, wires=i)

            else:
                raise ValueError(
                    f"Unsupported rotation type: {self.rotation}"
                )
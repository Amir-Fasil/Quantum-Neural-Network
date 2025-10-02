import torch
from torch import nn
import pennylane as qml
import numpy as np
from input_encoder import InputEncoder
from qiskit import QuantumCircuit

class VQCLayer(nn.Module):
    def __init__(
        self, 
        num_qubits: int,
        encoder: InputEncoder | None = None,
        ansatz_fn=None,
        n_layers: int = 1, 
        device_type: str = "default.qubit",
        measurement_fn=None,
        q_circuit: QuantumCircuit | None = None, 
    ):
        """
        Variational Quantum Circuit (VQC) Layer.
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.encoder = encoder
        self.n_layers = n_layers
        self.q_circuit = q_circuit

        # Initialize Pennylane device
        self.dev = qml.device(device_type, wires=self.num_qubits)

        # Default ansatz if not provided
        if ansatz_fn is None:
            self.ansatz_fn = lambda weights: qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            self.weight_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=num_qubits)
        else:
            self.ansatz_fn = ansatz_fn
            self.weight_shape = getattr(ansatz_fn, "weight_shape", (n_layers, num_qubits))

        # Trainable weights
        self.weights = nn.Parameter(torch.randn(*self.weight_shape) * np.pi)

        # Default measurement: Z expectation on all qubits
        if measurement_fn is None:
            self.measurement_fn = lambda: [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        else:
            self.measurement_fn = measurement_fn

        # Define QNode (custom circuit if provided)
        if self.q_circuit is not None:
            self.qnode = qml.QNode(self.q_circuit, self.dev, interface="torch")
        else:
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, weights, inputs=None):
        """Quantum circuit for the VQC layer."""
        # Encode classical input if encoder provided
        if self.encoder is not None and inputs is not None:
            self.encoder.classic_input = inputs.detach().cpu().numpy()
            embedding_fn = self.encoder.embedding_function(embedding_type="angle", gate_type="Y")
            embedding_fn()

        # Variational ansatz
        self.ansatz_fn(weights)

        # Measurement
        return self.measurement_fn()

    def forward(self, input_):
        """Forward pass with batch support."""
        input_ = input_.float() if isinstance(input_, torch.Tensor) else input_

        if len(input_.shape) == 1:  # single input
            return self.qnode(self.weights, inputs=input_)
        else:  # batch
            batch_out = [self.qnode(self.weights, inputs=x_i) for x_i in input_]
            return torch.stack(batch_out)
        
    def backward(self, grad_output):
        """
        Optional: define backward pass if using custom gradients.
        For standard PyTorch + Pennylane, autograd handles gradients automatically.
        """
        pass


if __name__ == '__main__':
    x = torch.tensor([0.1, 0.5, 0.9])
    encoder = InputEncoder(classic_input=x.numpy())

    # Default VQC
    vqc = VQCLayer(num_qubits=3, n_layers=2, encoder=encoder)
    print("Default VQC output:", vqc(x))

    # Custom ansatz
    ansatz = lambda weights: qml.StronglyEntanglingLayers(weights, wires=range(3))
    ansatz.weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3)
    vqc_custom = VQCLayer(num_qubits=3, encoder=encoder, ansatz_fn=ansatz)
    print("Custom ansatz output:", vqc_custom(x))

    # Custom measurement
    measure_qubit0 = lambda: qml.expval(qml.PauliZ(0))
    vqc_measure = VQCLayer(num_qubits=3, encoder=encoder, measurement_fn=measure_qubit0)
    print("Custom measurement output:", vqc_measure(x))

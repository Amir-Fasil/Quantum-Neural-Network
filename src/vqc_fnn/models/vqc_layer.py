from torch import nn
import torch
import pennylane as qml
from input_encoder import InputEncoder
from qiskit import QuantumCircuit
import numpy as np

class VQCLayer(nn.Module):
    def __init__(self, num_qubits, q_circuit: QuantumCircuit | None = None, n_layers: int = 1, encoder: InputEncoder | None = None):
        """
        Variational Quantum Circuit (VQC) Layer.

        Args:
            num_qubits (int): Number of qubits in the VQC.
            q_circuit (QuantumCircuit | None): Optional custom Qiskit or Pennylane circuit.
            n_layers (int): Number of entangler layers if using default Pennylane VQC.
            encoder (InputEncoder | None): Optional InputEncoder for classical-to-quantum encoding.
        """
        super(VQCLayer, self).__init__()
        self.num_qubits = num_qubits
        self.encoder = encoder
        self.n_layers = n_layers

        # Initialize Pennylane device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # Trainable parameters for the entangler layers
        self.weight_shape = qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=self.num_qubits)
        self.weights = nn.Parameter(torch.randn(*self.weight_shape) * np.pi)

        # Define QNode using the provided circuit or default
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, weights, inputs=None):
        """
        Quantum circuit for the VQC layer.

        Args:
            weights: Trainable parameters for entangler layers.
            inputs: Classical input tensor for encoding.
        """
        # Encode classical input if encoder provided
        if self.encoder is not None and inputs is not None:
            # Convert torch.Tensor to numpy
            self.encoder.classic_input = inputs.detach().cpu().numpy()
            # Generate the circuit
            self.encoder.encode_input(
                gate_type='Y',
                embedding_type='angle',
                operation_type='initialization',
                with_padding=False,
                device=self.dev,
                interface="torch",
                operations_list=None,
                return_state=False
            )()

        # Apply variational entangler layers
        qml.BasicEntanglerLayers(weights, wires=range(self.num_qubits))

        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, input_):
        """
        Forward pass through the VQC layer.

        Args:
            input_ (torch.Tensor): Input tensor of shape (batch_size, num_qubits) or (num_qubits,)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_qubits)
        """
        if isinstance(input_, torch.Tensor):
            input_ = input_.float()

        # Single input
        if len(input_.shape) == 1:
            return self.qnode(self.weights, inputs=input_)
        # Batch input
        else:
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
    vqc = VQCLayer(num_qubits=3, n_layers=2, encoder=encoder)

    output = vqc(x)
    print("VQC output:", output)
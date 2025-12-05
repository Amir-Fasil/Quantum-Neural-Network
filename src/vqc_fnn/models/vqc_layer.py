import torch
from torch import nn
import pennylane as qml
import numpy as np
from typing import Optional, Callable, Dict, Tuple, Any, Union, List


from input_encoder import InputEncoder 

class VQCLayer(nn.Module):
    """
    Variational Quantum Circuit (VQC) PyTorch layer that integrates the InputEncoder.
    
    This layer is designed for efficiency:
    - It caches QNodes and PyTorch weights based on the required number of qubits (n_qubits).
    - It uses pure PyTorch tensors for the forward pass, maintaining autograd compatibility.
    """

    def __init__(
        self,
        encoder: Optional[InputEncoder] = None,
        n_layers: int = 1,
        ansatz_fn: Optional[Callable[[Union[torch.Tensor, np.ndarray], List[int]], None]] = None,
        weight_shape_fn: Optional[Callable[[int], Tuple[int, ...]]] = None,
        measurement_fn: Optional[Callable[[int], list]] = None,
        device_type: str = "default.qubit",
    ):
        super().__init__()
       
        self.encoder = encoder or InputEncoder(device_type=device_type)
        self.n_layers = n_layers
        self.device_type = device_type
        self.ansatz_fn = ansatz_fn
        self.weight_shape_fn = weight_shape_fn

        
        if measurement_fn is None:
            self.measurement_fn = lambda n_qubits: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            self.measurement_fn = measurement_fn

        self.qnode_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        self.weights_dict: Dict[int, nn.Parameter] = {}


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

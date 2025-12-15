import torch
from torch import nn
import pennylane as qml
import numpy as np
from typing import Optional, Callable, Dict, Tuple, Any, Union, List


from .input_encoder import InputEncoder 

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

    def _default_weight_shape(self, n_qubits: int) -> Tuple[int, ...]:
        """Provides default shape for BasicEntanglerLayers."""
        return qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)

    @staticmethod
    def _next_power_of_two_int(n: int) -> int:
        """Utility: compute next power of two (pure python)."""
        if n <= 0: return 1
        if (n & (n - 1)) == 0: return n
        return 1 << (n.bit_length())
    
    def _ensure_qnode_and_weights(self, n_qubits: int, embedding_type: str, gate_type: str):
        key = (n_qubits, embedding_type, gate_type)

        if n_qubits not in self.weights_dict:
            shape = tuple(self.weight_shape_fn(n_qubits)) if self.weight_shape_fn else self._default_weight_shape(n_qubits)
            param = nn.Parameter(torch.randn(*shape) * np.pi)
            self.register_parameter(f"weights_{n_qubits}", param)
            self.weights_dict[n_qubits] = param

        if key not in self.qnode_cache:
            dev = qml.device(self.device_type, wires=n_qubits)
            wires = list(range(n_qubits))

            embedding_type_local = embedding_type
            gate_type_local = gate_type
            ansatz_local = self.ansatz_fn
            measurement_fn_local = self.measurement_fn

           
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(weights, x_data):
                """
                weights: torch tensor shaped per ansatz
                x_data: 1D torch tensor containing the classical input sample (potentially padded)
                """
                
    
                if embedding_type_local == "angle":
                    
                    qml.AngleEmbedding(x_data, wires=wires, rotation=gate_type_local)

                elif embedding_type_local == "amplitude":
                   
                    qml.AmplitudeEmbedding(x_data, wires=wires, normalize=True, pad_with=0.0)

                else:
                    raise ValueError(f"Unsupported embedding_type: {embedding_type_local}")

            
                if ansatz_local is None:
                    qml.BasicEntanglerLayers(weights, wires=wires)
                else:
                    ansatz_local(weights, wires)


                measurements = measurement_fn_local(n_qubits)
                return measurements

            self.qnode_cache[key] = {"device": dev, "qnode": circuit, "wires": wires}

        return self.qnode_cache[key], self.weights_dict[n_qubits]



    def forward(self, x: torch.Tensor, embedding_type: str = "angle", gate_type: str = "Y") -> torch.Tensor:
        """
        x: tensor (batch_size, n_features) or (n_features,)
        Returns: tensor (batch_size, n_measurements)
        """
        
        
        x = x.to(dtype=torch.get_default_dtype())
        single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True

        batch_out = []
        batch_size = x.shape[0]

        for i in range(batch_size):
            sample = x[i]  
            length = sample.numel()

           
            if embedding_type == "amplitude":
                next_pow2 = self._next_power_of_two_int(length)
                
            
                num_wires = int(np.log2(next_pow2)) if next_pow2 > 1 else 1
                
              
                if next_pow2 != length:
                    pad = torch.zeros(next_pow2 - length, dtype=sample.dtype, device=sample.device)
                    sample_arg = torch.cat([sample, pad], dim=0)
                else:
                    sample_arg = sample
                
                n_qubits = num_wires
            else:
               
                n_qubits = int(length)
                sample_arg = sample

          
            qnode_meta, weights_param = self._ensure_qnode_and_weights(n_qubits, embedding_type, gate_type)
            qnode = qnode_meta["qnode"]

            
            out = qnode(weights_param, sample_arg) 

            
            if torch.is_tensor(out):
                out_t = out.view(-1)
            elif isinstance(out, (list, tuple)):
                out_t = torch.stack([torch.as_tensor(o, dtype=sample.dtype, device=sample.device) for o in out]).view(-1)
            else:
                out_t = torch.as_tensor(np.array(out), dtype=sample.dtype, device=sample.device).view(-1)

            batch_out.append(out_t)

        batch_out = torch.stack(batch_out, dim=0)
        if single:
            return batch_out.squeeze(0)
        return batch_out

        
    def backward(self, grad_output):
        """
        Optional: define backward pass if using custom gradients.
        For standard PyTorch + Pennylane, autograd handles gradients automatically.
        """
        pass


    

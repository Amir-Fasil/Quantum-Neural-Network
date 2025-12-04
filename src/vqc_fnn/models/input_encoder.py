import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Union, Tuple

ArrayLike = Union[np.ndarray, List[float]]
class InputEncoder:
    """
    Stateless, modular quantum input encoder for feedforward VQC networks.

    Supports:
    - AngleEmbedding (rotation gates)
    - AmplitudeEmbedding (normalized, padded)
    
    IMPORTANT:
    - QNode is defined once and accepts data as an argument for efficient batching.
    """
    def __init__(self, device_type: str = "default.qubit"):

        self.device_type = device_type
        self.random_generator = np.random.default_rng()


    @staticmethod
    def add_padding(vector: np.ndarray) -> Tuple[np.ndarray, int]:
        """Pads a vector to the next power of 2 for amplitude embedding."""
        vector = np.asarray(vector, dtype=np.float64).flatten()
        length = len(vector)
  
        next_pow2 = 2 ** int(np.ceil(np.log2(max(length, 1))))
        padded_vector = np.pad(vector, (0, next_pow2 - length), 'constant')
        return padded_vector, next_pow2

    def prepare_state(self, embedding_type: str):
        """
        Apply Hadamard gates to all qubits for superposition
        Only used for angle embedding.
        """
        if embedding_type == "angle":
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)


    def embedding_template(
        self,
        embedding_type: str = "angle",
        gate_type: str = "Y"
    ) -> Callable[[ArrayLike], None]:
        """
        Returns a function that takes 'classic_input' and performs the embedding.
        This function defines the quantum operations within the QNode.
        """

        def func(classic_input: ArrayLike):
            classic_input_arr = np.asarray(classic_input, dtype=np.float64).flatten()
            
           
            if embedding_type == "angle":
                n_qubits = len(classic_input_arr)

                qml.AngleEmbedding(
                    classic_input_arr,
                    wires=range(n_qubits),
                    rotation=gate_type
                )
            
            
            elif embedding_type == "amplitude":
                
                padded_input, n_qubits = InputEncoder.add_padding(classic_input_arr)

                qml.AmplitudeEmbedding(
                    padded_input,
                    wires=range(n_qubits),
                    normalize=True,
                    pad_with=0.0
                )
            else:
                raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        return func
    
    
    @staticmethod
    def operations_function(operation_list: Optional[List[Union[qml.operation.Operation, Callable]]] = None) -> Callable[[], None]:
        """Returns a function that applies a list of pre-defined operations."""
        def func():
            if operation_list:
                for op in operation_list:
                    if callable(op):
                        op()
                    else:
                        qml.apply(op)
        return func

    def build_full_circuit(
            self,
            embedding_type:str = "angle",
            gate_type:str = "Y",
            operation_list=None,
            device:qml.device=None,
            return_state: bool=True,
            interface: str="autograd" 

    ):
        """
        Build a complete QNode that includes embedding + operation + measurement
        """
        self.device=device if device is not None else qml.device(self.device_type, wires = self.n_qubits)
        embedding= self.embedding_function(embedding_type,gate_type)
        ops=self.operations_function(operation_list)

        @qml.qnode(self.device, interface = interface, diff_method = "parameter-shift")
        def circuit():
            embedding()
            ops()
            return qml.state() if return_state else qml.expval(qml.PauliZ(0))
        return circuit
    
# Example usage

if __name__ == "__main__":
    input_data = np.asarray([0,0,1])
    encoder = InputEncoder(input_data)
    dev = qml.device("default.qubit", wires=3)
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0,1]),
        qml.CNOT(wires=[1,2])
    ]

    #1 embedding only(standalone QNode)
    embedding_fn=encoder.embedding_function("angle","Y")

    @qml.qnode(dev)
    def embedding_only():
        embedding_fn()
        return qml.state()
    print("Embedding only:", embedding_only())

    #2 operation only(standalone QNode)
    ops_fn=encoder.operations_function(ops)

    @qml.qnode(dev)
    def ops_only():
        ops_fn()
        return qml.state()
    print("Operation only:", ops_only())

    #3 full circuit(embedding + ops via build_full_circuit)
    full_circuit = encoder.build_full_circuit(
        embedding_type= "angle",
        gate_type= "Y",
        operation_list= ops,
        device= dev,
        return_state= False
    )
    print("Full circuit <Z0>:", full_circuit())

    #4 custom composition(embedding from encoder + new operation defined here)
    @qml.qnode(dev)
    def custom_circuit():
        embedding_fn()
        qml.RX(np.pi/4, wires=0)

        return qml.expval(qml.PauliZ(0))
    print("Custom composition <Z0>:", custom_circuit())



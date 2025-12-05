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

        initial_sample: ArrayLike, 
        embedding_type: str = "angle",
        gate_type: str = "Y",
        operation_list: Optional[List[Union[qml.operation.Operation, Callable]]] = None,
        device: Optional[qml.device] = None,
        return_state: bool = True,
        interface: str = "autograd",
        observable: Optional[qml.operation.Observable] = None
    ) -> Callable[[ArrayLike], ArrayLike]: 
        """
        Builds and returns a single QNode template that accepts data as an argument.
        This is defined ONCE.
        """
        
        initial_sample = np.asarray(initial_sample).flatten()
        if embedding_type == "angle":
            n_qubits = len(initial_sample)
        elif embedding_type == "amplitude":
            _, n_qubits = self.add_padding(initial_sample)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

     
        embedding_fn = self.embedding_template(embedding_type, gate_type)
        ops_fn = self.operations_function(operation_list)

        dev = device if device is not None else qml.device(self.device_type, wires=n_qubits)

        if observable is None:
           
            observable = qml.PauliZ(0) 

        
        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(classic_input): 
            embedding_fn(classic_input) 
            ops_fn()
            
            return qml.state() if return_state else qml.expval(observable)

        return circuit
    
    def encode_batch(
        self,
        batch: ArrayLike,
        embedding_type: str = "angle",
        gate_type: str = "Y",
        operation_list: Optional[List[Union[qml.operation.Operation, Callable]]] = None,
        device: Optional[qml.device] = None,
        return_state: bool = True,
        observable: Optional[qml.operation.Observable] = None
    ) -> np.ndarray:
        """
        Encode input samples one-by-one using a single, reusable QNode template.
        Returns numpy array of circuit outputs.
        """
        batch = np.asarray(batch)
        
        if batch.ndim < 2:
          
            batch = np.array([batch])

        
        single_qnode = self.build_full_circuit(
            initial_sample=batch[0], 
            embedding_type=embedding_type,
            gate_type=gate_type,
            operation_list=operation_list,
            device=device,
            return_state=return_state,
            observable=observable
        )

        outputs = []
        for x in batch:
           
            outputs.append(single_qnode(x)) 

        return np.array(outputs)



if __name__ == "__main__":
    
    
    input_data = np.array([
        [0.3, 0.5, 0.8], 
        [0.2, 0.7, 0.9], 
        [0.3, 0.4, 0.5]  
    ])

    encoder = InputEncoder()

    #
    ops = [
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.1, wires=2)
    ]

    print("--- Angle Embedding (default: Z0 expectation) ---")
    batch_expvals = encoder.encode_batch(
        input_data, 
        embedding_type="angle",
        operation_list=ops,
        return_state=False
    )
    print(f"Output shape (3 expvals): {batch_expvals.shape}")
    print(f"Batch Z0 expvals:\n{batch_expvals}\n")

   
    print("--- Amplitude Embedding (return state) ---")
    batch_states = encoder.encode_batch(
        input_data, 
        embedding_type="amplitude", 
        operation_list=ops, 
        return_state=True
    )
    print(f"Output shape: {batch_states.shape}")
    print(f"State of first sample:\n{batch_states[0]}")

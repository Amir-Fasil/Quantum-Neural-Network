import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


class InputEncoder:
    def __init__(self, classic_input: np.ndarray, device_type: str = 'default.qubit'):
        """
        Initialize encoder with classical input.
        Args:
            classic_input: Input array
            device_type: PennyLane device type
        """
        self.classic_input = np.asarray(classic_input, dtype=np.float64)
        self.n_qubits = len(self.classic_input)
        self.device_type = device_type
        self.device = None
        self.random_generator = np.random.default_rng()

    def add_padding(self):
        """
        Pad input to the next power of 2 and update n_qubits if needed
        """
        length = len(self.classic_input)
        next_power_of_2 = 2 ** np.ceil(np.log2(max(length, 1))).astype(int)
        if next_power_of_2 > self.n_qubits:
            self.n_qubits = next_power_of_2

        self.classic_input = np.pad(
            self.classic_input,
            (0, self.n_qubits - length),
            'constant'
        )

    def prepare_state(self, embedding_type: str):
        """
        Apply Hadamard gates to all qubits for superposition
        Only used for angle embedding.
        """
        if embedding_type == "angle":
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)


    #Circuit Modules
    def embedding_function(self,embedding_type:str = "angle",gate_type:str = "Y"):
        """
        Returns a function that applies only the embedding.
        Can be inserted into another QNode
        """
        def func():
            if embedding_type == "angle":
                self.prepare_state(embedding_type)
                qml.AngleEmbedding(
                    self.classic_input, 
                    wires=range(self.n_qubits), 
                    rotation=gate_type)
            elif embedding_type == "amplitude":
                self.add_padding()
                qml.AmplitudeEmbedding(
                    self.classic_input,
                    wires=range(self.n_qubits),
                    normalize=True,
                    pad_with=0.0
                )
            else:
                raise ValueError("embedding_type must be 'angle' or 'amplitude' ")

        return func
    def operations_function(self, operation_list=None):
        """
        Returns a function that applies only the given operations.
        Can be inserted into another QNode
        """
        def func():
            if operation_list:
                for op in operation_list:
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



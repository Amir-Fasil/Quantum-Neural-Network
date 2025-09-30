import pennylane as qml
import numpy as np

class InputEncoder:
    def __init__(self,classic_input:np.ndarray,device_type:str='default.qubit'):

        """
        Initialize encoder with classical input.
        Args:
            classic_input: Input array
            device_type: Pennylane device type
        """
        self.classic_input = np.asarray(classic_input,dtype=np.float64)
        self.n_qubits = len(self.classic_input)
        self.device_type = device_type
        self.device = None
        self.random_generator = np.random.default_rng()

    def add_padding(self):
        """
        Pad input to the next power of 2 and update n_qubits if needed
        """
        length = len(self.classic_input)
        next_power_of_2 = 2 ** np.ceil(np.log2(max(length,1))).astype(int)
        if next_power_of_2> self.n_qubits:
            self.n_qubits=next_power_of_2

        self.classic_input=np.pad(self.classic_input , (0,self.n_qubits-length ),'constant')

    def prepare_state(self):
        """
        Apply hadamrad gates to all qubits for superpositon
        """
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

    def encode_input(self, gate_type:str = 'Y', embedding_type:str = 'angle', operation_type:str = 'initialization', with_padding:bool = False , device:qml.device = None):
        """
        Encode input into quantum state

        Args:
            gate_type: Rotation gate for angle embedding 
            embedding_type: angle or amplitude
            operation_type: initialization (hadamard + embedding)
            with_padding:pad to power of 2
            device: optional pre-existing device to reuse

        Returns: 
            qml.QNode: qunatum circuit for encoding
        """
        if with_padding or embedding_type == 'amplitude':
            self.add_padding()

        else:
            self.n_qubits=len(self.classic_input)

        self.device=device if device is not None else qml.device(self.device_type, wires = self.n_qubits)

        @qml.qnode(self.device, interface = 'torch', diff_method = 'parameter-shift')
        def circuit():
            if operation_type == "initialization":
                if embedding_type == "angle":
                    self.prepare_state()
                    qml.AngleEmbedding(self.classic_input, wires=range(self.n_qubits), rotation=gate_type)

                elif embedding_type == "amplitude":
                    qml.AmplitudeEmbedding(self.classic_input, wires=range(self.n_qubits), normalize=True, pad_with=0.0)

            else:
                qml.AmplitudeEmbedding(self.classic_input, wires=range(self.n_qubits), normalize=True, pad_with=0.0)

            return  qml.state()
        return circuit
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test input
    input_data = np.asarray([0, 0, 1])
    encoder = InputEncoder(input_data)
    circuit = encoder.encode_input(with_padding=False)
    state = circuit()
    print("Input:", input_data)
    print("n_qubits:", encoder.n_qubits)
    print("State:", state)
    fig, ax = qml.draw_mpl(circuit)()
    plt.show()


        


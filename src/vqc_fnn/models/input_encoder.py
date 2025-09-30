import pennylane as qml
import numpy as np


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

    def encode_input(
        self,
        gate_type: str = 'Y',
        embedding_type: str = 'angle',
        operation_type: str = 'initialization',
        with_padding: bool = False,
        device: qml.device = None,
        interface: str = 'autograd',
        operations_list=None,
        return_state: bool = True
    ):
        """
        Encode input into quantum state and optionally apply operations.

        Args:
            gate_type: Rotation gate for angle embedding ('X','Y','Z')
            embedding_type: 'angle' or 'amplitude'
            operation_type: 'initialization' or 'operations'
            with_padding: pad to power of 2
            device: optional pre-existing device to reuse
            interface: ML interface for QNode ('autograd','torch','jax','tf')
            operations_list: list of quantum ops to apply after embedding
            return_state: if True returns statevector, else expval <Z0>

        Returns:
            qml.QNode: quantum circuit for encoding
        """
        # Validate rotation gate
        valid_rotations = ['X', 'Y', 'Z']
        if embedding_type == "angle" and gate_type not in valid_rotations:
            raise ValueError(f"gate_type must be one of {valid_rotations}")

        # Handle padding
        if embedding_type == "amplitude" or with_padding:
            self.add_padding()
        else:
            self.n_qubits = len(self.classic_input)

        # Device initialization
        self.device = device if device is not None else qml.device(
            self.device_type, wires=self.n_qubits
        )

        @qml.qnode(self.device, interface=interface, diff_method='parameter-shift')
        def circuit():
            # Initialization embedding
            if operation_type == "initialization":
                if embedding_type == "angle":
                    self.prepare_state(embedding_type)
                    qml.AngleEmbedding(
                        self.classic_input,
                        wires=range(self.n_qubits),
                        rotation=gate_type
                    )
                elif embedding_type == "amplitude":
                    qml.AmplitudeEmbedding(
                        self.classic_input,
                        wires=range(self.n_qubits),
                        normalize=True,
                        pad_with=0.0
                    )

            # Apply additional operations on top of embedding
            if operations_list:
                for op in operations_list:
                    qml.apply(op)

            # Measurement
            if return_state:
                return qml.state()
            else:
                return qml.expval(qml.PauliZ(0))

        return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Input and device
    input_data = np.asarray([0, 0, 1])
    encoder = InputEncoder(input_data)
    dev = qml.device("default.qubit", wires=3)

    # Case 1: Initialization (angle embedding)
    # circuit1 = encoder.encode_input(
    #     device=dev,
    #     embedding_type="angle",
    #     gate_type="Y",
    #     with_padding=False,
    #     return_state=True
    # )
    # state = circuit1()
    # print("Initialization State:", state)

    # Case 2: Initialization + operations
    ops = [
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.CNOT(wires=[1, 2])
    ]
    circuit2 = encoder.encode_input(
        device=dev,
        embedding_type="angle",
        gate_type="Y",
        with_padding=False,
        operations_list=ops,
        return_state=False
    )
    result = circuit2()
    print("Expectation value <Z0>:", result)

    # Draw circuit
    fig, ax = qml.draw_mpl(circuit2)()
    plt.show()

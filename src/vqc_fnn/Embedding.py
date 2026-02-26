import pennylane as qml
import math

class EmbeddingLayer:
    """
    Handles the encoding of classical data into quantum states.
    Supported methods: 'angle', 'amplitude'

    """

    def __init__(self, method="angle"):

        self.method = method.lower()
        if self.method not in ["angle", "amplitude"]:

            raise ValueError(f"Unsupported embedding method: {method}")


    def apply(self, features, wires):

        """Applies the chosen embedding to the quantum tape."""

        if self.method == "angle":

            # Encodes N features into N qubits using RX/RY/RZ rotations
            qml.AngleEmbedding(features=features, wires=wires, rotation='Y')
            
        elif self.method == "amplitude":
            
            # Encodes 2^N features into N qubits using the state vector
            # normalize=True ensures the data mathematically fits on the Bloch sphere
            qml.AmplitudeEmbedding(features=features, wires=wires, normalize=True)


    def get_required_qubits(self, num_features):

        """Helper method to calculate how many qubits you need for your data."""

        if self.method == "angle":
            return num_features
        elif self.method == "amplitude":
            return math.ceil(math.log2(num_features))
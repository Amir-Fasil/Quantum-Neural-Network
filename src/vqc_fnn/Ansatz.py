import pennylane as qml

class AnsatzLayer:

    """
    Handles the trainable quantum gates (the hidden layers).
    Supported methods: 'basic', 'strong'
    """

    def __init__(self, method="basic", n_layers=3):

        self.method = method.lower()
        self.n_layers = n_layers
        
        if self.method not in ["basic", "strong"]:
            raise ValueError(f"Unsupported ansatz method: {method}")

    def apply(self, weights, wires):

        """Applies the chosen parameterized layers to the quantum tape."""
        if self.method == "basic":
            qml.BasicEntanglerLayers(weights=weights, wires=wires)
            
        elif self.method == "strong":
            qml.StronglyEntanglingLayers(weights=weights, wires=wires)

    def get_weight_shape(self, n_qubits):

        """
        Dynamically calculates the exact tensor shape required for the weights 
        so you never get a matrix mismatch error during initialization.
        """
        if self.method == "basic":
            # Returns shape: (n_layers, n_qubits)
            return qml.BasicEntanglerLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)
            
        elif self.method == "strong":
            # Returns shape: (n_layers, n_qubits, 3)
            return qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)
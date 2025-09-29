from torch import nn
import pennylane as qml
from qiskit import QuantumCircuit
from input_encoder import InputEncoder
class VQCLayer(nn.Module):
    def __init__(self, num_qubits, q_circuit: QuantumCircuit | qml.QuantumCircuit):
        super(VQCLayer, self).__init__()
        
    def forward(self, input_):
       pass
    def backward(self, grad_output):
       pass
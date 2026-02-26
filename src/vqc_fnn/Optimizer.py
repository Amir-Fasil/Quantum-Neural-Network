import pennylane as qml
from pennylane import numpy as np

class Trainer:
    """
    Handles the classical optimization loop.
    """

    def __init__(self, model, optimizer_type="adam", stepsize=0.1):

        self.model = model
        
        # Set up the optimizer
        if optimizer_type.lower() == "adam":

            self.opt = qml.AdamOptimizer(stepsize=stepsize)
        else:

            self.opt = qml.GradientDescentOptimizer(stepsize=stepsize)


    def cost_function(self, weights, X, Y): 

        """Calculates the Mean Squared Error (MSE) over the batch."""

        predictions = np.array([self.model.forward(x, weights) for x in X])
        return np.mean((predictions - Y) ** 2)


    def fit(self, X, Y, epochs=20):

        """
        The function automatically initializes weights and runs the training loop.
        This is where the tranining happens.
        """

        print("Initializing weights...")
        # getting the shape weights.
        weight_shape = self.model.ansatz.get_weight_shape(self.model.n_qubits)
        # Initialize random weights 
        weights = np.random.random(weight_shape, requires_grad=True)
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # 3. Take an optimization step
            weights, _, _ = self.opt.step(self.cost_function, weights, X, Y)
            
            # 4. Print progress
            if (epoch + 1) % 5 == 0:
                current_cost = self.cost_function(weights, X, Y)
                print(f"Epoch {epoch + 1:3d} | Cost: {current_cost:.5f}")
                
        return weights
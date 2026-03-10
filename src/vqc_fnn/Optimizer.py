import pennylane as qml
from pennylane import numpy as np


_OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "gd": qml.GradientDescentOptimizer,
    "nesterov": qml.NesterovMomentumOptimizer,
    "spsa": qml.SPSAOptimizer,
}


class Trainer:
    """
    Handles the classical optimization loop for a VQCModel.

    Parameters
    ----------
    model : VQCModel
        Model exposing ``forward(x, weights)``, ``ansatz``, ``n_qubits``,
        ``train()``, and ``eval()``.
    optimizer_type : str
        One of 'adam', 'gd', 'nesterov', 'spsa'.
    stepsize : float
        Learning rate passed to the optimizer.
    batch_size : int or None
        If set, each epoch trains on random mini-batches of this size.
    """

    def __init__(self, model, optimizer_type="adam", stepsize=0.1, batch_size=None):
        self.model = model
        self.batch_size = batch_size

        key = optimizer_type.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{optimizer_type}'. "
                f"Choose from {list(_OPTIMIZERS)}"
            )
        self.opt = _OPTIMIZERS[key](stepsize=stepsize)

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    def cost_function(self, weights, X, Y):
        """Mean Squared Error over the given samples."""
        predictions = np.array([self.model.forward(x, weights) for x in X])
        return np.mean((predictions - Y) ** 2)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        Y,
        epochs=20,
        X_val=None,
        Y_val=None,
        patience=None,
        verbose_every=5,
    ):
        """
        Initialize weights and run the training loop.

        Parameters
        ----------
        X, Y : array-like
            Training features and labels.
        epochs : int
            Number of training epochs.
        X_val, Y_val : array-like or None
            Optional validation set for tracking generalization.
        patience : int or None
            If set (and validation data is provided), training stops after
            *patience* consecutive epochs with no improvement in validation
            cost (early stopping).
        verbose_every : int
            Print a progress line every *verbose_every* epochs (0 = silent).

        Returns
        -------
        dict
            ``weights``, ``train_history``, ``val_history``
        """
        weight_shape = self.model.ansatz.get_weight_shape(self.model.n_qubits)
        weights = np.random.random(weight_shape, requires_grad=True)

        train_history = []
        val_history = []
        best_val_cost = float("inf")
        stale_epochs = 0

        self.model.train()

        for epoch in range(epochs):
            if self.batch_size is not None and self.batch_size < len(X):
                indices = np.random.choice(len(X), self.batch_size, replace=False)
                X_batch, Y_batch = X[indices], Y[indices]
            else:
                X_batch, Y_batch = X, Y

            weights, _, _ = self.opt.step(self.cost_function, weights, X_batch, Y_batch)

            train_cost = self.cost_function(weights, X, Y)
            train_history.append(float(train_cost))

            val_cost = None
            if X_val is not None and Y_val is not None:
                self.model.eval()
                val_cost = float(self.cost_function(weights, X_val, Y_val))
                val_history.append(val_cost)
                self.model.train()

                if patience is not None:
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= patience:
                            if verbose_every:
                                print(
                                    f"Early stopping at epoch {epoch + 1} "
                                    f"(no improvement for {patience} epochs)"
                                )
                            break

            if verbose_every and (epoch + 1) % verbose_every == 0:
                msg = f"Epoch {epoch + 1:4d} | Train cost: {train_cost:.5f}"
                if val_cost is not None:
                    msg += f" | Val cost: {val_cost:.5f}"
                print(msg)

        self.model.eval()
        return {
            "weights": weights,
            "train_history": train_history,
            "val_history": val_history,
        }

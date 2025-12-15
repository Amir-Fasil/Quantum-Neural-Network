import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn, optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


from models.hybrid_qnn import HybridQNN


def get_iris_loaders(batch_size=16, val_ratio=0.2):
    """
    Load the Iris dataset and return PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        val_ratio (float): Fraction of dataset to use for validation.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train, validation, test sets
    val_size = int(len(dataset) * val_ratio)
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def evaluate_model(model, loader, device):
    """
    Evaluate the model on a given DataLoader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for evaluation.
        device (torch.device): Device to run the evaluation on.

    Returns:
        Tuple[float, float]: average loss, accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_hybrid_qnn(n_epochs=5, batch_size=16, learning_rate=0.01):
    """
    Train the HybridQNN on the Iris dataset.

    Args:
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Optimizer learning rate.

    Returns:
        Tuple[nn.Module, dict, int]: trained model, training history, number of epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_iris_loaders(batch_size=batch_size)
    num_features = train_loader.dataset[0][0].shape[0]
    num_classes = 3

    # Initialize model
    model = HybridQNN(num_features=num_features, num_classes=num_classes, n_vqc_layers=2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{n_epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return model, history, n_epochs


def plot_history(history, n_epochs):
    """
    Plot training and validation accuracy.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        n_epochs (int): Number of epochs.
    """
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, history, n_epochs = train_hybrid_qnn(n_epochs=5, batch_size=16, learning_rate=0.01)
    plot_history(history, n_epochs)

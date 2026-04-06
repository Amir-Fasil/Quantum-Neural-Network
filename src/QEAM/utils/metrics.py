from sklearn.metrics import accuracy_score as sk_accuracy
from sklearn.metrics import f1_score as sk_f1

def accuracy_score(y_true, y_pred):
    """
    Args:
        y_true (torch.Tensor or np.ndarray): True labels
        y_pred (torch.Tensor or np.ndarray): Predicted labels

    Returns:
        float: Accuracy
    """
    y_true = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else y_true
    y_pred = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else y_pred
    return sk_accuracy(y_true, y_pred)

def f1_score(y_true, y_pred, average='macro'):
    """
    Args:
        y_true (torch.Tensor or np.ndarray): True labels
        y_pred (torch.Tensor or np.ndarray): Predicted labels
        average (str): 'macro', 'micro', or 'weighted'

    Returns:
        float: F1 score
    """
    y_true = y_true.detach().cpu().numpy() if hasattr(y_true, "detach") else y_true
    y_pred = y_pred.detach().cpu().numpy() if hasattr(y_pred, "detach") else y_pred
    return sk_f1(y_true, y_pred, average=average)
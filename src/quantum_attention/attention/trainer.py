r"""
Training and evaluation scripts for the QSANN model.
"""

import logging
import os
from tqdm import tqdm
from typing import Optional, Tuple

import torch
from quantum_attention.attention.model import QSANN
from quantum_attention.attention.dataset import TextDataset, deal_vocab, build_iter
from quantum_attention.utils.logger import initialize_metrics_logger, log_metrics

def evaluate(model: torch.nn.Module, data_loader: list) -> Tuple[float, float]:
    r"""Evaluate the model.

    Args:
        model: The trained model to be evaluated.
        data_loader: The dataloader of the data used to evaluate the model.

    Returns:
        Return the average loss and accuracy in the data of the input dataloader.
    """
    dev_loss = 0.0
    model.eval()
    labels_all = []
    predicts_all = []
    with torch.no_grad():
        for texts, labels in data_loader:
            predictions = model(texts)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            preds = torch.stack(predictions).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            loss = torch.mean((preds - labels_tensor) ** 2)
            dev_loss += loss.item()

            labels_all.extend(labels)
            preds_binary = (preds < 0.5).int().tolist()
            if isinstance(preds_binary, int):
                preds_binary = [preds_binary]
            predicts_all.extend(preds_binary)

    dev_acc = sum(labels_all[idx] == predicts_all[idx] for idx in range(len(labels_all)))
    return (dev_loss / len(labels_all), dev_acc / len(labels_all)) if len(labels_all) != 0 else (0.0, 0.0)


def test(model: torch.nn.Module, model_path: str, test_loader: list) -> None:
    r"""Use the test dataset to test the model."""
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        test_loss, test_acc = evaluate(model, test_loader)
    msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
    logging.info(msg)
    print(msg)


def train(
        model_name: str, dataset: str, num_qubits: int, num_layers: int,
        depth_ebd: int, depth_query: int, depth_key: int, depth_value: int,
        batch_size: int, num_epochs: int, learning_rate: float = 0.01,
        saved_dir: str = '', using_validation: bool = False,
        early_stopping: int = 1000,
        metrics_log_path: Optional[str] = None,
) -> None:
    r"""
    The function of training the QSANN model.

    Args:
        model_name: The name of the model. It is the filename of the saved model.
        dataset: The dataset used to train the model, which should be a directory.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_layers: The number of the self-attention layers.
        depth_ebd: The depth of the embedding circuit.
        depth_query: The depth of the query circuit.
        depth_key: The depth of the key circuit.
        depth_value: The depth of the value circuit.
        batch_size: The size of the batch samplers.
        num_epochs: The number of the epochs to train the model.
        learning_rate: The learning rate used to update the parameters. Defaults to ``0.01`` .
        saved_dir: The directory to saved the trained model and the training log. Defaults to use the current path.
        using_validation: If the datasets contains the validation dataset.
            Defaults to ``False`` , which means the validation dataset is not included.
        early_stopping: Number of iterations with no improvement after which training will be stopped.
            Defaults to ``1000`` .
        metrics_log_path: Optional CSV file used to store structured metrics such as
            training loss, accuracy, and evaluation metrics. Defaults to ``None`` .
    """
    if not saved_dir:
        saved_dir = './'
    elif saved_dir[-1] != '/':
        saved_dir += '/'
    if dataset[-1] != '/':
        dataset += '/'
    os.makedirs(saved_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{saved_dir}{model_name}.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )
    metrics_log_path = initialize_metrics_logger(metrics_log_path)
    word2idx = deal_vocab(f'{dataset}vocab.txt')
    len_vocab = len(word2idx)
    train_dataset = TextDataset(file_path=f'{dataset}train.txt', word2idx=word2idx)
    if using_validation:
        dev_dataset = TextDataset(file_path=f'{dataset}validate.txt',  word2idx=word2idx)
    test_dataset = TextDataset(file_path=f'{dataset}test.txt', word2idx=word2idx)
    train_iter = build_iter(train_dataset, batch_size=batch_size, shuffle=True)
    if using_validation:
        dev_iter = build_iter(dev_dataset, batch_size=batch_size, shuffle=True)
    test_iter = build_iter(test_dataset, batch_size=batch_size, shuffle=True)
    model = QSANN(
        num_qubits=num_qubits, len_vocab=len_vocab, num_layers=num_layers,
        depth_ebd=depth_ebd, depth_query=depth_query, depth_key=depth_key, depth_value=depth_value,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    stopping_flag = False
    for epoch in range(num_epochs):
        p_bar = tqdm(
            total=len(train_iter),
            desc=f'Epoch[{epoch: 3d}]',
            ascii=True,
            dynamic_ncols=True,
        )
        for texts, labels in train_iter:
            p_bar.update(1)
            opt.zero_grad()
            predictions = model(texts)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            preds = torch.stack(predictions).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            loss = torch.mean((preds - labels_tensor) ** 2)
            loss.backward()
            opt.step()
            if total_batch % 10 == 0:
                train_predictions = (preds >= 0.5).int().tolist()
                if isinstance(train_predictions, int):
                    train_predictions = [train_predictions]
                train_acc = sum(labels[idx] == train_predictions[idx] for idx in range(len(labels))) / len(labels)
                log_metrics(
                    metrics_log_path,
                    epoch=epoch,
                    iteration=total_batch,
                    split='train',
                    loss=loss.item(),
                    accuracy=train_acc,
                    learning_rate=learning_rate,
                    num_samples=len(labels),
                    extra_metrics={"model": "quantum_attention"},
                )
                if using_validation:
                    with torch.no_grad():
                        dev_loss, dev_acc = evaluate(model, dev_iter)
                        is_best_model = dev_loss < dev_best_loss
                        if dev_loss < dev_best_loss:
                            torch.save(model.state_dict(), f'{saved_dir}{model_name}.pt')
                            improve = '*'
                            last_improve = total_batch
                            dev_best_loss = dev_loss
                        else:
                            improve = ' '
                        log_metrics(
                            metrics_log_path,
                            epoch=epoch,
                            iteration=total_batch,
                            split='validation',
                            loss=dev_loss,
                            accuracy=dev_acc,
                            learning_rate=learning_rate,
                            best_model=is_best_model,
                            num_samples=len(dev_dataset),
                            extra_metrics={"model": "quantum_attention"},
                        )
                    msg = (
                        f"Iter:{total_batch: 5d}, Train loss:{loss.item(): 3.5f}, acc:{train_acc: 3.2%}; "
                        f"Val loss:{dev_loss: 3.5f}, acc:{dev_acc: 3.2%}{improve}"
                    )
                else:
                    with torch.no_grad():
                        test_loss, test_acc = evaluate(model, test_iter)
                        torch.save(model.state_dict(), f'{saved_dir}{model_name}.pt')
                        log_metrics(
                            metrics_log_path,
                            epoch=epoch,
                            iteration=total_batch,
                            split='test',
                            loss=test_loss,
                            accuracy=test_acc,
                            learning_rate=learning_rate,
                            best_model=True,
                            num_samples=len(test_dataset),
                            extra_metrics={"model": "quantum_attention"},
                        )
                    msg = (
                        f"Iter:{total_batch: 5d}, Train loss:{loss.item(): 3.5f}, acc:{train_acc: 3.2%}; "
                        f"Test loss:{test_loss: 3.5f}, acc:{test_acc: 3.2%}"
                    )
                model.train()
                p_bar.set_postfix_str(msg)
                logging.info(msg)
            total_batch += 1
            if using_validation and total_batch - last_improve >= early_stopping:
                stopping_flag = True
                break
        p_bar.close()
        if stopping_flag:
            break
    if stopping_flag:
        msg = "No optimization for a long time, auto-stopping..."
    else:
        msg = "The training of the model has been finished."
    logging.info(msg)
    print("Testing message: ", msg)
    if using_validation:
        test(model, f'{saved_dir}/{model_name}.pt', test_iter)
    else:
        torch.save(model.state_dict(), f'{saved_dir}/{model_name}.pt')
        with torch.no_grad():
            test_loss, test_acc = evaluate(model, test_iter)
        log_metrics(
            metrics_log_path,
            epoch=num_epochs - 1,
            iteration=total_batch,
            split='final_test',
            loss=test_loss,
            accuracy=test_acc,
            learning_rate=learning_rate,
            best_model=True,
            num_samples=len(test_dataset),
            extra_metrics={"model": "quantum_attention"},
        )
        msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
        logging.info(msg)
        print(msg)

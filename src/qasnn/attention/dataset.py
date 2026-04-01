r"""
Data processing and dataset utilities for QSANN model.
"""

import random
from typing import Dict
from torch.utils.data import Dataset

def deal_vocab(vocab_path: str) -> Dict[str, int]:
    r"""
    Get the map from the word to the index by the input vocabulary file.

    Args:
        vocab_path: The path of the vocabulary file.

    Returns:
        Return the map from the word to the corresponding index.
    """
    with open(vocab_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    word2idx = {word.strip(): idx for idx, word in enumerate(lines)}
    return word2idx


class TextDataset(Dataset):
    r"""
    The class to implement the text dataset.

    Args:
        file_path: The dataset file.
        word2idx: The map from the word to the corresponding index.
        pad_size: The size pad the text sequence to. Defaults to ``0``, which means no padding.
    """
    def __init__(self, file_path: str, word2idx: dict, pad_size: int = 0):
        super().__init__()
        self.contents = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                text, label = line.split('\t')
            else:
                text, label = line.rsplit(maxsplit=1)
            text = [word2idx.get(word, 0) for word in text.split()]
            if pad_size != 0:
                if len(text) >= pad_size:
                    text = text[:pad_size]
                else:
                    text.extend([0] * (pad_size - len(text)))
            if not label.isdigit():
                label = 1 if label.lower() in ['yes', 'true', 'positive', '1'] else 0
            else:
                label = int(label)
            self.contents.append((text, label))
        self.len_data = len(self.contents)

    def __getitem__(self, idx):
        return self.contents[idx]

    def __len__(self):
        return self.len_data


def build_iter(dataset: TextDataset, batch_size: int, shuffle: bool = False) -> list:
    r"""
    Build the iteration of the batch data.

    Args:
        dataset: The dataset to be built.
        batch_size: The number of the data in a batch.
        shuffle: Whether to randomly shuffle the order of the data. Defaults to ``False``.

    Returns:
        The built iteration which contains the batches of the data.
    """
    data_iter = []
    if shuffle:
        random.shuffle(dataset.contents)
    for idx in range(0, len(dataset), batch_size):
        batch_data = dataset[idx: idx + batch_size]
        texts = [token_ids for token_ids, _ in batch_data]
        labels = [label for _, label in batch_data]
        data_iter.append((texts, labels))
    return data_iter

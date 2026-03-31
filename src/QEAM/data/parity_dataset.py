import torch
from torch.utils.data import Dataset, DataLoader
import random


class ParityDataset(Dataset):
    """
    Sequence parity classification.

    Label = parity of number of 1s in sequence.
    """

    def __init__(self, seq_len=16, size=1000):
        self.seq_len = seq_len
        self.size = size

        self.data = []
        self.labels = []

        for _ in range(size):
            seq = torch.randint(0, 2, (seq_len,))
            label = seq.sum() % 2

            self.data.append(seq)
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx].long(),
            "labels": torch.tensor(self.labels[idx]).long()
        }


def load_parity_dataset(seq_len=16, train_size=1000, test_size=200, batch_size=16):

    train_dataset = ParityDataset(seq_len, train_size)
    test_dataset = ParityDataset(seq_len, test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, 2
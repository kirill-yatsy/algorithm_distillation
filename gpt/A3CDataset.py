import numpy as np
import torch

from gpt.config import CFG


class A3CDataset(torch.utils.data.Dataset):
    def __init__(self, global_history, tokenizer):
        self.global_history = global_history
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.global_history)

    def crop(self, arr):
        if len(arr) > CFG.block_size // 4:
            # make sequence CFG.block_size wize by randomly cropping the sequence
            start_index = np.random.randint(0, len(arr) - CFG.block_size // 4)
            arr = arr[start_index : start_index + CFG.block_size // 4]

        take_first = np.random.randint(2, len(arr))
        target = arr[-1]
        arr = arr[: take_first - 1]
        return arr, target

    def __getitem__(self, idx):
        learning_history = self.global_history[idx]

        learning_history, target = self.crop(learning_history)

        tokenized = self.tokenizer(learning_history)
        tensor = torch.tensor(tokenized, dtype=torch.long)
        action = torch.tensor(target[2], dtype=torch.long)
        if tensor.shape[0] > CFG.block_size:
            print(tensor.shape)

        return tensor, action
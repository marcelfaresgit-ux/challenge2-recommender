import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.users = torch.tensor(frame["user_idx"].to_numpy(), dtype=torch.long)
        self.items = torch.tensor(frame["item_idx"].to_numpy(), dtype=torch.long)
        self.targets = torch.tensor(frame["target"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index], self.targets[index]

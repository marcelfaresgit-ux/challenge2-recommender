import torch
from torch import nn


class NeuralRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_embedding(user_idx)
        item_vector = self.item_embedding(item_idx)
        features = torch.cat([user_vector, item_vector], dim=1)
        return self.network(features).squeeze(1)

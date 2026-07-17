import torch
import torch.nn as nn
from src.models.gait_cnn import GaitCNN


class AuthModel(nn.Module):
    """Siamese CNN+LSTM authenticator (CNNfix+LSTM variant from the professor's code).

    Both windows pass through the same CNN encoder, then the concatenated feature
    maps are fed to a 2-layer LSTM. Whether the CNN weights are frozen or not is
    controlled by the caller (NB03 freezes them; the class itself is agnostic).

    Forward pass:
      x1, x2 → CNN.get_feature_maps() → t1, t2  each (B, 16, 128)
      concat([t1, t2], dim=1)                    → (B, 32, 128)
      LSTM(input=128, hidden=64, layers=2)        → last hidden state (B, 64)
      FC(64 → 2)                                  → logits (B, 2)
        index 0 = same person
        index 1 = different person
      softmax[:, 1] = P(different person); attack success = softmax[:, 1] < 0.5
    """

    def __init__(self, cnn_encoder: GaitCNN):
        super().__init__()
        self.cnn  = cnn_encoder

        # n_hidden=64, n_layers=2 matches the professor's config
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Return logits (B, 2) for [same person, different person].

        D5 label convention: raw {1,2} → after (y-1) → {0=same, 1=different},
        so softmax[:, 1] is P(different person).
        """
        t1 = self.cnn.get_feature_maps(x1)       # (B, 16, 128)
        t2 = self.cnn.get_feature_maps(x2)       # (B, 16, 128)
        ct = torch.cat([t1, t2], dim=1)          # (B, 32, 128)
        out, _ = self.lstm(ct)                   # (B, 32, 64)
        return self.fc(out[:, -1, :])            # (B, 2)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Return P(different person) ∈ [0, 1] for each pair, no grad."""
        with torch.no_grad():
            logits = self.forward(x1, x2)
            return torch.softmax(logits, dim=1)[:, 1]   # P(different person)

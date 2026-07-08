import torch
import torch.nn as nn


class GaitCNN(nn.Module):
    """CNN encoder matching the professor's TF architecture (Zou et al. 2020).

    Input:  (batch, 6, 128): 6 IMU channels, 128 timesteps
    Internally reshaped to (batch, 1, 6, 128) for 2D convolutions.

    Layer shapes (batch dim omitted):
      after conv1 + pool1 : (32,  6, 32)
      after conv2         : (64,  6, 32)
      after conv3 + pool2 : (128, 6, 16)
      after conv4         : (128, 1, 16)
      embedding (flatten) : (2048,)
      feature maps        : (16, 128)   ← used by the authenticator LSTM
      logits              : (n_classes,)
    """

    def __init__(self, n_classes: int, n_channels: int = 6):
        super().__init__()
        self.n_classes  = n_classes
        self.n_channels = n_channels

        # Block 1: conv 1×9 stride 2 → pool 1×2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 2: conv 1×3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        # Block 3: conv 1×3 → pool 1×2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 4: conv 6×1 (collapses channel height → produces feature maps)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(n_channels, 1), stride=(1, 1), padding=0)

        self.relu = nn.ReLU()
        self.fc   = nn.Linear(2048, n_classes)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv stack so get_feature_maps() and encode() don't duplicate the forward pass."""
        x = x.unsqueeze(1)                          # (B, 1, 6, 128)
        x = self.relu(self.conv1(x))                # (B, 32, 6, 64)
        x = self.pool1(x)                           # (B, 32, 6, 32)
        x = self.relu(self.conv2(x))                # (B, 64, 6, 32)
        x = self.relu(self.conv3(x))                # (B, 128, 6, 32)
        x = self.pool2(x)                           # (B, 128, 6, 16)
        x = self.relu(self.conv4(x))                # (B, 128, 1, 16)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps shaped (batch, 16, 128) for the authenticator LSTM."""
        out = self._backbone(x)                     # (B, 128, 1, 16)
        out = out.squeeze(2)                        # (B, 128, 16)
        return out.transpose(1, 2)                  # (B, 16, 128)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the flattened 2048-dim embedding for t-SNE and distance analysis.
        For the authenticator LSTM use get_feature_maps() instead: it preserves the temporal structure."""
        out = self._backbone(x)                     # (B, 128, 1, 16)
        return out.view(out.size(0), -1)            # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode(x))

import numpy as np
import torch
import torch.nn as nn

from .gait_cnn import GaitCNN

# Feature-map shape at Conv4 output: (B, FM_CHANNELS, FM_FEATURES)
FM_CHANNELS  = 16
FM_FEATURES  = 128


class CDMLGaitCNN(nn.Module):
    """GaitCNN with a Code Division Modulation Layer (CDML) at the feature-map level.

    Two key sizes are supported via *fm_code_dim*:

        fm_code_dim=128  (default, per-feature):
            code shape (128,) — broadcast across all 16 channels
            out = conv4(x) * code[None, None, :]       # (B, 16, 128)

        fm_code_dim=2048 (per-channel-per-feature):
            code shape (2048,) stored flat, reshaped to (16, 128) at apply time
            out = conv4(x) * code.view(16, 128)[None, :, :]   # (B, 16, 128)
            4× more entropy; stronger gradient isolation and harder MIA without key.

    During Phase 1 (CL training) per-task codes provide gradient isolation.
    During Phase 2 (LSTM auth training) a fixed global key conditions the surface.
    """

    def __init__(self, n_classes: int, n_channels: int = 6, fm_code_dim: int = 128):
        super().__init__()
        if fm_code_dim not in (FM_FEATURES, FM_CHANNELS * FM_FEATURES):
            raise ValueError(
                f"fm_code_dim must be {FM_FEATURES} (per-feature) or "
                f"{FM_CHANNELS * FM_FEATURES} (per-channel-per-feature), got {fm_code_dim}")
        self.backbone    = GaitCNN(n_classes=n_classes, n_channels=n_channels)
        self.code_dim    = fm_code_dim
        self._codes: dict[int, torch.Tensor] = {}
        self._active_code: torch.Tensor | None = None

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def fc(self):
        return self.backbone.fc

    def register_code(self, task_id: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.choice([-1.0, 1.0], size=self.code_dim).astype(np.float32)
        self._codes[task_id] = torch.from_numpy(raw)

    def set_task(self, task_id: int) -> None:
        if task_id not in self._codes:
            raise KeyError(f"No code registered for task {task_id}. "
                           "Call register_code() first.")
        self._active_code = self._codes[task_id]

    def set_fm_code_direct(self, code: torch.Tensor) -> None:
        if code.shape != (self.code_dim,):
            raise ValueError(f"code must be shape ({self.code_dim},), "
                             f"got {tuple(code.shape)}")
        self._active_code = code.float()

    def clear_task(self) -> None:
        self._active_code = None

    def get_code(self, task_id: int) -> torch.Tensor:
        return self._codes[task_id]

    # ── nn.Module interface ─────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """2048-dim embedding without any code applied."""
        return self.backbone.encode(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 16, 128) feature maps, modulated by the active code if set."""
        out = self.backbone.get_feature_maps(x)
        if self._active_code is not None:
            code = self._active_code.to(x.device)
            if self.code_dim == FM_FEATURES:
                out = out * code[None, None, :]          # broadcast across channels
            else:
                out = out * code.view(FM_CHANNELS, FM_FEATURES)[None, :, :]  # per-channel
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_feature_maps(x)   # (B, 16, 128), code applied if active
        h    = feat.flatten(1)             # (B, 2048)
        return self.backbone.fc(h)

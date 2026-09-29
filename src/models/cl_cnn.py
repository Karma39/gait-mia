import numpy as np
import torch
import torch.nn as nn

from .gait_cnn import GaitCNN


class CDMLGaitCNN(nn.Module):
    """GaitCNN with a Code Division Modulation Layer (CDML).

    Before the FC classification head, the 2048-dim embedding is modulated
    element-wise by a task-specific binary code s_k ∈ {-1, +1}^2048:
        h̃ = h ⊙ s_k
    Different tasks see differently-rotated embeddings from the FC head's
    perspective, reducing gradient interference between tasks (Milani, WIFS 2024).

    Privacy note: the code is applied AFTER the flatten, downstream of the
    get_feature_maps() tap used by the LSTM authenticator.  The authenticator
    therefore receives unmodulated conv4 feature maps regardless of which code
    is active — the CDML privacy guarantee does not extend to the authentication
    layer.

    Usage:
        model = CDMLGaitCNN(n_classes=98)
        model.register_code(task_id=0, seed=42)
        model.set_task(0)           # activate code for task 0
        logits = model(x)
    """

    def __init__(self, n_classes: int, n_channels: int = 6):
        super().__init__()
        self.backbone = GaitCNN(n_classes=n_classes, n_channels=n_channels)
        self.embedding_dim = 2048
        # codes[task_id] = float32 tensor of shape (2048,) with values ±1
        self._codes: dict[int, torch.Tensor] = {}
        self._active_code: torch.Tensor | None = None

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def fc(self):
        return self.backbone.fc

    def register_code(self, task_id: int, seed: int) -> None:
        """Generate and store the ±1 binary code for *task_id* from *seed*."""
        rng = np.random.default_rng(seed)
        raw = rng.choice([-1.0, 1.0], size=self.embedding_dim).astype(np.float32)
        self._codes[task_id] = torch.from_numpy(raw)

    def set_task(self, task_id: int) -> None:
        """Activate the code for *task_id*.  Call before each forward pass."""
        if task_id not in self._codes:
            raise KeyError(f"No code registered for task {task_id}. "
                           "Call register_code() first.")
        self._active_code = self._codes[task_id]

    def clear_task(self) -> None:
        """Deactivate modulation (identity forward pass, useful for eval)."""
        self._active_code = None

    def get_code(self, task_id: int) -> torch.Tensor:
        return self._codes[task_id]

    # ── nn.Module interface ─────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """2048-dim embedding *before* modulation (for t-SNE / distance analysis)."""
        return self.backbone.encode(x)

    def encode_modulated(self, x: torch.Tensor) -> torch.Tensor:
        """2048-dim embedding *after* applying the active code."""
        h = self.backbone.encode(x)
        if self._active_code is not None:
            h = h * self._active_code.to(x.device)
        return h

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_feature_maps(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode_modulated(x)
        return self.backbone.fc(h)

"""Continual-learning trainer for class-incremental gait identification.

Supports:
  - Naive fine-tuning (std): plain cross-entropy on the new task only.
  - CDML: same but with code modulation active (CDMLGaitCNN required).

Both match the experimental protocol in Milani WIFS 2024:
  400 epochs per task, Adam, LR 0.001 with exponential decay γ=0.9.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task_id: int
    method: str
    epochs: int
    train_acc: float
    # accuracy on *each* task evaluated right after training this task
    per_task_acc: dict[int, float] = field(default_factory=dict)
    # MIA AUC on Task-0 subjects (confidence-based, see mia_confidence_auc)
    mia_auc_task0: float = float("nan")
    elapsed_s: float = 0.0


class CLTrainer:
    """Train a model sequentially across CL tasks.

    Parameters
    ----------
    model:
        A GaitCNN or CDMLGaitCNN with an ``fc`` head pre-built for *n_total_classes*.
    device:
        Torch device.
    epochs_per_task:
        Training epochs per task (paper: 400).
    lr:
        Initial Adam learning rate (paper: 0.001).
    lr_gamma:
        Exponential LR decay factor per epoch (paper: 0.9).
    use_cdml:
        If True, call ``model.set_task(task_id)`` before each task's training
        and evaluation.  Requires *model* to be a ``CDMLGaitCNN``.
    seed_base:
        Seed for CDML code generation: task k uses seed ``seed_base + k``.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epochs_per_task: int = 400,
        lr: float = 1e-3,
        lr_gamma: float = 0.9,
        use_cdml: bool = False,
        seed_base: int = 1000,
    ):
        self.model = model
        self.device = device
        self.epochs_per_task = epochs_per_task
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.use_cdml = use_cdml
        self.seed_base = seed_base
        self._criterion = nn.CrossEntropyLoss()

        if use_cdml:
            assert hasattr(model, "register_code"), (
                "use_cdml=True requires a CDMLGaitCNN with register_code()."
            )

    # ── public API ──────────────────────────────────────────────────────────

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        eval_loaders: dict[int, DataLoader] | None = None,
        mia_loader_task0: DataLoader | None = None,
        mia_nonmember_loader: DataLoader | None = None,
        task0_class_range: tuple[int, int] | None = None,
        progress_cb: Callable[[int, float], None] | None = None,
    ) -> TaskResult:
        """Train one CL task and return evaluation metrics.

        Parameters
        ----------
        task_id:
            0-indexed task number.
        train_loader:
            DataLoader for this task's training data (labels = global class indices).
        eval_loaders:
            Dict mapping task_id → DataLoader for evaluation (accuracy only).
        mia_loader_task0:
            DataLoader for Task-0 subjects (genuine windows, used for MIA scoring).
        mia_nonmember_loader:
            DataLoader for non-member (held-out) subjects.
        task0_class_range:
            (start, end) global class index range for Task-0 subjects, needed to
            compute per-subject classification confidence for MIA scoring.
        progress_cb:
            Optional callback(epoch, train_acc) called every 50 epochs.
        """
        t0 = time.time()
        method = "cdml" if self.use_cdml else "std"

        if self.use_cdml:
            self.model.register_code(task_id, seed=self.seed_base + task_id)
            self.model.set_task(task_id)

        self._current_loader = train_loader
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)

        last_train_acc = 0.0
        for epoch in range(1, self.epochs_per_task + 1):
            last_train_acc = self._train_epoch(optimizer)
            scheduler.step()
            if epoch % 50 == 0:
                log.debug(
                    "task %d/%s  epoch %d/%d  train_acc=%.3f",
                    task_id, method, epoch, self.epochs_per_task, last_train_acc,
                )
                if progress_cb:
                    progress_cb(epoch, last_train_acc)

        # per-task accuracy
        per_task_acc: dict[int, float] = {}
        if eval_loaders:
            for tid, loader in eval_loaders.items():
                per_task_acc[tid] = self._accuracy(loader, eval_task_id=tid)

        # MIA AUC (confidence-based)
        mia_auc = float("nan")
        if (mia_loader_task0 is not None
                and mia_nonmember_loader is not None
                and task0_class_range is not None):
            mia_auc = self._mia_confidence_auc(
                mia_loader_task0,
                mia_nonmember_loader,
                task0_class_range,
                eval_task_id=0,
            )

        result = TaskResult(
            task_id=task_id,
            method=method,
            epochs=self.epochs_per_task,
            train_acc=last_train_acc,
            per_task_acc=per_task_acc,
            mia_auc_task0=mia_auc,
            elapsed_s=time.time() - t0,
        )
        log.info(
            "Task %d [%s] done  train_acc=%.3f  mia_auc=%.3f  t=%.0fs",
            task_id, method, last_train_acc, mia_auc, result.elapsed_s,
        )
        return result

    # ── internal helpers ────────────────────────────────────────────────────

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        correct = total = 0
        # train_loader is passed in per-epoch via a closure; store it on self
        for x, y in self._current_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self._criterion(logits, y)
            loss.backward()
            optimizer.step()
            correct += (logits.detach().argmax(1) == y).sum().item()
            total += len(y)
        return correct / total if total else 0.0

    def _accuracy(self, loader: DataLoader, eval_task_id: int | None = None) -> float:
        if self.use_cdml and eval_task_id is not None:
            self.model.set_task(eval_task_id)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total += len(y)
        return correct / total if total else 0.0

    def _mia_confidence_auc(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
        class_range: tuple[int, int],
        eval_task_id: int,
    ) -> float:
        """Confidence-based MIA AUC from Milani WIFS 2024.

        Score for subject s = mean softmax probability on s's correct class.
        AUC measures separability between Task-0 members and held-out non-members.

        class_range: (start, end) — the global class indices belonging to Task-0.
        """
        from sklearn.metrics import roc_auc_score

        if self.use_cdml:
            self.model.set_task(eval_task_id)
        self.model.eval()

        def mean_confidence(loader: DataLoader, is_member: bool) -> list[float]:
            scores = []
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    probs = torch.softmax(self.model(x), dim=1)
                    # For each window, score = probability assigned to its true class
                    for i in range(len(y)):
                        cls = y[i].item()
                        scores.append((cls, probs[i, cls].item()))
            # Aggregate per-subject: mean confidence per class index
            from collections import defaultdict
            per_subj: dict = defaultdict(list)
            for cls, sc in scores:
                per_subj[cls].append(sc)
            return [float(np.mean(v)) for v in per_subj.values()]

        member_scores = mean_confidence(member_loader, is_member=True)
        nonmember_scores = mean_confidence(nonmember_loader, is_member=False)

        if not member_scores or not nonmember_scores:
            return float("nan")

        all_scores = member_scores + nonmember_scores
        labels = [1] * len(member_scores) + [0] * len(nonmember_scores)

        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, all_scores))

    def set_loader(self, loader: DataLoader) -> None:
        """Set the current task's training DataLoader (called before train_task)."""
        self._current_loader = loader

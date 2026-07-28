"""
PGD evasion attack on the gait authentication pipeline, targeting sensor-level input only.

Surface A: perturb raw IMU windows (B, 6, 128).
  Gradient path: x → CNN → feature maps → LSTM → FC → logits.

Perturbation norm: L2.
  L∞ concentrates the budget into a few large spikes, which correspond to
  impact events in IMU signals and are easy to detect. L2 bounds the total
  signal energy, producing smooth perturbations that resemble natural gait
  variation across the full window.

Attack objective (impersonation):
  target_label=0: push P(different) below 0.5 so the model accepts the pair.
"""

import numpy as np
import torch
import torch.nn.functional as F


def pgd_sensor(
    model,
    x1: torch.Tensor,
    x2: torch.Tensor,
    target_label: int,
    eps: float,
    alpha: float,
    K: int,
) -> torch.Tensor:
    """
    PGD attack on raw sensor input (Surface A), L2 perturbation constraint.

    model       : AuthModel in eval mode
    x1          : (B, 6, 128) float, probe window (the one we perturb)
    x2          : (B, 6, 128) float, reference window (fixed)
    target_label: 0 = impersonation (push P(different) below 0.5 so model accepts)
    eps         : L2 budget, total perturbation energy per sample
    alpha       : step size (eps/K recommended)
    K           : PGD iterations

    Returns x1_adv (B, 6, 128), detached.
    """
    x2     = x2.detach()
    x1_adv = x1.clone().detach()
    target = torch.full((x1.shape[0],), target_label, dtype=torch.long)

    for _ in range(K):
        x1_adv = x1_adv.requires_grad_(True)
        logits = model(x1_adv, x2)
        loss   = F.cross_entropy(logits, target)
        (grad,) = torch.autograd.grad(loss, x1_adv)

        with torch.no_grad():
            # L2 steepest ascent: normalise gradient to unit L2 norm per sample
            grad_norm = grad.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
            x1_adv    = x1_adv - alpha * grad / grad_norm

            # L2 projection: shrink delta onto L2 ball of radius eps if outside
            delta      = x1_adv - x1
            delta_norm = delta.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
            delta      = delta * torch.clamp(eps / delta_norm, max=1.0)
            x1_adv     = (x1 + delta).detach()

    return x1_adv


def pgd_repr_probe(
    model,
    t1: torch.Tensor,
    t2: torch.Tensor,
    target_label: int,
    eps: float,
    alpha: float,
    K: int,
) -> torch.Tensor:
    """
    PGD on Surface B, perturbing only the probe feature maps (t1).

    In the deployed system the phone sends t1 = CNN(x1) to the server; t2 is the
    enrolled reference stored server-side and is not transmitted. A MITM attacker
    can intercept and modify t1 only. This function models that constraint.

    model : AuthModel (only model.lstm and model.fc used)
    t1    : (B, 16, 128) probe CNN features, perturbed
    t2    : (B, 16, 128) reference CNN features, fixed
    Returns t1_adv (B, 16, 128), detached.
    """
    t2 = t2.detach()
    t1_adv = t1.clone().detach()
    target = torch.full((t1.shape[0],), target_label, dtype=torch.long)

    for _ in range(K):
        t1_adv = t1_adv.requires_grad_(True)
        feat = torch.cat([t1_adv, t2], dim=1)   # (B, 32, 128)
        out, _ = model.lstm(feat)
        logits = model.fc(out[:, -1, :])
        loss = F.cross_entropy(logits, target)
        (grad,) = torch.autograd.grad(loss, t1_adv)

        with torch.no_grad():
            t1_adv = t1_adv - alpha * grad.sign()
            delta = torch.clamp(t1_adv - t1, -eps, eps)
            t1_adv = (t1 + delta).detach()

    return t1_adv


# ── Perturbation metrics ──────────────────────────────────────────────────────

def psr(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    """
    Perturbation-to-Signal Ratio = mean_batch( ||δ||₂ / ||x||₂ ).
    Used for Surface A where the signal has physical meaning.
    """
    delta  = (x_adv - x_clean).reshape(x_clean.shape[0], -1)
    signal = x_clean.reshape(x_clean.shape[0], -1)
    return float((delta.norm(dim=1) / (signal.norm(dim=1) + 1e-8)).mean())


def mean_l2(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    """Mean L2 norm of the perturbation across the batch (Surface B metric)."""
    delta = (x_adv - x_clean).reshape(x_clean.shape[0], -1)
    return float(delta.norm(dim=1).mean())


# ── Evaluation helpers ────────────────────────────────────────────────────────

def batch_psame(model, x1: torch.Tensor, x2: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Return P(different person) for all pairs, in batches (Surface A).

    The name is misleading: this returns softmax[:, 1] = P(different person), not P(same).
    Labels are 0=same, 1=different, so attack success means score < 0.5 (model accepts impostor).
    """
    scores = []
    with torch.no_grad():
        for s in range(0, x1.shape[0], batch_size):
            logits = model(x1[s:s + batch_size], x2[s:s + batch_size])
            scores.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(scores)


def batch_psame_repr(model, feat: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Return P(different person) from feature maps, in batches (Surface B)."""
    scores = []
    with torch.no_grad():
        for s in range(0, feat.shape[0], batch_size):
            out, _ = model.lstm(feat[s:s + batch_size])
            logits = model.fc(out[:, -1, :])
            scores.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(scores)


def attack_success_rate(scores_adv: np.ndarray, target_label: int, threshold: float = 0.5) -> float:
    """Fraction of pairs where the attack reached its goal."""
    if target_label == 0:
        return float((scores_adv < threshold).mean())
    return float((scores_adv >= threshold).mean())


def pgd_sensor_variable_eps(
    model,
    x1: torch.Tensor,
    x2: torch.Tensor,
    target_label: int,
    eps_arr: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    PGD on raw sensor input with per-sample L2 budget.

    eps_arr : (B,) tensor, individual L2 budget per sample
    alpha   : eps_arr / K per sample (one step = full budget / K)
    Used during batched binary search where each pair has a different bracket midpoint.
    """
    x2 = x2.detach()
    x1_adv = x1.clone().detach()
    target = torch.full((x1.shape[0],), target_label, dtype=torch.long)
    eps_v   = eps_arr.view(-1, 1, 1)
    alpha_v = (eps_arr / K).view(-1, 1, 1)

    for _ in range(K):
        x1_adv = x1_adv.requires_grad_(True)
        logits = model(x1_adv, x2)
        loss   = F.cross_entropy(logits, target)
        (grad,) = torch.autograd.grad(loss, x1_adv)

        with torch.no_grad():
            grad_norm  = grad.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
            x1_adv     = x1_adv - alpha_v * grad / grad_norm
            delta      = x1_adv - x1
            delta_norm = delta.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
            delta      = delta * torch.clamp(eps_v / delta_norm, max=1.0)
            x1_adv     = (x1 + delta).detach()

    return x1_adv


def run_pgd_sensor_batched(
    model,
    x1: torch.Tensor,
    x2: torch.Tensor,
    target_label: int,
    eps: float,
    K: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """Apply pgd_sensor (L2) in mini-batches; returns full x1_adv tensor."""
    alpha = eps / K
    parts = []
    for s in range(0, x1.shape[0], batch_size):
        parts.append(pgd_sensor(model, x1[s:s + batch_size], x2[s:s + batch_size],
                                target_label, eps, alpha, K))
    return torch.cat(parts, dim=0)


def run_pgd_repr_probe_batched(
    model,
    t1: torch.Tensor,
    t2: torch.Tensor,
    target_label: int,
    eps: float,
    K: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """Apply pgd_repr_probe in mini-batches; returns full t1_adv tensor."""
    alpha = eps / K
    parts = []
    for s in range(0, t1.shape[0], batch_size):
        parts.append(pgd_repr_probe(model, t1[s:s + batch_size], t2[s:s + batch_size],
                                    target_label, eps, alpha, K))
    return torch.cat(parts, dim=0)


def batch_psame_repr_probe(
    model,
    t1: torch.Tensor,
    t2: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    """P(different person) from separate t1, t2 feature maps (Surface B scorer)."""
    scores = []
    with torch.no_grad():
        for s in range(0, t1.shape[0], batch_size):
            feat = torch.cat([t1[s:s + batch_size], t2[s:s + batch_size]], dim=1)
            out, _ = model.lstm(feat)
            logits = model.fc(out[:, -1, :])
            scores.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(scores)

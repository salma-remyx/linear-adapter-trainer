# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Synthetic hard negatives in embedding space for triplet training.

Adapted from *SynCo: Synthetic Hard Negatives for Contrastive Visual
Representation Learning* (https://arxiv.org/abs/2410.02401). SynCo shows that
generating synthetic hard negatives directly on the representation space —
rather than only sampling real ones — yields stronger contrastive features.

This module ports that core mechanism to the linear-adapter trainer. For each
anchor it ranks a candidate pool by similarity, keeps the hardest few, and
synthesizes a new negative from them with one of three parameter-free
strategies:

* ``interpolate`` — a random convex combination of the hardest negatives
  (a point on the manifold spanned by near-anchor negatives);
* ``extrapolate`` — the hardest negative pushed away from the pool centroid
  (a slightly out-of-distribution but still anchor-aligned negative);
* ``mix_anchor`` — the hardest negative blended toward the anchor direction,
  producing the hardest achievable synthetic negative.

Only a representative three of SynCo's six vision strategies are implemented,
and its MoCo momentum queue is replaced by the in-batch mined-negative pool the
trainer already holds — no extra encoder, buffer, or corpus access is needed.
The synthetic negative is fed to the existing triplet loss by selecting, per
anchor, the harder of the mined and synthetic negatives (which for a
single-term triplet loss is equivalent to adding the synthetic negative to the
objective and keeping the dominant term).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STRATEGIES = ("interpolate", "extrapolate", "mix_anchor")


@dataclass(slots=True)
class SyntheticNegativeConfig:
    """Configuration for synthetic hard-negative generation.

    Attributes:
        enabled: Turn synthesis on. When ``False`` the trainer is unchanged.
        strategy: One of :data:`STRATEGIES`.
        n_hard: Size of the per-anchor hardest-neighbor set drawn from the pool.
        alpha: Anchor-mixing weight for ``mix_anchor`` (0 keeps the negative,
            1 replaces it with the anchor direction).
        beta: Extrapolation strength for ``extrapolate``.
    """

    enabled: bool = False
    strategy: str = "mix_anchor"
    n_hard: int = 4
    alpha: float = 0.5
    beta: float = 0.25

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}.")
        if self.n_hard < 1:
            raise ValueError("n_hard must be >= 1.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0.")


def _unit(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _hardest_set(anchors: torch.Tensor, pool: torch.Tensor, n_hard: int) -> torch.Tensor:
    """Gather, per anchor, the ``n_hard`` pool rows most similar to it.

    Returns a ``(batch, k, dim)`` tensor with the hardest negative first.
    """
    sims = _unit(anchors) @ _unit(pool).t()  # (batch, pool)
    k = min(n_hard, pool.shape[0])
    idx = torch.topk(sims, k=k, dim=1).indices  # (batch, k)
    return pool[idx]  # (batch, k, dim)


def _interpolate(anchors: torch.Tensor, hard: torch.Tensor, cfg, gen) -> torch.Tensor:
    batch, k, _ = hard.shape
    weights = torch.rand(batch, k, generator=gen)
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(hard)
    return torch.einsum("bk,bkd->bd", weights, hard)


def _extrapolate(anchors: torch.Tensor, hard: torch.Tensor, cfg, gen) -> torch.Tensor:
    hardest = hard[:, 0]
    centroid = hard.mean(dim=1)
    return hardest + cfg.beta * (hardest - centroid)


def _mix_anchor(anchors: torch.Tensor, hard: torch.Tensor, cfg, gen) -> torch.Tensor:
    hardest = hard[:, 0]
    # Keep the synthetic negative at the negative's scale so cosine and
    # euclidean distances both stay comparable to real negatives.
    scale = hardest.norm(dim=-1, keepdim=True)
    anchor_dir = _unit(anchors) * scale
    return (1.0 - cfg.alpha) * hardest + cfg.alpha * anchor_dir


_STRATEGY_FNS = {
    "interpolate": _interpolate,
    "extrapolate": _extrapolate,
    "mix_anchor": _mix_anchor,
}


def synthesize_hard_negatives(
    anchors: torch.Tensor,
    pool: torch.Tensor,
    *,
    config: SyntheticNegativeConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate one synthetic hard negative per anchor from ``pool``.

    Args:
        anchors: ``(batch, dim)`` reference embeddings (the adapted queries).
        pool: ``(pool, dim)`` candidate negative embeddings.
        config: Strategy and mixing hyper-parameters.
        generator: Optional RNG for the stochastic ``interpolate`` strategy.

    Returns:
        A ``(batch, dim)`` tensor of synthetic negatives.
    """
    if anchors.ndim != 2 or pool.ndim != 2:
        raise ValueError("anchors and pool must be 2-D (batch, dim) tensors.")
    if pool.shape[0] == 0:
        raise ValueError("pool must contain at least one candidate negative.")
    hard = _hardest_set(anchors, pool, config.n_hard)
    return _STRATEGY_FNS[config.strategy](anchors, hard, config, generator)


def augment_hard_negatives(
    anchors: torch.Tensor,
    negatives: torch.Tensor,
    *,
    config: SyntheticNegativeConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return per-anchor negatives hardened with synthetic examples.

    Uses the batch's own mined ``negatives`` as the candidate pool, synthesizes
    a hard negative per anchor, and keeps whichever of the mined and synthetic
    negative is more similar to the anchor. When ``config.enabled`` is ``False``
    the mined negatives are returned unchanged.
    """
    if not config.enabled:
        return negatives
    synthetic = synthesize_hard_negatives(anchors, negatives, config=config, generator=generator)
    mined_sim = torch.nn.functional.cosine_similarity(_unit(anchors), _unit(negatives), dim=-1)
    synth_sim = torch.nn.functional.cosine_similarity(_unit(anchors), _unit(synthetic), dim=-1)
    take_synth = (synth_sim > mined_sim).unsqueeze(-1)
    return torch.where(take_synth, synthetic, negatives)

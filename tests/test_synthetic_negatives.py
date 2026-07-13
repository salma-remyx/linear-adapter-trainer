# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from linear_adapter_trainer import (
    AdapterTrainer,
    DatasetConfig,
    DatasetGenerator,
    HashingEmbedder,
    KnowledgeBase,
    TemplateQueryGenerator,
    TrainingConfig,
)
from linear_adapter_trainer.adapter.synthetic_negatives import (
    SyntheticNegativeConfig,
    augment_hard_negatives,
    synthesize_hard_negatives,
)

CORPUS = [
    "Photosynthesis lets plants convert sunlight into chemical energy.",
    "Black holes are regions of spacetime where gravity traps light.",
    "TCP/IP protocols provide reliable, ordered packet delivery.",
    "Vaccines train the immune system to recognize pathogens.",
    "Volcanoes release lava, ash, and gases from a magma chamber.",
    "Espresso forces hot water through finely ground coffee.",
]


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def test_config_rejects_bad_arguments():
    with pytest.raises(ValueError):
        SyntheticNegativeConfig(strategy="nope")
    with pytest.raises(ValueError):
        SyntheticNegativeConfig(alpha=2.0)
    with pytest.raises(ValueError):
        SyntheticNegativeConfig(n_hard=0)


def test_mix_anchor_negative_is_harder_than_pool():
    torch.manual_seed(0)
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pool = torch.tensor([[0.2, 1.0], [1.0, 0.3], [-1.0, -1.0]])
    cfg = SyntheticNegativeConfig(enabled=True, strategy="mix_anchor", alpha=0.6)
    synth = synthesize_hard_negatives(anchors, pool, config=cfg, generator=_generator())

    synth_sim = torch.nn.functional.cosine_similarity(anchors, synth, dim=-1)
    pool_sim = (
        (torch.nn.functional.cosine_similarity(anchors.unsqueeze(1), pool.unsqueeze(0), dim=-1))
        .max(dim=1)
        .values
    )
    # A synthetic negative mixed toward the anchor is at least as hard as the
    # hardest real pool member.
    assert torch.all(synth_sim >= pool_sim - 1e-6)


def test_augment_disabled_is_identity():
    anchors = torch.randn(4, 8)
    negatives = torch.randn(4, 8)
    cfg = SyntheticNegativeConfig(enabled=False)
    out = augment_hard_negatives(anchors, negatives, config=cfg, generator=_generator())
    assert torch.equal(out, negatives)


def test_augment_preserves_shape_for_all_strategies():
    anchors = torch.randn(5, 16)
    negatives = torch.randn(5, 16)
    for strategy in ("interpolate", "extrapolate", "mix_anchor"):
        cfg = SyntheticNegativeConfig(enabled=True, strategy=strategy)
        out = augment_hard_negatives(anchors, negatives, config=cfg, generator=_generator())
        assert out.shape == negatives.shape


def _dataset(kb: KnowledgeBase):
    return DatasetGenerator(
        knowledge_base=kb,
        embedder=HashingEmbedder(dimension=256),
        query_generator=TemplateQueryGenerator(seed=0),
        config=DatasetConfig(
            queries_per_chunk=3,
            negatives_per_query=2,
            strategy="mixed",
            mix={"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2},
            val_fraction=0.25,
            seed=0,
        ),
    ).generate(show_progress=False)


def test_trainer_runs_with_synthetic_negatives_and_never_degrades():
    # Wire the synthetic-negative hook through the real AdapterTrainer loop and
    # confirm it trains end-to-end (offline) without degrading the baseline.
    kb = KnowledgeBase.from_texts(CORPUS, ids=[f"c{i}" for i in range(len(CORPUS))])
    dataset = _dataset(kb)
    embedder = HashingEmbedder(dimension=256)
    trainer = AdapterTrainer(
        kb,
        embedder,
        TrainingConfig(
            epochs=5,
            batch_size=16,
            learning_rate=5e-3,
            monitor="mrr",
            patience=0,
            eval_ks=(1, 3, 5),
            synthetic_negatives=SyntheticNegativeConfig(
                enabled=True, strategy="mix_anchor", alpha=0.5
            ),
        ),
    )
    result = trainer.fit(dataset, verbose=False)

    assert "mrr" in result.best_metrics
    assert result.best_metrics["mrr"] >= result.baseline_metrics["mrr"] - 1e-9
    assert len(result.history) >= 1

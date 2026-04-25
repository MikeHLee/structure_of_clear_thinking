"""
Unit tests for merge_scorer.py — HANDOFF_08 W2 deliverable.

Run with:
    cd topics/ontological_induction_sequence_modeling
    python -m pytest tests/test_merge_scorer.py -v
"""

import json
import math
import sys
import os
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ghrr_encoder import GHRREncoder, HypervectorConfig
from hierarchical_tokenizer import (
    HierarchicalToken, HierarchicalTokenizer, Modality, NULL_PROVENANCE,
)
from merge_scorer import (
    CoherenceOracle,
    MergeGraph,
    MergeNode,
    MergeScorer,
    MergeScorerNet,
    MergeLossWeights,
    _merge_tokens,
    merge_loss,
)
from olog_core import OlogGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ghrr():
    return GHRREncoder(HypervectorConfig(dim=4096, seed=42))


@pytest.fixture(scope="module")
def tokenizer(ghrr):
    return HierarchicalTokenizer(ghrr)


@pytest.fixture(scope="module")
def olog():
    g = OlogGraph("test_olog")
    for t in ("Customer", "Cart", "Order", "Payment"):
        g.add_type(t)
    g.add_aspect("Customer", "Cart",    "creates")
    g.add_aspect("Cart",     "Order",   "becomes")
    g.add_aspect("Order",    "Payment", "requires")
    return g


@pytest.fixture(scope="module")
def scorer(olog):
    return MergeScorer(input_dim=4096, hidden=128, threshold=0.0,
                       device="cpu", olog=olog)


@pytest.fixture(scope="module")
def sequence(tokenizer):
    return [
        tokenizer.encode_token("Alice",   "Customer", "Alice",   Modality.ASSERTION, position=0),
        tokenizer.encode_token("creates", "Customer", "creates", Modality.ASSERTION, position=1),
        tokenizer.encode_token("cart_1",  "Cart",     "cart_1",  Modality.ASSERTION, position=2),
        tokenizer.encode_token("becomes", "Cart",     "becomes", Modality.ASSERTION, position=3),
        tokenizer.encode_token("ord_7",   "Order",    "ord_7",   Modality.ASSERTION, position=4),
    ]


# ---------------------------------------------------------------------------
# CoherenceOracle
# ---------------------------------------------------------------------------

class TestCoherenceOracle:
    def test_valid_edge_label_1(self, tokenizer, olog):
        oracle = CoherenceOracle(olog)
        ti = tokenizer.encode_token("a", "Customer", "a", Modality.ASSERTION)
        tj = tokenizer.encode_token("b", "Cart",     "b", Modality.ASSERTION)
        assert oracle.label(ti, tj) == 1.0

    def test_invalid_edge_label_0(self, tokenizer, olog):
        oracle = CoherenceOracle(olog)
        ti = tokenizer.encode_token("a", "Payment",  "a", Modality.ASSERTION)
        tj = tokenizer.encode_token("b", "Customer", "b", Modality.ASSERTION)
        assert oracle.label(ti, tj) == 0.0

    def test_untyped_returns_half(self, tokenizer, olog):
        oracle = CoherenceOracle(olog)
        ti = tokenizer.encode_token("a", "__UNTYPED__", "a", Modality.UNTYPED)
        tj = tokenizer.encode_token("b", "Cart",         "b", Modality.ASSERTION)
        assert oracle.label(ti, tj) == 0.5

    def test_no_olog_all_soft(self, tokenizer):
        oracle = CoherenceOracle(olog=None)
        ti = tokenizer.encode_token("a", "TypeA", "a", Modality.ASSERTION)
        tj = tokenizer.encode_token("b", "TypeB", "b", Modality.ASSERTION)
        assert oracle.label(ti, tj) == 0.5

    def test_batch_labels_shape(self, tokenizer, olog, sequence):
        oracle = CoherenceOracle(olog)
        pairs = list(zip(sequence, sequence[1:]))
        labels = oracle.batch_labels(pairs)
        assert labels.shape == (len(pairs),)
        assert labels.dtype == torch.float32


# ---------------------------------------------------------------------------
# MergeScorerNet
# ---------------------------------------------------------------------------

class TestMergeScorerNet:
    def test_output_shape_batched(self):
        net = MergeScorerNet(input_dim=64, hidden=16)
        ei = torch.randn(4, 64)
        ej = torch.randn(4, 64)
        out = net(ei, ej)
        assert out.shape == (4,)

    def test_output_shape_unbatched(self):
        net = MergeScorerNet(input_dim=64, hidden=16)
        out = net(torch.randn(64), torch.randn(64))
        assert out.shape == ()   # scalar

    def test_asymmetric(self):
        net = MergeScorerNet(input_dim=64, hidden=16)
        ei = torch.randn(64)
        ej = torch.randn(64)
        assert net(ei, ej).item() != net(ej, ei).item()

    def test_gradients_flow(self):
        net = MergeScorerNet(input_dim=64, hidden=16)
        ei = torch.randn(4, 64)
        ej = torch.randn(4, 64)
        loss = net(ei, ej).mean()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# merge_loss
# ---------------------------------------------------------------------------

class TestMergeLoss:
    def test_total_is_finite(self):
        logits = torch.randn(8)
        labels = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0])
        total, comps = merge_loss(logits, labels, lm_targets=None)
        assert math.isfinite(total.item())

    def test_components_present(self):
        logits = torch.randn(8)
        labels = torch.zeros(8)
        _, comps = merge_loss(logits, labels, lm_targets=None)
        assert set(comps.keys()) == {"coherence", "lm", "length", "entropy", "total"}

    def test_lm_term_absent_when_none(self):
        logits = torch.randn(8)
        labels = torch.zeros(8)
        _, comps = merge_loss(logits, labels, lm_targets=None)
        assert comps["lm"] == 0.0

    def test_entropy_positive(self):
        logits = torch.randn(8)
        labels = torch.zeros(8)
        _, comps = merge_loss(logits, labels, lm_targets=None)
        assert comps["entropy"] >= 0.0

    def test_loss_decreases_on_coherent_pairs(self):
        """After training steps on fixed inputs, coherence loss should drop."""
        torch.manual_seed(0)
        net    = MergeScorerNet(input_dim=64, hidden=16)
        opt    = torch.optim.Adam(net.parameters(), lr=1e-2)
        labels = torch.ones(8)
        # Fix inputs so gradient signal is consistent across steps
        ei_fixed = torch.randn(8, 64)
        ej_fixed = torch.randn(8, 64)
        before = None
        for _ in range(30):
            opt.zero_grad()
            logits = net(ei_fixed, ej_fixed)
            loss, comps = merge_loss(logits, labels, lm_targets=None,
                                     weights=MergeLossWeights(coherence=1.0, lm=0, length=0, entropy=0))
            if before is None:
                before = comps["coherence"]
            loss.backward()
            opt.step()
        assert comps["coherence"] < before, "Coherence loss should decrease"


# ---------------------------------------------------------------------------
# MergeScorer — score / train_step / greedy_merge
# ---------------------------------------------------------------------------

class TestMergeScorer:
    def test_score_returns_float(self, scorer, sequence):
        s = scorer.score(sequence[0], sequence[1])
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_score_batch_shape(self, scorer, sequence):
        pairs = list(zip(sequence, sequence[1:]))
        logits = scorer.score_batch(pairs)
        assert logits.shape == (len(pairs),)

    def test_train_step_returns_components(self, scorer, sequence):
        pairs = list(zip(sequence, sequence[1:]))
        opt   = torch.optim.Adam(scorer.net.parameters(), lr=1e-4)
        comps = scorer.train_step(pairs, opt)
        assert "total" in comps
        assert math.isfinite(comps["total"])

    def test_greedy_merge_reduces_length_at_low_threshold(self, scorer, tokenizer, sequence):
        scorer.threshold = -1e9   # accept all merges
        merged, mg = scorer.greedy_merge(sequence, tokenizer)
        # With threshold = -inf every pair gets merged (left-to-right, multi-pass)
        assert len(merged) < len(sequence)
        scorer.threshold = 0.0    # restore

    def test_greedy_merge_no_merge_at_high_threshold(self, scorer, tokenizer, sequence):
        scorer.threshold = 1e9    # reject all merges
        merged, mg = scorer.greedy_merge(sequence, tokenizer)
        assert len(merged) == len(sequence)
        assert mg._counter == 0
        scorer.threshold = 0.0    # restore

    def test_merge_graph_leaf_count(self, scorer, tokenizer, sequence):
        merged, mg = scorer.greedy_merge(sequence, tokenizer)
        assert len(mg.leaf_tokens()) == len(sequence)

    def test_merge_graph_final_matches_output(self, scorer, tokenizer, sequence):
        merged, mg = scorer.greedy_merge(sequence, tokenizer)
        assert len(mg.final_tokens()) == len(merged)

    def test_checkpoint_roundtrip(self, scorer, sequence, tmp_path):
        ckpt = tmp_path / "scorer.pt"
        scorer.save_checkpoint(ckpt)
        loaded = MergeScorer.load_checkpoint(ckpt, input_dim=4096, hidden=128)
        s1 = scorer.score(sequence[0], sequence[1])
        s2 = loaded.score(sequence[0], sequence[1])
        assert abs(s1 - s2) < 1e-5


# ---------------------------------------------------------------------------
# _merge_tokens
# ---------------------------------------------------------------------------

class TestMergeTokens:
    def test_merged_text_contains_both(self, tokenizer, sequence):
        merged = _merge_tokens(sequence[0], sequence[2], tokenizer)
        assert "Alice" in merged.text
        assert "cart_1" in merged.text

    def test_merged_type_is_right_type(self, tokenizer, sequence):
        merged = _merge_tokens(sequence[0], sequence[2], tokenizer)
        assert merged.slots.type_code == sequence[2].slots.type_code

    def test_merged_provenance_is_left(self, tokenizer):
        tok_i = tokenizer.encode_token("a", "Customer", "a",
                                       Modality.ASSERTION, provenance_code="prov_left")
        tok_j = tokenizer.encode_token("b", "Cart", "b",
                                       Modality.ASSERTION, provenance_code="prov_right")
        merged = _merge_tokens(tok_i, tok_j, tokenizer)
        assert merged.slots.provenance_code == "prov_left"

    def test_merged_modality_promotion(self, tokenizer):
        tok_i = tokenizer.encode_token("a", "Customer", "a", Modality.HYPOTHESIS)
        tok_j = tokenizer.encode_token("b", "Cart",     "b", Modality.ASSERTION)
        merged = _merge_tokens(tok_i, tok_j, tokenizer)
        assert merged.slots.modality_code == Modality.ASSERTION

    def test_merged_embedding_is_unit(self, tokenizer, sequence):
        merged = _merge_tokens(sequence[0], sequence[2], tokenizer)
        norm = np.linalg.norm(merged.embedding)
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MergeGraph
# ---------------------------------------------------------------------------

class TestMergeGraph:
    def test_leaf_nodes(self, scorer, tokenizer, sequence):
        scorer.threshold = 1e9   # no merges
        _, mg = scorer.greedy_merge(sequence, tokenizer)
        assert len(mg.leaf_tokens()) == len(sequence)
        scorer.threshold = 0.0

    def test_export_schema(self, scorer, tokenizer, sequence, tmp_path):
        _, mg = scorer.greedy_merge(sequence, tokenizer)
        mg.save(tmp_path / "mg.json")
        with open(tmp_path / "mg.json") as f:
            data = json.load(f)
        assert "n_merges" in data
        assert "n_final"  in data
        assert "n_leaves" in data
        assert "tree"     in data

    def test_leaf_count_invariant(self, scorer, tokenizer, sequence):
        for thr in (-1e9, 0.0, 1e9):
            scorer.threshold = thr
            _, mg = scorer.greedy_merge(sequence, tokenizer)
            assert len(mg.leaf_tokens()) == len(sequence)
        scorer.threshold = 0.0

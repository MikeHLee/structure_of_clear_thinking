"""
Unit tests for ontological_attention.py W3 additions:
  - SlotMaskMode enum
  - create_mask_from_hierarchical()
  - embed_hierarchical_tokens()
  - forward_hierarchical()
  - run_slot_ablation()

Existing TypedToken-based tests are unaffected.

Run with:
    cd topics/ontological_induction_sequence_modeling
    python -m pytest tests/test_ontological_attention_w3.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from olog_core import OlogGraph
from ghrr_encoder import GHRREncoder, HypervectorConfig
from hierarchical_tokenizer import (
    HierarchicalTokenizer, Modality, NULL_PROVENANCE, UNTYPED_TYPE,
)
from ontological_attention import (
    OntologicalAttention,
    SlotMaskMode,
    SlotAblationResult,
    run_slot_ablation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def olog():
    g = OlogGraph("W3Test")
    for t in ("Customer", "Cart", "Order", "Payment"):
        g.add_type(t)
    g.add_aspect("Customer", "Cart",    "creates")
    g.add_aspect("Cart",     "Order",   "becomes")
    g.add_aspect("Order",    "Payment", "requires")
    return g


@pytest.fixture(scope="module")
def attention(olog):
    return OntologicalAttention(olog, embed_dim=64, allow_identity=True)


@pytest.fixture(scope="module")
def tokenizer():
    ghrr = GHRREncoder(HypervectorConfig(dim=4096, seed=42))
    return HierarchicalTokenizer(ghrr)


@pytest.fixture(scope="module")
def sequence(tokenizer):
    """5-token sequence covering all 4 types + one UNTYPED."""
    return [
        tokenizer.encode_token("Alice",   "Customer", "Alice",   Modality.ASSERTION,  position=0),
        tokenizer.encode_token("cart_1",  "Cart",     "cart_1",  Modality.ASSERTION,  position=1),
        tokenizer.encode_token("ord_7",   "Order",    "ord_7",   Modality.ASSERTION,  position=2),
        tokenizer.encode_token("pay_3",   "Payment",  "pay_3",   Modality.ASSERTION,  position=3),
        tokenizer.encode_token("unknown", UNTYPED_TYPE, "unk",   Modality.UNTYPED,    position=4),
    ]


@pytest.fixture(scope="module")
def mixed_modality_seq(tokenizer):
    """Sequence mixing ASSERTION and HYPOTHESIS tokens of the same type."""
    return [
        tokenizer.encode_token("Alice",     "Customer", "Alice",   Modality.ASSERTION,  position=0),
        tokenizer.encode_token("maybe_c",   "Customer", "maybe_c", Modality.HYPOTHESIS, position=1),
        tokenizer.encode_token("cart_1",    "Cart",     "cart_1",  Modality.ASSERTION,  position=2),
    ]


@pytest.fixture(scope="module")
def multi_prov_seq(tokenizer):
    """Sequence where some tokens share provenance and others don't."""
    return [
        tokenizer.encode_token("a", "Customer", "a", Modality.ASSERTION, provenance_code="src_A", position=0),
        tokenizer.encode_token("b", "Cart",     "b", Modality.ASSERTION, provenance_code="src_A", position=1),
        tokenizer.encode_token("c", "Order",    "c", Modality.ASSERTION, provenance_code="src_B", position=2),
    ]


# ---------------------------------------------------------------------------
# SlotMaskMode enum
# ---------------------------------------------------------------------------

class TestSlotMaskModeEnum:
    def test_three_values(self):
        assert len(SlotMaskMode) == 3

    def test_values(self):
        assert SlotMaskMode.TYPE_ONLY.value     == "type_only"
        assert SlotMaskMode.TYPE_MODALITY.value == "type_modality"
        assert SlotMaskMode.FULL.value          == "full"


# ---------------------------------------------------------------------------
# create_mask_from_hierarchical — TYPE_ONLY
# ---------------------------------------------------------------------------

class TestTypeOnlyMask:
    def test_shape(self, attention, sequence):
        n = len(sequence)
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask.shape == (n, n)

    def test_self_attention_diagonal(self, attention, sequence):
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert all(m.mask[i, i] == 1 for i in range(len(sequence)))

    def test_valid_edge_allowed(self, attention, sequence):
        # Customer(0) → Cart(1): edge exists
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask[0, 1] == 1

    def test_invalid_reverse_blocked(self, attention, sequence):
        # Payment(3) → Customer(0): no morphism path (DAG)
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask[3, 0] == 0

    def test_untyped_attends_everywhere(self, attention, sequence):
        # UNTYPED(4) should have mask[4, j] = 1 for all j
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask[4, :].sum() == len(sequence)

    def test_untyped_is_attended_by_all(self, attention, sequence):
        # Everyone attends to UNTYPED(4)
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask[:, 4].sum() == len(sequence)

    def test_transitive_reachability(self, attention, sequence):
        # Customer(0) should reach Payment(3) via Customer→Cart→Order→Payment
        m = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        assert m.mask[0, 3] == 1


# ---------------------------------------------------------------------------
# create_mask_from_hierarchical — TYPE_MODALITY
# ---------------------------------------------------------------------------

class TestTypeModalityMask:
    def test_assertion_cannot_attend_hypothesis(self, attention, mixed_modality_seq):
        # Alice(ASSERTION,Customer,0) → maybe_c(HYPOTHESIS,Customer,1)
        m = attention.create_mask_from_hierarchical(
            mixed_modality_seq, SlotMaskMode.TYPE_MODALITY
        )
        # Customer is self-reachable (identity), but ASSERTION→HYPOTHESIS blocked
        assert m.mask[0, 1] == 0

    def test_hypothesis_can_attend_assertion(self, attention, mixed_modality_seq):
        # maybe_c(HYPOTHESIS,Customer,1) → Alice(ASSERTION,Customer,0)
        # HYPOTHESIS→ASSERTION is NOT blocked (only ASSERTION→HYPOTHESIS is)
        m = attention.create_mask_from_hierarchical(
            mixed_modality_seq, SlotMaskMode.TYPE_MODALITY
        )
        assert m.mask[1, 0] == 1

    def test_assertion_attends_assertion(self, attention, mixed_modality_seq):
        # Alice(ASSERTION) → cart_1(ASSERTION) — type-reachable, same modality family
        m = attention.create_mask_from_hierarchical(
            mixed_modality_seq, SlotMaskMode.TYPE_MODALITY
        )
        assert m.mask[0, 2] == 1

    def test_type_modality_subset_of_type_only(self, attention, sequence):
        m_type = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        m_tmod = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_MODALITY)
        # TYPE_MODALITY can only block pairs that TYPE_ONLY allows
        assert (m_tmod.mask <= m_type.mask).all()


# ---------------------------------------------------------------------------
# create_mask_from_hierarchical — FULL
# ---------------------------------------------------------------------------

class TestFullMask:
    def test_full_subset_of_type_modality(self, attention, sequence):
        m_tmod = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_MODALITY)
        m_full = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.FULL)
        assert (m_full.mask <= m_tmod.mask).all()

    def test_coverage_ordering(self, attention, sequence):
        # TYPE_ONLY >= TYPE_MODALITY >= FULL in terms of allowed pairs
        m_type = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_ONLY)
        m_tmod = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.TYPE_MODALITY)
        m_full = attention.create_mask_from_hierarchical(sequence, SlotMaskMode.FULL)
        assert m_type.mask.sum() >= m_tmod.mask.sum() >= m_full.mask.sum()


# ---------------------------------------------------------------------------
# embed_hierarchical_tokens
# ---------------------------------------------------------------------------

class TestEmbedHierarchical:
    def test_shape(self, attention, sequence):
        X = attention.embed_hierarchical_tokens(sequence)
        assert X.shape == (len(sequence), attention.embed_dim)

    def test_empty_sequence(self, attention):
        X = attention.embed_hierarchical_tokens([])
        assert X.shape == (0, attention.embed_dim)

    def test_deterministic(self, attention, sequence):
        X1 = attention.embed_hierarchical_tokens(sequence)
        X2 = attention.embed_hierarchical_tokens(sequence)
        np.testing.assert_array_equal(X1, X2)

    def test_different_tokens_different_embeddings(self, attention, sequence):
        X = attention.embed_hierarchical_tokens(sequence)
        # Not all rows identical (tokens have different content)
        assert not np.allclose(X[0], X[1])


# ---------------------------------------------------------------------------
# forward_hierarchical
# ---------------------------------------------------------------------------

class TestForwardHierarchical:
    def test_output_shape(self, attention, sequence):
        out, _ = attention.forward_hierarchical(sequence)
        assert out.shape == (len(sequence), attention.embed_dim)

    def test_attention_weights_returned(self, attention, sequence):
        out, weights = attention.forward_hierarchical(sequence, return_attention=True)
        assert weights is not None
        assert weights.shape == (len(sequence), len(sequence))

    def test_attention_weights_none_by_default(self, attention, sequence):
        _, weights = attention.forward_hierarchical(sequence)
        assert weights is None

    def test_masked_rows_sum_to_one_or_zero(self, attention, sequence):
        _, weights = attention.forward_hierarchical(
            sequence, return_attention=True, mode=SlotMaskMode.TYPE_ONLY
        )
        # Each row should sum to ~1 (valid) or ~0 (fully masked, no valid key)
        row_sums = weights.sum(axis=-1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-5 or abs(s) < 1e-5

    def test_three_modes_give_different_outputs(self, attention, mixed_modality_seq):
        outs = [
            attention.forward_hierarchical(mixed_modality_seq, mode=m)[0]
            for m in SlotMaskMode
        ]
        # At least two modes should differ (modality gate blocks at least one pair)
        any_diff = any(not np.allclose(outs[i], outs[j])
                       for i in range(3) for j in range(i+1, 3))
        assert any_diff, "Different SlotMaskModes should produce different outputs"


# ---------------------------------------------------------------------------
# run_slot_ablation
# ---------------------------------------------------------------------------

class TestRunSlotAblation:
    def test_returns_all_modes(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        assert set(results.keys()) == set(SlotMaskMode)

    def test_result_type(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        for v in results.values():
            assert isinstance(v, SlotAblationResult)

    def test_coverage_in_unit_interval(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        for r in results.values():
            assert 0.0 <= r.coverage_ratio <= 1.0

    def test_coverage_ordering(self, attention, mixed_modality_seq):
        results = run_slot_ablation(attention, mixed_modality_seq)
        c_type = results[SlotMaskMode.TYPE_ONLY].coverage_ratio
        c_tmod = results[SlotMaskMode.TYPE_MODALITY].coverage_ratio
        c_full = results[SlotMaskMode.FULL].coverage_ratio
        assert c_type >= c_tmod >= c_full

    def test_mod_blocked_nonzero_for_mixed_modality(self, attention, mixed_modality_seq):
        results = run_slot_ablation(attention, mixed_modality_seq)
        assert results[SlotMaskMode.TYPE_MODALITY].mod_blocked > 0

    def test_mod_blocked_zero_for_uniform_assertion(self, attention, sequence):
        # sequence has all ASSERTION tokens (except UNTYPED, which bypasses)
        results = run_slot_ablation(attention, sequence[:4])   # drop UNTYPED
        assert results[SlotMaskMode.TYPE_MODALITY].mod_blocked == 0

    def test_entropy_in_unit_interval(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        for r in results.values():
            assert 0.0 <= r.entropy <= 1.0 + 1e-6

    def test_summary_string(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        for r in results.values():
            s = r.summary()
            assert r.mode.value in s
            assert "coverage" in s

    def test_mask_shape(self, attention, sequence):
        results = run_slot_ablation(attention, sequence)
        n = len(sequence)
        for r in results.values():
            assert r.mask.shape == (n, n)

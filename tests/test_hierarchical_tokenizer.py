"""
Unit tests for hierarchical_tokenizer.py — HANDOFF_08 W1 deliverable.

Run with:
    cd topics/ontological_induction_sequence_modeling
    python -m pytest tests/test_hierarchical_tokenizer.py -v
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ghrr_encoder import GHRREncoder, HypervectorConfig
from hierarchical_tokenizer import (
    HierarchicalTokenizer,
    Modality,
    NULL_PROVENANCE,
    UNTYPED_TYPE,
    CANNOT_ANSWER_TYPE,
    TokenSlots,
)


@pytest.fixture(scope="module")
def tokenizer():
    ghrr = GHRREncoder(HypervectorConfig(dim=4096, seed=42))
    return HierarchicalTokenizer(ghrr)


# ---------------------------------------------------------------------------
# Embedding shape & norm
# ---------------------------------------------------------------------------

class TestEmbeddingProperties:
    def test_embedding_shape(self, tokenizer):
        t = tokenizer.encode_token("foo", "TypeA", "foo", Modality.ASSERTION)
        assert t.embedding.shape == (4096,)

    def test_embedding_normalized(self, tokenizer):
        t = tokenizer.encode_token("foo", "TypeA", "foo", Modality.ASSERTION)
        norm = np.linalg.norm(t.embedding)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_slot_hvs_stored(self, tokenizer):
        t = tokenizer.encode_token("foo", "TypeA", "foo", Modality.ASSERTION)
        for attr in ("_type_hv", "_content_hv", "_modality_hv", "_provenance_hv"):
            assert getattr(t, attr) is not None


# ---------------------------------------------------------------------------
# Non-commutativity — core correctness guarantee
# ---------------------------------------------------------------------------

class TestNonCommutativity:
    def test_slot_order_matters(self, tokenizer):
        """bind(type, content) ≠ bind(content, type)"""
        ghrr = tokenizer._ghrr
        type_hv  = tokenizer.type_vocab.encode("Entity")
        cont_hv  = tokenizer.content_vocab.encode("value")
        mod_hv   = tokenizer.modality_vocab.encode(Modality.ASSERTION.value)
        prov_hv  = tokenizer.provenance_vocab.encode(NULL_PROVENANCE)

        fwd = tokenizer.bind_slots(type_hv, cont_hv, mod_hv, prov_hv)
        rev = tokenizer.bind_slots(cont_hv, type_hv, mod_hv, prov_hv)
        sim = ghrr.similarity(fwd, rev)
        assert sim < 0.3, f"Binding should be non-commutative, got sim={sim:.4f}"

    def test_distinct_types_distinct_embeddings(self, tokenizer):
        t1 = tokenizer.encode_token("x", "TypeA", "x", Modality.ASSERTION)
        t2 = tokenizer.encode_token("x", "TypeB", "x", Modality.ASSERTION)
        sim = tokenizer.similarity(t1, t2)
        assert sim < 0.9, "Different types should yield different embeddings"

    def test_distinct_modalities_distinct_embeddings(self, tokenizer):
        t1 = tokenizer.encode_token("x", "TypeA", "x", Modality.ASSERTION)
        t2 = tokenizer.encode_token("x", "TypeA", "x", Modality.HYPOTHESIS)
        sim = tokenizer.similarity(t1, t2)
        assert sim < 0.9


# ---------------------------------------------------------------------------
# Type-slot isolation
# ---------------------------------------------------------------------------

class TestTypeSlotIsolation:
    def test_same_type_high_type_sim(self, tokenizer):
        """Two tokens with the same type_code should have type_sim ≈ 1."""
        a = tokenizer.encode_token("/m/027rn",  "Entity", "/m/027rn",  Modality.ASSERTION)
        b = tokenizer.encode_token("/m/017dcd", "Entity", "/m/017dcd", Modality.ASSERTION)
        sim = tokenizer.type_similarity(a, b)
        assert sim > 0.99, f"Same type → type_sim should be ~1.0, got {sim:.4f}"

    def test_diff_type_low_type_sim(self, tokenizer):
        """Entity vs Relation type slots should be near-orthogonal."""
        a = tokenizer.encode_token("/m/027rn",  "Entity",     "/m/027rn",  Modality.ASSERTION)
        c = tokenizer.encode_token("contains",  "__relation__", "contains", Modality.ASSERTION)
        sim = tokenizer.type_similarity(a, c)
        assert sim < 0.3, f"Diff types → type_sim should be small, got {sim:.4f}"


# ---------------------------------------------------------------------------
# Triple tokenization (FB15K-237 interface)
# ---------------------------------------------------------------------------

class TestTripleTokenization:
    def test_triple_returns_three_tokens(self, tokenizer):
        tokens = tokenizer.tokenize_triple(
            "/m/027rn",
            "/location/location/contains",
            "/m/06cx9",
            head_type="Entity",
            tail_type="Entity",
        )
        assert len(tokens) == 3

    def test_triple_positions(self, tokenizer):
        tokens = tokenizer.tokenize_triple("/m/027rn", "/loc/contains", "/m/06cx9")
        assert [t.position for t in tokens] == [0, 1, 2]

    def test_triple_head_tail_same_type_slot(self, tokenizer):
        tokens = tokenizer.tokenize_triple(
            "/m/A", "/rel/r", "/m/B",
            head_type="Entity", tail_type="Entity",
        )
        head, rel, tail = tokens
        assert head.slots.type_code == "Entity"
        assert tail.slots.type_code == "Entity"
        assert rel.slots.type_code  == "__relation__"

    def test_triple_provenance_stored(self, tokenizer):
        tokens = tokenizer.tokenize_triple(
            "/m/A", "/rel/r", "/m/B", provenance_id="prov_001"
        )
        for t in tokens:
            assert t.slots.provenance_code == "prov_001"

    def test_two_different_triples_have_different_embeddings(self, tokenizer):
        t1 = tokenizer.tokenize_triple("/m/A", "/rel/r", "/m/B")
        t2 = tokenizer.tokenize_triple("/m/X", "/rel/r", "/m/Y")
        # Head tokens should differ in content
        sim = tokenizer.similarity(t1[0], t2[0])
        assert sim < 0.99


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

class TestSpecialTokens:
    def test_cannot_answer_type_code(self, tokenizer):
        ca = tokenizer.cannot_answer_token()
        assert ca.slots.type_code == CANNOT_ANSWER_TYPE
        assert not ca.slots.is_typed()

    def test_untyped_token_type_code(self, tokenizer):
        un = tokenizer.untyped_token("some span", "some_content")
        assert un.slots.type_code == UNTYPED_TYPE
        assert not un.slots.is_typed()

    def test_cannot_answer_not_grounded(self, tokenizer):
        ca = tokenizer.cannot_answer_token()
        assert not ca.slots.is_grounded()

    def test_grounded_token(self, tokenizer):
        t = tokenizer.encode_token("foo", "TypeA", "foo",
                                   Modality.ASSERTION, provenance_code="witness_42")
        assert t.slots.is_grounded()


# ---------------------------------------------------------------------------
# Vocabulary growth
# ---------------------------------------------------------------------------

class TestVocabGrowth:
    def test_vocab_grows_on_new_type(self, tokenizer):
        before = len(tokenizer.type_vocab)
        tokenizer.encode_token("x", "__brand_new_type__", "x", Modality.ASSERTION)
        after = len(tokenizer.type_vocab)
        assert after > before

    def test_repeated_encode_no_vocab_growth(self, tokenizer):
        tokenizer.encode_token("x", "StableType", "x", Modality.ASSERTION)
        before = len(tokenizer.type_vocab)
        tokenizer.encode_token("y", "StableType", "y", Modality.ASSERTION)
        after = len(tokenizer.type_vocab)
        assert before == after, "Same type should not grow vocab"


# ---------------------------------------------------------------------------
# Unbind (round-trip sanity)
# ---------------------------------------------------------------------------

class TestUnbind:
    def test_unbind_type_matches_stored(self, tokenizer):
        t = tokenizer.encode_token("/m/027rn", "Entity", "/m/027rn", Modality.ASSERTION)
        recovered = tokenizer.unbind_type(t)
        sim = tokenizer._ghrr.similarity(recovered, t._type_hv)
        assert sim > 0.95, f"Unbind should recover type_hv, got sim={sim:.4f}"

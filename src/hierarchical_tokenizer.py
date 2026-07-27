"""
Hierarchical Ontological Tokenizer — HANDOFF_08 §2.1 / W1

Each token is a structured 4-slot tuple bound into a single GHRR hypervector:

    τ = bind(type_hv, bind(content_hv, bind(modality_hv, provenance_hv)))
        (left-associative, non-commutative — slot order is semantic)

Slot roles
----------
type_code       : which Olog object / morphism class this token belongs to.
                  Consumed by OntologicalAttention masks.
content_code    : the specific filler (entity name, value, surface span).
                  What the decoder is free to choose *within* the type.
modality_code   : epistemic status — ASSERTION / HYPOTHESIS / QUESTION /
                  CITATION / UNTYPED.
provenance_code : witness id → a pointer to a context span or external source.
                  What ProofObject leaves point to.

Special tokens
--------------
UNTYPED         : fallback when the merge scorer cannot assign a type with
                  confidence > τ.  Hallucination guarantees do not apply.
CANNOT_ANSWER   : emitted when §5 step-5 feasible set is empty.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from ghrr_encoder import GHRREncoder, HypervectorConfig


# ---------------------------------------------------------------------------
# Constants & enumerations
# ---------------------------------------------------------------------------

class Modality(str, Enum):
    ASSERTION  = "assertion"
    HYPOTHESIS = "hypothesis"
    QUESTION   = "question"
    CITATION   = "citation"
    UNTYPED    = "untyped"


UNTYPED_TYPE      = "__UNTYPED__"
CANNOT_ANSWER_TYPE = "__CANNOT_ANSWER__"
NULL_PROVENANCE   = "__NULL_PROV__"


# ---------------------------------------------------------------------------
# Slot vocabulary — manages per-slot hypervector lookup tables
# ---------------------------------------------------------------------------

class SlotVocab:
    """
    Maintains a deterministic hypervector for every string token in a slot.

    All entries are generated via the GHRREncoder's hash-seeded scheme so the
    vocab is reproducible across runs and can grow incrementally (no re-index).
    """

    def __init__(self, slot_name: str, ghrr: GHRREncoder) -> None:
        self.slot_name = slot_name
        self._ghrr = ghrr
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, token: str) -> np.ndarray:
        if token not in self._cache:
            # Namespace the token by slot to keep different slots orthogonal.
            namespaced = f"__slot_{self.slot_name}__{token}"
            self._cache[token] = self._ghrr.encode_type(namespaced)
        return self._cache[token]

    def __len__(self) -> int:
        return len(self._cache)

    def tokens(self) -> List[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Token data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenSlots:
    """Raw string labels for each of the 4 slots before embedding."""
    type_code:       str
    content_code:    str
    modality_code:   Modality
    provenance_code: str    # witness id, or NULL_PROVENANCE

    def is_typed(self) -> bool:
        return self.type_code not in (UNTYPED_TYPE, CANNOT_ANSWER_TYPE)

    def is_grounded(self) -> bool:
        return self.provenance_code != NULL_PROVENANCE


@dataclass
class HierarchicalToken:
    """
    A fully-bound ontological token.

    `embedding` is the GHRR bind of all four slot hypervectors:
        embedding = bind(type_hv ⊛ bind(content_hv ⊛ bind(modality_hv ⊛ prov_hv)))

    The slots attribute retains the unbound string labels so callers (e.g.
    OntologicalAttention, ProofEngine) can inspect individual slots without
    unbinding.
    """
    slots:     TokenSlots
    embedding: np.ndarray     # shape: (dim,)
    text:      str            # original surface form
    position:  int = -1       # sequence position (set by tokenizer)

    # Slot hypervectors retained for unbinding / ablation studies
    _type_hv:       Optional[np.ndarray] = field(default=None, repr=False)
    _content_hv:    Optional[np.ndarray] = field(default=None, repr=False)
    _modality_hv:   Optional[np.ndarray] = field(default=None, repr=False)
    _provenance_hv: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"HierarchicalToken("
            f"type={self.slots.type_code!r}, "
            f"content={self.slots.content_code!r}, "
            f"mod={self.slots.modality_code.value}, "
            f"prov={self.slots.provenance_code!r}, "
            f"pos={self.position})"
        )


# ---------------------------------------------------------------------------
# Hierarchical tokenizer
# ---------------------------------------------------------------------------

class HierarchicalTokenizer:
    """
    Converts text spans + ontological metadata into HierarchicalTokens.

    Binding convention (non-commutative, left-associative):
        step1 = bind(modality_hv, provenance_hv)
        step2 = bind(content_hv,  step1)
        step3 = bind(type_hv,     step2)          ← final embedding

    Innermost bind applied first so that unbinding type gives back a vector
    that still encodes (content ⊛ modality ⊛ provenance), matching the
    OntologicalAttention mask which only inspects the type slot.

    Parameters
    ----------
    ghrr : GHRREncoder
        Shared encoder.  Caller owns it so config (dim, seed) is decided
        upstream.
    """

    def __init__(self, ghrr: Optional[GHRREncoder] = None) -> None:
        self._ghrr = ghrr or GHRREncoder()

        self.type_vocab      = SlotVocab("type",       self._ghrr)
        self.content_vocab   = SlotVocab("content",    self._ghrr)
        self.modality_vocab  = SlotVocab("modality",   self._ghrr)
        self.provenance_vocab = SlotVocab("provenance", self._ghrr)

        # Pre-warm special tokens so they are always in vocab
        for t in (UNTYPED_TYPE, CANNOT_ANSWER_TYPE):
            self.type_vocab.encode(t)
        self.provenance_vocab.encode(NULL_PROVENANCE)
        for m in Modality:
            self.modality_vocab.encode(m.value)

        # Cache CANNOT_ANSWER token (singleton)
        self._cannot_answer_token: Optional[HierarchicalToken] = None

    # ------------------------------------------------------------------
    # Core encode / bind
    # ------------------------------------------------------------------

    def bind_slots(
        self,
        type_hv:       np.ndarray,
        content_hv:    np.ndarray,
        modality_hv:   np.ndarray,
        provenance_hv: np.ndarray,
    ) -> np.ndarray:
        """
        Chain-bind four slot hypervectors.

        Order: type ⊛ (content ⊛ (modality ⊛ provenance))
        Non-commutativity ensures a permuted slot tuple produces a distinct
        (approximately orthogonal) hypervector.
        """
        inner = self._ghrr.bind(modality_hv,  provenance_hv)
        mid   = self._ghrr.bind(content_hv,   inner)
        outer = self._ghrr.bind(type_hv,       mid)
        return outer

    def encode_token(
        self,
        text:            str,
        type_code:       str,
        content_code:    str,
        modality:        Modality = Modality.ASSERTION,
        provenance_code: str      = NULL_PROVENANCE,
        position:        int      = -1,
    ) -> HierarchicalToken:
        """
        Build a single HierarchicalToken from explicit slot labels.

        This is the primary construction path; merge_scorer.py will call it
        after assigning slot labels.
        """
        type_hv  = self.type_vocab.encode(type_code)
        cont_hv  = self.content_vocab.encode(content_code)
        mod_hv   = self.modality_vocab.encode(modality.value)
        prov_hv  = self.provenance_vocab.encode(provenance_code)

        emb = self.bind_slots(type_hv, cont_hv, mod_hv, prov_hv)

        return HierarchicalToken(
            slots=TokenSlots(
                type_code=type_code,
                content_code=content_code,
                modality_code=modality,
                provenance_code=provenance_code,
            ),
            embedding=emb,
            text=text,
            position=position,
            _type_hv=type_hv,
            _content_hv=cont_hv,
            _modality_hv=mod_hv,
            _provenance_hv=prov_hv,
        )

    def cannot_answer_token(self, position: int = -1) -> HierarchicalToken:
        """Return the singleton CANNOT_ANSWER token (§5 step-5 firewall)."""
        tok = self.encode_token(
            text="<CANNOT_ANSWER>",
            type_code=CANNOT_ANSWER_TYPE,
            content_code=CANNOT_ANSWER_TYPE,
            modality=Modality.UNTYPED,
            provenance_code=NULL_PROVENANCE,
            position=position,
        )
        return tok

    def untyped_token(
        self,
        text:            str,
        content_code:    str,
        provenance_code: str = NULL_PROVENANCE,
        position:        int = -1,
    ) -> HierarchicalToken:
        """
        Wrap a span whose type cannot be determined with confidence.

        The type slot is set to UNTYPED so the attention mask knows not to
        route reasoning through this token.  Proof object marks the span
        as 'unverified' (not 'verified') per §7.4.
        """
        return self.encode_token(
            text=text,
            type_code=UNTYPED_TYPE,
            content_code=content_code,
            modality=Modality.UNTYPED,
            provenance_code=provenance_code,
            position=position,
        )

    # ------------------------------------------------------------------
    # Unbinding (ablation / interpretability)
    # ------------------------------------------------------------------

    def unbind_type(self, token: HierarchicalToken) -> np.ndarray:
        """
        Recover the type-slot hypervector from a bound token.

        Uses the GHRR pseudoinverse.  Useful for ablation studies where we
        want to compare type_hv similarity across tokens without decoding.
        """
        if token._type_hv is not None:
            return token._type_hv
        # Fallback: unbind from embedding using content+modality+provenance
        inner = self._ghrr.bind(
            self.modality_vocab.encode(token.slots.modality_code.value),
            self.provenance_vocab.encode(token.slots.provenance_code),
        )
        mid = self._ghrr.bind(
            self.content_vocab.encode(token.slots.content_code), inner
        )
        return self._ghrr.unbind(token.embedding, mid)

    def similarity(self, a: HierarchicalToken, b: HierarchicalToken) -> float:
        """Cosine similarity between two token embeddings."""
        return float(self._ghrr.similarity(a.embedding, b.embedding))

    def type_similarity(self, a: HierarchicalToken, b: HierarchicalToken) -> float:
        """Similarity restricted to the type slot hypervectors."""
        hv_a = a._type_hv if a._type_hv is not None else self.type_vocab.encode(a.slots.type_code)
        hv_b = b._type_hv if b._type_hv is not None else self.type_vocab.encode(b.slots.type_code)
        return float(self._ghrr.similarity(hv_a, hv_b))

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def tokenize_triple(
        self,
        head:     str,
        relation: str,
        tail:     str,
        head_type:     str = UNTYPED_TYPE,
        tail_type:     str = UNTYPED_TYPE,
        provenance_id: str = NULL_PROVENANCE,
    ) -> List[HierarchicalToken]:
        """
        Tokenize a KG triple (h, r, t) into 3 HierarchicalTokens.

        Maps naturally to FB15K-237 / WN18RR triples for W1 unit tests.
        The relation token uses type_code = "__relation__" and
        content_code = relation label.

        head_type and tail_type default to UNTYPED; callers that have Olog
        type information should supply it.
        """
        head_tok = self.encode_token(
            text=head,
            type_code=head_type,
            content_code=head,
            modality=Modality.ASSERTION,
            provenance_code=provenance_id,
            position=0,
        )
        rel_tok = self.encode_token(
            text=relation,
            type_code="__relation__",
            content_code=relation,
            modality=Modality.ASSERTION,
            provenance_code=provenance_id,
            position=1,
        )
        tail_tok = self.encode_token(
            text=tail,
            type_code=tail_type,
            content_code=tail,
            modality=Modality.ASSERTION,
            provenance_code=provenance_id,
            position=2,
        )
        return [head_tok, rel_tok, tail_tok]

    def tokenize_sequence(
        self,
        items: List[Tuple[str, str, str, Modality, str]],
    ) -> List[HierarchicalToken]:
        """
        Tokenize a list of (text, type_code, content_code, modality, provenance) tuples.

        Assigns sequential positions.
        """
        return [
            self.encode_token(
                text=text,
                type_code=type_code,
                content_code=content_code,
                modality=modality,
                provenance_code=provenance_code,
                position=i,
            )
            for i, (text, type_code, content_code, modality, provenance_code)
            in enumerate(items)
        ]


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

def _demo():
    print("=" * 64)
    print("HierarchicalTokenizer — W1 smoke test")
    print("=" * 64)

    ghrr = GHRREncoder(HypervectorConfig(dim=4096, seed=42))
    tok  = HierarchicalTokenizer(ghrr)

    # --- FB15K-237-style triple ---
    triples = [
        ("/m/027rn", "/location/location/contains", "/m/06cx9"),
        ("/m/017dcd", "/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor", "/m/06v8s0"),
    ]

    print("\n[1] Triple tokenization (FB15K-237 format)")
    for h, r, t in triples:
        tokens = tok.tokenize_triple(h, r, t,
                                     head_type="Entity",
                                     tail_type="Entity",
                                     provenance_id=f"fb15k_{hashlib.md5((h+r+t).encode()).hexdigest()[:8]}")
        for token in tokens:
            print(f"  {token}")
            print(f"    embedding norm: {np.linalg.norm(token.embedding):.4f}")

    # --- Non-commutativity: type ⊛ content ≠ content ⊛ type ---
    print("\n[2] Non-commutativity of bind_slots")
    type_hv  = tok.type_vocab.encode("Entity")
    cont_hv  = tok.content_vocab.encode("/m/027rn")
    mod_hv   = tok.modality_vocab.encode(Modality.ASSERTION.value)
    prov_hv  = tok.provenance_vocab.encode(NULL_PROVENANCE)

    bound_normal   = tok.bind_slots(type_hv, cont_hv, mod_hv, prov_hv)
    bound_swapped  = tok.bind_slots(cont_hv, type_hv, mod_hv, prov_hv)
    sim = ghrr.similarity(bound_normal, bound_swapped)
    print(f"  sim(type⊛content, content⊛type) = {sim:.4f}  (want: near 0)")

    # --- Type-slot similarity across same-type tokens ---
    print("\n[3] Type-slot similarity")
    tok_a = tok.encode_token("/m/027rn",  "Entity", "/m/027rn",  Modality.ASSERTION)
    tok_b = tok.encode_token("/m/017dcd", "Entity", "/m/017dcd", Modality.ASSERTION)
    tok_c = tok.encode_token("contains",  "__relation__", "contains", Modality.ASSERTION)

    sim_same_type = tok.type_similarity(tok_a, tok_b)
    sim_diff_type = tok.type_similarity(tok_a, tok_c)
    print(f"  Entity vs Entity (type sim):    {sim_same_type:.4f}  (want: ~1.0)")
    print(f"  Entity vs Relation (type sim):  {sim_diff_type:.4f}  (want: near 0)")

    # --- CANNOT_ANSWER token ---
    print("\n[4] Special tokens")
    ca = tok.cannot_answer_token()
    un = tok.untyped_token("some unknown span", "some_content")
    print(f"  {ca}")
    print(f"  {un}")
    print(f"  CANNOT_ANSWER is_typed: {ca.slots.is_typed()}")
    print(f"  UNTYPED is_typed:       {un.slots.is_typed()}")

    # --- Vocab sizes ---
    print("\n[5] Vocab sizes after demo")
    print(f"  type_vocab:       {len(tok.type_vocab)}")
    print(f"  content_vocab:    {len(tok.content_vocab)}")
    print(f"  modality_vocab:   {len(tok.modality_vocab)}")
    print(f"  provenance_vocab: {len(tok.provenance_vocab)}")

    print("\n" + "=" * 64)
    print("Smoke test complete.")
    print("=" * 64)


if __name__ == "__main__":
    import hashlib
    _demo()

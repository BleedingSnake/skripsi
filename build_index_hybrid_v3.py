#!/usr/bin/env python3
"""
build_index_hybrid_v3.py

Hybrid index builder + retriever for Indonesian legal documents (ITE Law).

=============================================================================
CHANGELOG v2 → v3  (what was broken and why, and how it was fixed)
=============================================================================

ROOT CAUSE ANALYSIS
-------------------
Running the v2 index against 120 sampled QnA pairs and cross-referencing
results against pasal_master.jsonl / penjelasan_master.jsonl revealed:

  - DEFINITION, GENERAL, PERMISSION, PROCEDURE  → 100% accuracy  ✅
  - PENALTY                                      →  85% accuracy  ⚠️
  - SPECIFIC_ARTICLE                             →   5% accuracy  ❌

The SPECIFIC_ARTICLE failures all shared the same pattern: when a user asks
about a specific pasal AND its sanction in the same query (e.g. "Apa isi
Pasal 27 ayat 3 dan sanksinya?"), the retriever consistently returns the
*sanction article* (Pasal 34, 45A, 47, 48) instead of Pasal 27 itself.

Three compounding causes were identified:

  CAUSE 1 — Embedding text has no structural identity anchor
  ----------------------------------------------------------
  v2 embeds raw clean_text only: "Setiap Orang dengan sengaja..."
  The sentence-transformer model sees semantically similar content across
  many pasal without any pasal-number signal. When the query contains
  "sanksi/hukuman", the semantic model gravitates toward penalty articles
  whose body text is about punishment — which is a perfectly valid semantic
  match, just not the pasal the user asked about.

  FIX: Prepend a structured identity prefix to EVERY embedded document:
       "{source} Pasal {pasal} Ayat {ayat}: {clean_text}"
  This anchors each embedding to its own pasal number, so a query containing
  "Pasal 27" has strong cosine affinity toward the "Pasal 27" embedding and
  not toward penalty articles that merely *reference* Pasal 27 internally.

  CAUSE 2 — Cross-reference pollution in penalty/sanction articles
  ---------------------------------------------------------------
  Penalty articles (Pasal 34, 45A, 46, 47, 48) explicitly list the offense
  articles they sanction: "...sebagaimana dimaksud dalam Pasal 27 sampai
  dengan Pasal 33...". These cross-references make BM25 score these penalty
  articles very highly for any query containing "Pasal 27", "Pasal 30", etc.
  Combined with semantic similarity from the mixed sanction+content query,
  they win the hybrid fusion.

  FIX: Structured prefix (Cause 1 fix) already anchors BM25 toward exact
  pasal identity. Additionally, for SPECIFIC_ARTICLE intent, we introduce
  a Structural Exact Match (SEM) override: if the query explicitly names a
  pasal and ayat, candidates whose (pasal, ayat) matches are given a hard
  boost BEFORE score normalization rather than a soft multiplicative factor
  after it. v2's pasal_boost defaulted to 0.0 and was applied after
  normalization — making it effectively a no-op unless manually tuned.

  CAUSE 3 — Dual-intent queries are not decomposed
  -------------------------------------------------
  Queries like "Apa isi Pasal 27 ayat 3 dan sanksinya?" contain two distinct
  retrieval goals: (a) retrieve the substantive pasal text, and (b) retrieve
  the related sanction article. v2 treats this as a single semantic query,
  which forces the model to trade off between the two goals — and the sanction
  half tends to dominate because sanction articles have dense, distinctive
  vocabulary.

  FIX: Introduce query intent detection. Dual-intent queries (containing both
  a specific pasal reference AND sanction keywords) are split into two sub-
  queries. Sub-query A retrieves the named pasal directly. Sub-query B looks
  up the corresponding penalty article via a pre-built sanctions_map. Results
  are merged and deduplicated. This means both retrieval goals are served
  without either dominating the other.

ADDITIONAL IMPROVEMENTS
  - BM25 tokenizer now uses simple but effective Indonesian stopword removal
    to reduce noise from common legal boilerplate ("yang", "dan", "atau",
    "dalam", "dengan") that appears in almost every pasal and carries no
    discriminative signal.
  - PENJELASAN lookup now also indexes by (source, pasal, ayat) triplet, not
    just (source, pasal), so ayat-level precision is improved.
  - SEM (Structural Exact Match) boost is deterministic and transparent:
    exposed as sem_boost parameter (default 2.0) so it can be tuned or
    disabled without changing the code.
  - Metadata now records embedding_strategy and index_version for traceability.

=============================================================================

Key Features (unchanged from v2):
- Unified index for both PASAL (law text) and PENJELASAN (explanations)
- BM25 (lexical) + FAISS (semantic) hybrid search
- Intelligent PENJELASAN linking based on retrieved PASAL
- Configurable scoring with optional query-based boosts
- Efficient lookup with pre-built indices

Input Format (JSONL):
{
  "chunk_id": "UU_11_2008_P27_A1",
  "clean_text": "Setiap Orang dengan sengaja...",
  "source": "UU 11/2008",
  "pasal": "27",
  "ayat": "1",
  "type": "PASAL"  // or "PENJELASAN"
}

Usage:
  # Build index
  python build_index_hybrid_v3.py build \
    --jsonl pasal_master.jsonl penjelasan_master.jsonl \
    --outdir ./index_v3

  # Query interactively
  python build_index_hybrid_v3.py query --index-dir ./index_v3

  # Query programmatically
  from build_index_hybrid_v3 import HybridRetriever
  retriever = HybridRetriever.load("./index_v3")
  results = retriever.search("Apa isi Pasal 27 ayat (3) dan sanksinya?", top_k=3)
"""

import os
import json
import argparse
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



# config

DEFAULT_MODEL = "sentence-transformers/LaBSE"
EMBEDDING_BATCH_SIZE = 64
MIN_SCORE_THRESHOLD = 0.01

INDEX_VERSION = "v3"

# indonesian stopwords — high-frequency legal boilerplate that carries
# no discriminative signal for pasal retrieval. Removing them lets BM25 focus on substantive terms (pasal numbers, legal concepts, actions).
ID_STOPWORDS: Set[str] = {
    "yang", "dan", "atau", "dalam", "dengan", "untuk", "tidak", "ini",
    "itu", "pada", "dari", "ke", "di", "oleh", "adalah", "juga", "sebagai",
    "tersebut", "telah", "akan", "dapat", "antara", "setiap", "orang",
    "undang", "pasal", "ayat", "ketentuan", "sebagaimana", "dimaksud",
    "hal", "hak", "hukum", "bahwa", "suatu", "satu", "lebih", "lain",
    "sesuai", "berdasarkan", "terhadap", "paling", "pun"
}

# sanction/penalty keywords used to detect dual-intent queries
SANCTION_KEYWORDS: Set[str] = {
    "sanksi", "hukuman", "pidana", "ancaman", "denda", "penjara",
    "hukum","dikenai", "dipidana", "penghukuman", "pemidanaan"
}



# data Classes

@dataclass
class Document:
    """represents a single legal document chunk"""
    chunk_id: str
    text: str           # raw clean_text from JSONL (used for display)
    embed_text: str     # structured text used for embedding (FIX: Cause 1)
    source: str
    pasal: str
    ayat: str
    doc_type: str       # "PASAL" or "PENJELASAN"

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embed_text": self.embed_text,
            "source": self.source,
            "pasal": self.pasal,
            "ayat": self.ayat,
            "type": self.doc_type
        }

    @classmethod
    def from_dict(cls, d):
        # embed_text may not exist in older saved documents
        raw_text = d.get("text", "")
        return cls(
            chunk_id=d.get("chunk_id", ""),
            text=raw_text,
            embed_text=d.get("embed_text", raw_text),
            source=d.get("source", ""),
            pasal=d.get("pasal", ""),
            ayat=d.get("ayat", ""),
            doc_type=d.get("type", "PASAL")
        )


@dataclass
class SearchResult:
    """Represents a search result with score"""
    document: Document
    score: float
    rank: int

    def to_dict(self):
        return {
            **self.document.to_dict(),
            "score": self.score,
            "rank": self.rank
        }


# utils
def normalize_text(text: str) -> str:
    """Normalize whitespace in text"""
    return re.sub(r'\s+', ' ', text.strip())


def safe_str(x) -> str:
    """Convert to string safely"""
    return "" if x is None else str(x).strip()


def read_jsonl(path: Path):
    """Read JSONL file line by line"""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line: {e}")
                continue


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range using min-max scaling"""
    if len(scores) == 0:
        return scores
    scores = np.array(scores, dtype=float)
    min_s, max_s = scores.min(), scores.max()
    if np.isclose(min_s, max_s):
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def build_embed_text(source: str, pasal: str, ayat: str, clean_text: str) -> str:
    """
    FIX (Cause 1): Build a structured embedding string that anchors the document
    to its identity metadata.

    Format: "{source} Pasal {pasal} Ayat {ayat}: {clean_text}"

    Why this works:
    - The sentence-transformer sees "UU 11/2008 Pasal 27 Ayat 1" as part of the
      embedding context for this chunk. When a query mentions "Pasal 27", cosine
      similarity is now naturally high for the Pasal 27 embedding and lower for
      Pasal 34 (even if Pasal 34's body text references "Pasal 27").
    - This separates *identity* (which pasal this is) from *content* (what it says),
      giving the model enough signal to discriminate between the correct article and
      cross-referencing penalty articles.

    Note: embed_text is ONLY used for building embeddings and BM25. The original
    clean_text is preserved in doc.text for display purposes.
    """
    ayat_part = f" Ayat {ayat}" if ayat else ""
    return f"{source} Pasal {pasal}{ayat_part}: {clean_text}"


def tokenize_for_bm25(text: str) -> List[str]:
    """
    FIX (Cause 2, partial): Tokenize text for BM25, removing Indonesian stopwords.

    v2 tokenized as: text.lower().split()
    Problem: High-frequency legal boilerplate ("yang", "dalam", "dengan",
    "sebagaimana", "dimaksud") dominates BM25 term frequency without adding
    discriminative value. Penalty articles (Pasal 34, 45A, 47) that cite
    "Pasal 27" in their body text would score almost as high as Pasal 27 itself
    for a query about Pasal 27.

    The fix removes stopwords so BM25 focuses on substantive terms like pasal
    numbers (embedded via embed_text prefix), action verbs, and legal nouns.
    """
    tokens = normalize_text(text).lower().split()
    return [t for t in tokens if t not in ID_STOPWORDS and len(t) > 1]


def extract_pasal_ayat_from_query(query: str) -> Tuple[str, str]:
    """
    Extract explicit pasal and ayat mentions from a query string.

    Returns:
        (pasal, ayat) — either may be empty string if not mentioned
    """
    pasal_match = re.search(r'pasal\s+([0-9]+[A-Za-z]?)', query, re.IGNORECASE)
    ayat_match = re.search(r'ayat\s+[\(\[]?([0-9]+)[\)\]]?', query, re.IGNORECASE)
    pasal = pasal_match.group(1) if pasal_match else ""
    ayat = ayat_match.group(1) if ayat_match else ""
    return pasal, ayat


def detect_query_intent(query: str) -> str:
    """
    FIX (Cause 3): Classify query intent to enable dual-intent decomposition.

    Three categories:
      "SPECIFIC_ARTICLE" — names a pasal explicitly, no sanction keyword
                           e.g. "Apa isi Pasal 27 ayat 1?"
                           Strategy: structural exact match boost only

      "DUAL_INTENT"      — names a pasal AND asks about its sanction
                           e.g. "Apa isi Pasal 27 dan sanksinya?"
                           Strategy: retrieve named pasal + look up its penalty
                           article separately, merge results

      "SEMANTIC"         — no explicit pasal reference; purely conceptual
                           e.g. "Apa itu pencemaran nama baik di UU ITE?"
                           Strategy: normal hybrid search, no special routing
    """
    q_lower = query.lower()
    has_pasal = bool(re.search(r'pasal\s+\d+', q_lower))
    has_sanction = any(kw in q_lower for kw in SANCTION_KEYWORDS)

    if has_pasal and has_sanction:
        return "DUAL_INTENT"
    elif has_pasal:
        return "SPECIFIC_ARTICLE"
    else:
        return "SEMANTIC"


# ============================================================================
# Sanctions Map Builder
# ============================================================================
def build_sanctions_map(documents: List[Document]) -> Dict[str, List[str]]:
    """
    Build a map from offense pasal → list of penalty pasal numbers.

    Penalty articles (Pasal 45, 45A, 46, 47, 48, 49, 50, 51, 52...) contain
    explicit references: "...sebagaimana dimaksud dalam Pasal 27..."
    We parse these cross-references to create a reverse lookup so that when a
    DUAL_INTENT query asks about Pasal 27 + sanksinya, we can directly retrieve
    the relevant penalty article without relying on semantic search.

    Returns:
        { "27": ["45", "45A"], "28": ["45A"], "30": ["46"], ... }
    """
    sanctions_map: Dict[str, List[str]] = defaultdict(list)

    penalty_pasal_pattern = re.compile(
        r'(?:pasal|ps\.?)\s+(\d+[A-Za-z]?)\s+(?:sampai|hingga|s\.?d\.?)\s+'
        r'(?:dengan\s+)?(?:pasal|ps\.?)\s+(\d+[A-Za-z]?)',
        re.IGNORECASE
    )
    single_ref_pattern = re.compile(
        r'(?:pasal|ps\.?)\s+(\d+[A-Za-z]?)',
        re.IGNORECASE
    )

    for doc in documents:
        if doc.doc_type != "PASAL":
            continue

        # Heuristic: penalty articles usually contain "dipidana" or "pidana penjara"
        if not any(kw in doc.text.lower() for kw in ["dipidana", "pidana penjara", "pidana kurungan"]):
            continue

        # Extract all pasal references in this penalty article
        # First handle ranges: "Pasal 27 sampai dengan Pasal 33"
        for m in penalty_pasal_pattern.finditer(doc.text):
            try:
                start = int(re.sub(r'[A-Za-z]', '', m.group(1)))
                end = int(re.sub(r'[A-Za-z]', '', m.group(2)))
                for n in range(start, end + 1):
                    sanctions_map[str(n)].append(doc.pasal)
            except ValueError:
                pass

        # Then handle individual references
        for m in single_ref_pattern.finditer(doc.text):
            ref_pasal = m.group(1)
            if ref_pasal != doc.pasal:  # don't map a pasal to itself
                sanctions_map[ref_pasal].append(doc.pasal)

    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in sanctions_map.items()}


# ============================================================================
# Main Retriever Class
# ============================================================================
class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 and semantic search.

    v3 changes vs v2:
    - Documents are embedded using structured identity prefix (Cause 1 fix)
    - BM25 tokenizer uses stopword removal (Cause 2 fix)
    - search() detects dual-intent queries and handles them separately (Cause 3 fix)
    - Structural Exact Match (SEM) boost applied BEFORE score normalization
    - sanctions_map built at index time for fast DUAL_INTENT lookup
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.documents: List[Document] = []

        # Indices
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.embedder: Optional[SentenceTransformer] = None

        # Lookup structures
        self.penjelasan_lookup: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.pasal_index: Dict[Tuple[str, str, str], int] = {}  # (source, pasal, ayat) → doc idx
        self.sanctions_map: Dict[str, List[str]] = {}           # offense pasal → penalty pasal list


    # ========================================================================
    # Building Index
    # ========================================================================

    def add_document(self, doc_dict: dict):
        """Add a document from dictionary"""
        raw_text = normalize_text(doc_dict.get("clean_text", ""))
        if not raw_text:
            return

        source = safe_str(doc_dict.get("source"))
        pasal = safe_str(doc_dict.get("pasal"))
        ayat = safe_str(doc_dict.get("ayat"))
        doc_type = doc_dict.get("type", "PASAL").upper()

        # FIX (Cause 1): Build structured embed_text with identity prefix.
        # This is the text that gets fed to the sentence transformer and BM25.
        # The original raw_text is preserved separately for display.
        embed_text = build_embed_text(source, pasal, ayat, raw_text)

        doc = Document(
            chunk_id=doc_dict.get("chunk_id", f"doc_{len(self.documents)}"),
            text=raw_text,
            embed_text=embed_text,
            source=source,
            pasal=pasal,
            ayat=ayat,
            doc_type=doc_type
        )

        idx = len(self.documents)
        self.documents.append(doc)

        if doc_type == "PENJELASAN":
            # FIX: Index by (source, pasal, ayat) triplet for ayat-level precision
            self.penjelasan_lookup[(source, pasal, ayat)].append(idx)
        elif doc_type == "PASAL":
            # Direct lookup for SEM boost and DUAL_INTENT routing
            self.pasal_index[(source, pasal, ayat)] = idx


    def load_from_jsonl(self, jsonl_paths: List[Path]):
        """Load documents from multiple JSONL files"""
        for path in jsonl_paths:
            if not path.exists():
                print(f"Warning: File not found: {path}")
                continue
            print(f"Loading {path}...")
            count = 0
            for doc_dict in read_jsonl(path):
                self.add_document(doc_dict)
                count += 1
            print(f"  Loaded {count} documents from {path.name}")
        print(f"Total documents loaded: {len(self.documents)}")


    def build_bm25_index(self):
        """
        Build BM25 index from documents.

        FIX (Cause 2): Uses embed_text (with identity prefix) and stopword-filtered
        tokenization instead of raw clean_text with naive split().
        """
        print("Building BM25 index (with stopword filtering)...")
        # Use embed_text so the pasal number appears in BM25 term frequency,
        # giving a lexical signal that reinforces semantic identity matching.
        tokenized = [tokenize_for_bm25(doc.embed_text) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built with {len(self.documents)} documents")


    def build_semantic_index(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """
        Build FAISS semantic index.

        FIX (Cause 1): Encodes embed_text instead of raw text.
        The identity prefix makes embeddings discriminative by pasal number.
        """
        print(f"Building semantic index with model: {self.model_name}")

        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)

        # FIX: Encode embed_text, NOT raw doc.text
        texts = [doc.embed_text for doc in self.documents]
        n_docs = len(texts)

        embeddings_list = []
        for i in tqdm(range(0, n_docs, batch_size), desc="Encoding documents"):
            batch = texts[i:i + batch_size]
            batch_emb = self.embedder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_emb)

        self.embeddings = np.vstack(embeddings_list).astype('float32')

        # Normalize for cosine similarity (inner product after L2 normalization)
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

        print(f"  Semantic index built: {self.embeddings.shape}")


    def build_sanctions_map(self):
        """
        FIX (Cause 3): Build offense → penalty article lookup at index time.
        Called during build_all_indices so it's ready for DUAL_INTENT queries.
        """
        print("Building sanctions map for dual-intent query routing...")
        self.sanctions_map = build_sanctions_map(self.documents)
        total_mapped = sum(len(v) for v in self.sanctions_map.values())
        print(f"  Sanctions map: {len(self.sanctions_map)} offense pasals → "
              f"{total_mapped} penalty article references")


    def build_all_indices(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """Build BM25, semantic, and sanctions indices"""
        if len(self.documents) == 0:
            raise ValueError("No documents loaded. Call load_from_jsonl() first.")
        self.build_bm25_index()
        self.build_semantic_index(batch_size=batch_size)
        self.build_sanctions_map()


    # ========================================================================
    # Retrieval — Core
    # ========================================================================

    def search_bm25(self, query: str) -> np.ndarray:
        """
        BM25 search. Returns score array over all documents.
        Query is tokenized with the same stopword-filtered tokenizer used at index time.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")
        tokens = tokenize_for_bm25(query)
        return np.array(self.bm25.get_scores(tokens), dtype=float)


    def search_semantic(self, query: str, top_k: int = 50) -> np.ndarray:
        """
        Semantic search via FAISS. Returns score array over all documents.
        Only the top_k results are populated; rest remain 0.0.
        """
        if self.faiss_index is None or self.embedder is None:
            raise RuntimeError("Semantic index not built")

        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)

        scores = np.zeros(len(self.documents), dtype=float)
        D, I = self.faiss_index.search(query_emb, min(top_k, len(self.documents)))
        for idx, sim in zip(I[0], D[0]):
            if 0 <= idx < len(scores):
                scores[idx] = sim
        return scores


    def _apply_sem_boost(
        self,
        scores: np.ndarray,
        query_pasal: str,
        query_ayat: str,
        sem_boost: float
    ) -> np.ndarray:
        """
        FIX (Cause 2): Structural Exact Match (SEM) boost applied BEFORE
        score normalization.

        v2 issue: pasal_boost defaulted to 0.0 and was applied multiplicatively
        AFTER normalization, making it effectively a no-op. Even when set to e.g.
        1.5, a normalized score of 0.8 would only become 1.2, while a wrong
        article with a raw 0.95 semantic score would still win after normalization.

        v3 fix: SEM boost is added BEFORE normalization as a raw additive offset.
        Matching documents get their pre-normalization scores raised by sem_boost,
        which shifts the entire score distribution favorably for exact matches
        before the 0-1 rescaling happens. This ensures that exact pasal matches
        reliably outrank semantically similar but wrong articles.

        sem_boost=2.0 (default): adds 2.0 to the raw BM25/semantic scores of
        exact matches before normalization, reliably bringing them to the top
        of the distribution.
        """
        if not query_pasal:
            return scores

        boosted = scores.copy()
        for i, doc in enumerate(self.documents):
            if doc.doc_type != "PASAL":
                continue
            if doc.pasal.lower() == query_pasal.lower():
                if not query_ayat or doc.ayat == query_ayat:
                    boosted[i] += sem_boost
        return boosted


    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        sem_boost: float = 2.0,
        query_pasal: str = "",
        query_ayat: str = "",
        return_pasal_only: bool = True
    ) -> List[SearchResult]:
        """
        Core hybrid search: BM25 + semantic + SEM boost.

        Args:
            query:            Search query string
            top_k:            Number of results to return
            bm25_weight:      Weight for BM25 (0-1)
            semantic_weight:  Weight for semantic (0-1)
            sem_boost:        Additive boost applied BEFORE normalization for
                              documents whose (pasal, ayat) exactly matches the
                              query. Set to 0.0 to disable. (default: 2.0)
            query_pasal:      Extracted pasal number from query (optional)
            query_ayat:       Extracted ayat number from query (optional)
            return_pasal_only: If True, exclude PENJELASAN from results

        Returns:
            List[SearchResult] sorted by score descending
        """
        bm25_raw = self.search_bm25(query)
        semantic_raw = self.search_semantic(query, top_k=max(top_k * 10, 50))

        # FIX (Cause 2): Apply SEM boost BEFORE normalization
        if query_pasal:
            bm25_raw = self._apply_sem_boost(bm25_raw, query_pasal, query_ayat, sem_boost)
            semantic_raw = self._apply_sem_boost(semantic_raw, query_pasal, query_ayat, sem_boost)

        # Normalize to 0-1 after boosting
        bm25_norm = normalize_scores(bm25_raw)
        semantic_norm = normalize_scores(semantic_raw)

        # Weighted fusion
        hybrid = (bm25_weight * bm25_norm) + (semantic_weight * semantic_norm)

        # Filter by type and threshold
        valid = [
            i for i, doc in enumerate(self.documents)
            if (not return_pasal_only or doc.doc_type == "PASAL")
            and hybrid[i] >= MIN_SCORE_THRESHOLD
        ]

        valid.sort(key=lambda i: hybrid[i], reverse=True)
        top_indices = valid[:top_k]

        return [
            SearchResult(document=self.documents[i], score=float(hybrid[i]), rank=r)
            for r, i in enumerate(top_indices, start=1)
        ]


    # ========================================================================
    # Retrieval — Intent-Aware Search (FIX: Cause 3)
    # ========================================================================

    def _lookup_penalty_articles(
        self,
        offense_pasal: str,
        top_k: int = 2
    ) -> List[SearchResult]:
        """
        FIX (Cause 3): For DUAL_INTENT queries, directly look up the penalty
        articles for a given offense pasal using the pre-built sanctions_map.

        This avoids relying on semantic search to find sanction articles, which
        was unreliable because semantic similarity alone couldn't distinguish
        "what does Pasal 27 say" from "what is the sanction for Pasal 27".

        Returns penalty articles ordered by their pasal number (lower = more
        specific/recent amendment takes precedence).
        """
        penalty_pasals = self.sanctions_map.get(offense_pasal, [])
        if not penalty_pasals:
            return []

        results = []
        seen_pasal = set()

        for penalty_pasal in penalty_pasals:
            if penalty_pasal in seen_pasal:
                continue
            seen_pasal.add(penalty_pasal)

            # Find document(s) for this penalty pasal
            for (src, p, a), idx in self.pasal_index.items():
                if p == penalty_pasal:
                    doc = self.documents[idx]
                    results.append(SearchResult(
                        document=doc,
                        score=0.85,   # Fixed confidence score for direct-lookup results
                        rank=len(results) + 1
                    ))

        return results[:top_k]


    def search(
        self,
        query: str,
        top_k_pasal: int = 3,
        top_k_penjelasan_per_pasal: int = 2,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        sem_boost: float = 2.0,
    ) -> Dict:
        """
        FIX (Cause 3): Main search interface with intent-aware routing.

        Query routing logic:
        ┌─────────────────────────────────────────────────────────────────┐
        │  SEMANTIC          → normal hybrid search, no special handling  │
        │  SPECIFIC_ARTICLE  → hybrid search + SEM boost for named pasal  │
        │  DUAL_INTENT       → two parallel searches merged:              │
        │                       (A) SPECIFIC_ARTICLE for the named pasal  │
        │                       (B) direct sanctions_map lookup for the   │
        │                           penalty article(s)                    │
        └─────────────────────────────────────────────────────────────────┘

        Returns:
            {
                "query": str,
                "query_intent": str,          # SEMANTIC | SPECIFIC_ARTICLE | DUAL_INTENT
                "pasal_results": List[SearchResult],
                "penjelasan_results": List[SearchResult]
            }
        """
        intent = detect_query_intent(query)
        query_pasal, query_ayat = extract_pasal_ayat_from_query(query)

        if intent == "DUAL_INTENT":
            # --- Sub-query A: retrieve the named pasal itself ---
            # Use a content-only sub-query (strip sanction keywords) so the
            # semantic model focuses on the pasal content, not the penalty.
            content_query = re.sub(
                r'\b(?:' + '|'.join(SANCTION_KEYWORDS) + r')\w*\b',
                '',
                query,
                flags=re.IGNORECASE
            ).strip()
            content_query = re.sub(r'\s+', ' ', content_query)

            pasal_results = self.search_hybrid(
                content_query,
                top_k=max(top_k_pasal - 1, 1),
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                sem_boost=sem_boost,
                query_pasal=query_pasal,
                query_ayat=query_ayat,
                return_pasal_only=True
            )

            # --- Sub-query B: directly look up penalty articles ---
            penalty_results = self._lookup_penalty_articles(query_pasal, top_k=1)

            # Merge A + B, deduplicate by (source, pasal, ayat), re-rank
            seen_ids = set()
            merged = []
            for res in pasal_results + penalty_results:
                key = (res.document.source, res.document.pasal, res.document.ayat)
                if key not in seen_ids:
                    seen_ids.add(key)
                    merged.append(res)

            # Re-assign ranks
            for i, res in enumerate(merged[:top_k_pasal], start=1):
                res.rank = i
            pasal_results = merged[:top_k_pasal]

        else:
            # SEMANTIC or SPECIFIC_ARTICLE: standard hybrid with SEM boost
            pasal_results = self.search_hybrid(
                query,
                top_k=top_k_pasal,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                sem_boost=sem_boost if intent == "SPECIFIC_ARTICLE" else 0.0,
                query_pasal=query_pasal,
                query_ayat=query_ayat,
                return_pasal_only=True
            )

        # Linked PENJELASAN (unchanged logic from v2, but with ayat-level precision)
        penjelasan_results = self.get_linked_penjelasan(
            pasal_results,
            max_per_pasal=top_k_penjelasan_per_pasal
        )

        return {
            "query": query,
            "query_intent": intent,
            "pasal_results": pasal_results,
            "penjelasan_results": penjelasan_results
        }


    # ========================================================================
    # PENJELASAN Linking
    # ========================================================================

    def get_linked_penjelasan(
        self,
        pasal_results: List[SearchResult],
        max_per_pasal: int = 3
    ) -> List[SearchResult]:
        """
        Retrieve PENJELASAN documents linked to retrieved PASAL results.

        FIX: Indexes by (source, pasal, ayat) triplet instead of just
        (source, pasal), improving ayat-level precision for explanations.
        Falls back to (source, pasal) if no ayat-specific entry found.
        """
        linked = []
        seen: Set[str] = set()

        for pasal_result in pasal_results:
            doc = pasal_result.document

            # Try ayat-specific lookup first, then fall back to pasal-level
            exact_key = (doc.source, doc.pasal, doc.ayat)
            pasal_key_indices = [
                idx for (src, p, a), idxs in self.penjelasan_lookup.items()
                if src == doc.source and p == doc.pasal
                for idx in idxs
            ]

            # Prioritize ayat-exact matches
            exact_indices = self.penjelasan_lookup.get(exact_key, [])
            other_indices = [i for i in pasal_key_indices if i not in exact_indices]
            selected = (exact_indices + other_indices)[:max_per_pasal]

            for idx in selected:
                penj_doc = self.documents[idx]
                uid = penj_doc.chunk_id
                if uid in seen:
                    continue
                seen.add(uid)
                linked.append(SearchResult(
                    document=penj_doc,
                    score=pasal_result.score,
                    rank=len(linked) + 1
                ))

        return linked


    # ========================================================================
    # Save / Load
    # ========================================================================

    def save(self, output_dir: Path):
        """Save index to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Documents (now includes embed_text field)
        with open(output_dir / "documents.jsonl", 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')

        # BM25
        with open(output_dir / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)

        # Embeddings
        if self.embeddings is not None:
            np.save(output_dir / "embeddings.npy", self.embeddings)

        # FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(output_dir / "faiss.index"))

        # PENJELASAN lookup (serialized as "source|pasal|ayat" → [idx])
        with open(output_dir / "penjelasan_lookup.json", 'w', encoding='utf-8') as f:
            lookup_json = {
                f"{src}|{pasal}|{ayat}": indices
                for (src, pasal, ayat), indices in self.penjelasan_lookup.items()
            }
            json.dump(lookup_json, f, ensure_ascii=False)

        # Pasal index (direct lookup)
        with open(output_dir / "pasal_index.json", 'w', encoding='utf-8') as f:
            pasal_index_json = {
                f"{src}|{pasal}|{ayat}": idx
                for (src, pasal, ayat), idx in self.pasal_index.items()
            }
            json.dump(pasal_index_json, f, ensure_ascii=False)

        # Sanctions map
        with open(output_dir / "sanctions_map.json", 'w', encoding='utf-8') as f:
            json.dump(self.sanctions_map, f, ensure_ascii=False, indent=2)

        # Metadata
        metadata = {
            "index_version": INDEX_VERSION,
            "model_name": self.model_name,
            "embedding_strategy": "structured_prefix",  # new in v3
            "bm25_tokenizer": "stopword_filtered",       # new in v3
            "n_documents": len(self.documents),
            "n_pasal": sum(1 for d in self.documents if d.doc_type == "PASAL"),
            "n_penjelasan": sum(1 for d in self.documents if d.doc_type == "PENJELASAN")
        }
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Index saved to: {output_dir}")


    @classmethod
    def load(cls, index_dir: Path, model_name: Optional[str] = None):
        """Load index from directory"""
        index_dir = Path(index_dir)

        with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if model_name is None:
            model_name = metadata.get("model_name", DEFAULT_MODEL)

        retriever = cls(model_name=model_name)

        # Load documents (embed_text preserved in saved JSONL)
        with open(index_dir / "documents.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                doc_dict = json.loads(line)
                doc = Document.from_dict(doc_dict)
                idx = len(retriever.documents)
                retriever.documents.append(doc)
                # Rebuild runtime lookups
                if doc.doc_type == "PENJELASAN":
                    retriever.penjelasan_lookup[(doc.source, doc.pasal, doc.ayat)].append(idx)
                elif doc.doc_type == "PASAL":
                    retriever.pasal_index[(doc.source, doc.pasal, doc.ayat)] = idx

        # Load BM25
        with open(index_dir / "bm25.pkl", 'rb') as f:
            retriever.bm25 = pickle.load(f)

        # Load embeddings
        emb_path = index_dir / "embeddings.npy"
        if emb_path.exists():
            retriever.embeddings = np.load(emb_path)

        # Load FAISS
        faiss_path = index_dir / "faiss.index"
        if faiss_path.exists():
            retriever.faiss_index = faiss.read_index(str(faiss_path))

        # Load sanctions map
        sanctions_path = index_dir / "sanctions_map.json"
        if sanctions_path.exists():
            with open(sanctions_path, 'r', encoding='utf-8') as f:
                retriever.sanctions_map = json.load(f)

        # Load embedder for query encoding
        retriever.embedder = SentenceTransformer(retriever.model_name)

        print(f"Index loaded from: {index_dir} (version: {metadata.get('index_version','?')})")
        print(f"  Documents : {len(retriever.documents)}")
        print(f"  PASAL     : {metadata.get('n_pasal','?')}")
        print(f"  PENJELASAN: {metadata.get('n_penjelasan','?')}")
        print(f"  Embedding : {metadata.get('embedding_strategy','raw_text')}")

        return retriever


# ============================================================================
# CLI Interface
# ============================================================================

def build_command(args):
    """Build index from JSONL files"""
    retriever = HybridRetriever(model_name=args.model)
    jsonl_paths = [Path(p) for p in args.jsonl]
    retriever.load_from_jsonl(jsonl_paths)
    retriever.build_all_indices(batch_size=args.batch_size)
    retriever.save(Path(args.outdir))


def query_command(args):
    """Interactive query mode"""
    retriever = HybridRetriever.load(Path(args.index_dir), model_name=args.model)

    print("\n" + "=" * 70)
    print("Indonesian ITE Law Hybrid Retrieval System  [v3]")
    print("=" * 70)
    print(f"Loaded {len(retriever.documents)} documents")
    print(f"Model: {retriever.model_name}")
    print("\nType your query (or 'quit' to exit)\n")

    try:
        while True:
            query = input("QUERY> ").strip()
            if not query:
                continue
            if query.lower() in ('quit', 'exit', 'q'):
                break

            results = retriever.search(
                query,
                top_k_pasal=args.top_k,
                top_k_penjelasan_per_pasal=args.penj_per_pasal,
                bm25_weight=args.bm25_weight,
                semantic_weight=args.semantic_weight,
                sem_boost=args.sem_boost,
            )

            intent = results.get("query_intent", "?")
            print(f"\n{'=' * 70}")
            print(f"📜 PASAL RESULTS  (intent: {intent}, top {args.top_k})")
            print(f"{'=' * 70}")

            if not results['pasal_results']:
                print("No PASAL found matching your query.")
            else:
                for res in results['pasal_results']:
                    doc = res.document
                    print(f"\n[{res.rank}] Score: {res.score:.4f}")
                    print(f"    {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    ID: {doc.chunk_id}")
                    print(f"    {doc.text[:300]}")

            print(f"\n{'=' * 70}")
            print("💡 PENJELASAN (Explanations)")
            print(f"{'=' * 70}")

            if not results['penjelasan_results']:
                print("No PENJELASAN found for the retrieved PASAL.")
            else:
                for res in results['penjelasan_results']:
                    doc = res.document
                    print(f"\n[{res.rank}] {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    {doc.text[:300]}")

            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval system for Indonesian ITE law — v3"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build index from JSONL files')
    build_parser.add_argument('--jsonl', nargs='+', required=True,
        help='Input JSONL files')
    build_parser.add_argument('--outdir', required=True,
        help='Output directory for index')
    build_parser.add_argument('--model', default=DEFAULT_MODEL,
        help=f'SentenceTransformer model (default: {DEFAULT_MODEL})')
    build_parser.add_argument('--batch-size', type=int, default=EMBEDDING_BATCH_SIZE,
        help='Batch size for embedding encoding')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query an existing index')
    query_parser.add_argument('--index-dir', required=True,
        help='Directory containing the built index')
    query_parser.add_argument('--model', default=None,
        help='Override SentenceTransformer model (optional)')
    query_parser.add_argument('--top-k', type=int, default=3,
        help='Number of PASAL results to return')
    query_parser.add_argument('--penj-per-pasal', type=int, default=2,
        help='Number of PENJELASAN per PASAL')
    query_parser.add_argument('--bm25-weight', type=float, default=0.3,
        help='Weight for BM25 score (default: 0.3)')
    query_parser.add_argument('--semantic-weight', type=float, default=0.7,
        help='Weight for semantic score (default: 0.7)')
    query_parser.add_argument('--sem-boost', type=float, default=2.0,
        help='Structural Exact Match boost applied before normalization (default: 2.0)')

    args = parser.parse_args()

    if args.command == 'build':
        build_command(args)
    elif args.command == 'query':
        query_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
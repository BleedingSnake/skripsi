#!/usr/bin/env python3
"""
intent_aware_retriever.py

Intent-aware retrieval system for Indonesian legal documents.
Detects user query intent and applies appropriate search strategies.

=============================================================================
CHANGELOG (rebased on build_index_hybrid_v3)
=============================================================================

Previously this module imported from build_index_hybrid_v2, which had a
broken SPECIFIC_ARTICLE retrieval pipeline (see v3 changelog for full details).
This update rebases on v3 and unifies the two overlapping intent systems:

  OLD state
  ---------
  - intent_aware_retriever.py owned a 7-class taxonomy:
      DEFINITION, PENALTY, PERMISSION, PROCEDURE, EXAMPLE,
      SPECIFIC_ARTICLE, GENERAL
  - build_index_hybrid_v3.py introduced a parallel 3-class structural taxonomy:
      SPECIFIC_ARTICLE, DUAL_INTENT, SEMANTIC
  - The two systems didn't talk to each other. A query classified as
    SPECIFIC_ARTICLE here would still go through the old v2 search_hybrid()
    path, bypassing v3's SEM boost and dual-intent decomposition entirely.

  WHAT CHANGED
  ------------
  1. Import source: build_index_hybrid_v2 → build_index_hybrid_v3
     IntentAwareRetriever now inherits the fixed HybridRetriever, which means
     SEM boost, stopword-filtered BM25, structured embeddings, and the
     sanctions_map are all available automatically.

  2. Unified intent taxonomy — one detect() call, full coverage:
     The v3 structural detector (SPECIFIC_ARTICLE / DUAL_INTENT / SEMANTIC)
     is now called FIRST inside detect(). If it fires, it takes precedence
     over the pattern-based checks. Pattern-based checks handle the remaining
     semantic intents (DEFINITION, PENALTY, PERMISSION, PROCEDURE, EXAMPLE)
     that v3 didn't cover. GENERAL is the catch-all.

     Priority order (highest → lowest):
       1. DUAL_INTENT        (v3 structural — explicit pasal + sanction keyword)
       2. SPECIFIC_ARTICLE   (v3 structural — explicit pasal, no sanction keyword)
       3. DEFINITION         (pattern-based)
       4. PENALTY            (pattern-based)
       5. PERMISSION         (pattern-based)
       6. PROCEDURE          (pattern-based)
       7. EXAMPLE            (pattern-based)
       8. GENERAL            (catch-all)

  3. search_with_intent() routes DUAL_INTENT and SPECIFIC_ARTICLE through
     v3's HybridRetriever.search() (which uses SEM boost + sanctions_map
     lookup), while all other intents continue through _search_with_strategy()
     with their existing per-intent weight/boost configs. This gives us the
     best of both systems.

  4. IntentAwareRetriever.load() now copies the three additional v3 state
     attributes (pasal_index, sanctions_map, embedder) from the base loader
     so the v3 routing paths work correctly.

  5. Output dict key normalised: "detected_intent" is always present and
     always holds the full unified label (DUAL_INTENT now surfaced, was
     previously invisible). The old "strategy_name" key is kept for
     backwards compatibility.

=============================================================================

Key Features:
- Unified 8-class intent detection (pattern + structural)
- Per-intent search strategies (weights, boosts, filters)
- DUAL_INTENT decomposition for "what does Pasal X say AND what's the sanction"
- Structural Exact Match (SEM) boost for SPECIFIC_ARTICLE queries
- PENJELASAN linking with ayat-level precision

Usage:
    from intent_aware_retriever import IntentAwareRetriever

    retriever = IntentAwareRetriever.load("./index_v3")
    results = retriever.search_with_intent("Apa itu pencemaran nama baik?")
    results = retriever.search_with_intent("Pasal 27 ayat 3 itu apa dan sanksinya?")
"""

import re
import itertools
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Rebase on v3 — this is the only import change from the previous version.
# All three fixes (structured embeddings, SEM boost, dual-intent routing)
# come along for free by switching this single import.
from build_index_hybrid_v3 import (
    HybridRetriever,
    SearchResult,
    detect_query_intent,    # v3 structural detector (SPECIFIC_ARTICLE / DUAL_INTENT / SEMANTIC)
    extract_pasal_ayat_from_query,
)


# ============================================================================
# Intent Detection Patterns (Comprehensive)
# ============================================================================

class IntentPatterns:
    """
    Comprehensive regex patterns for semantic intent detection.
    Covers formal, informal, and colloquial Indonesian queries.

    Note: SPECIFIC_ARTICLE and DUAL_INTENT are detected structurally by
    build_index_hybrid_v3.detect_query_intent() and take priority over
    these patterns. These patterns handle the remaining semantic intents.
    """

    # DEFINITION: User wants explanation/meaning of a concept
    DEFINITION = [
        r'\bapa itu\b',
        r'\bapa definisi\b',
        r'\bpengertian\b',
        r'\bapa yang dimaksud\b',
        r'\barti dari\b',
        r'\bmaksud dari\b',
        r'\bmaksudnya\b',
        r'\bapa artinya\b',
        r'\bapaan sih\b',
        r'\bjelaskan\b.*\bapa itu\b',
        r'\bjelasin\b',
        r'\bkasih tau\b.*\barti\b',
        r'\btahu gak\b.*\barti\b',
        r'\bmengartikan\b',
        r'\bdefinisi dari\b',
        r'\byang dimaksud dengan\b',
        r'\badalah\b.*\bapa\b',
        r'\byaitu\b.*\bapa\b',
    ]

    # PENALTY: User asks about punishment/sanctions (without explicit pasal ref)
    # Note: queries WITH an explicit pasal ref AND sanction keyword are caught
    # earlier as DUAL_INTENT by the structural detector.
    PENALTY = [
        r'\bhukuman\b',
        r'\bsanksi\b',
        r'\bdenda\b',
        r'\bpidana\b',
        r'\bancaman\b',
        r'\bberapa tahun\b',
        r'\bberapa lama\b',
        r'\blama penjara\b',
        r'\bkonsekuensi\b',
        r'\bakibat hukum\b',
        r'\brisiko hukum\b',
        r'\bkena denda\b',
        r'\bkena sanksi\b',
        r'\bkena hukuman\b',
        r'\bbisa dipenjara\b',
        r'\bbisa kena\b',
        r'\bkalo ketauan\b',
        r'\bberapa\b.*\bdenda\b',
        r'\bberapa\b.*\bhukuman\b',
        r'\bapa sanksinya\b',
        r'\bhukumannya apa\b',
    ]

    # PERMISSION: User asks if something is allowed or prohibited
    PERMISSION = [
        r'\bbolehkah\b',
        r'\bboleh\s+(?:gak|tidak|nggak|ga)\b',
        r'\bapakah boleh\b',
        r'\bbisakah\b',
        r'\bbisa\s+(?:gak|tidak|nggak|ga)\b',
        r'\bdilarang\b',
        r'\bdilarang\s+(?:gak|tidak|nggak|ga)\b',
        r'\bapakah dilarang\b',
        r'\bdiperbolehkan\b',
        r'\bapakah diperbolehkan\b',
        r'\bmelanggar\b',
        r'\bmelanggar\s+(?:gak|tidak|nggak|ga)\b',
        r'\bapakah melanggar\b',
        r'\btermasuk pelanggaran\b',
        r'\blegal\b',
        r'\blegal\s+(?:gak|tidak|nggak|ga)\b',
        r'\bilegal\b',
        # r'\bsah\b.*\bhukum\b',  # removed: too broad, fires on procedure queries about validity
        r'\bkalo\b.*\bboleh\b',
        r'\bkalo\b.*\bdilarang\b',
        r'\bngelakuin\b.*\bboleh\b',
        r'\bngelakuin\b.*\bmelanggar\b',
    ]

    # PROCEDURE: User asks how to do something
    PROCEDURE = [
        r'\bbagaimana cara\b',
        r'\bcara\b.*\b(?:untuk|agar|supaya)\b',
        r'\bprosedur\b',
        r'\btata cara\b',
        r'\blangkah[- ]langkah\b',
        r'\bproses\b',
        r'\bmekanisme\b',
        r'\bsyarat\b',
        r'\bpersyaratan\b',
        r'\bgimana\b.*\b(?:caranya|prosesnya)\b',
        r'\bmesti\b.*\bapa\b',
        r'\bharus\b.*\bapa\b',
        r'\bapa yang harus\b',
        r'\bmendaftar\b',
        r'\bmendaftarkan\b',
        r'\bpengajuan\b',
        r'\bpendaftaran\b',
    ]

    # EXAMPLE: User asks for examples or cases
    EXAMPLE = [
        r'\bcontoh\b',
        r'\bmisal\b',
        r'\bmisalnya\b',
        r'\bseperti\b.*\bcontoh\b',
        r'\bkasus\b',
        r'\bcontoh kasus\b',
        r'\bberikan contoh\b',
        r'\bkasih contoh\b',
        r'\bada contoh\b',
    ]


# ============================================================================
# Search Strategy Configuration
# ============================================================================

@dataclass
class SearchStrategy:
    """Configuration for intent-specific search behaviour"""
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    boost_penjelasan: float = 1.0
    boost_pasal: float = 1.0
    boost_phrases: Dict[str, float] = field(default_factory=dict)
    filter_pasal_range: Optional[Tuple[int, int]] = None
    filter_doc_type: Optional[str] = None
    min_score: float = 0.1


# Per-intent search strategy configs (unchanged from original — these are
# the well-tuned semantic strategies that v3 didn't attempt to replace)
STRATEGIES: Dict[str, SearchStrategy] = {
    "DEFINITION": SearchStrategy(
        bm25_weight=0.3,
        semantic_weight=0.7,
        boost_penjelasan=1.5,
        boost_pasal=1.2,
        boost_phrases={
            "yang dimaksud": 1.3,
            "adalah": 1.2,
            "yaitu": 1.2,
            "pengertian": 1.3,
            "definisi": 1.3,
        },
        min_score=0.2,
    ),
    "PENALTY": SearchStrategy(
        bm25_weight=0.6,       # raised: BM25 excels at exact penalty keywords
        semantic_weight=0.4,
        boost_penjelasan=0.8,  # penalty info lives in pasal body, not penjelasan
        boost_pasal=1.4,
        filter_pasal_range=None,  # removed: conduct pasals (27-44) also carry sanctions
        boost_phrases={
            "pidana": 1.4,
            "denda": 1.4,
            "penjara": 1.3,
            "hukuman": 1.3,
            "sanksi": 1.3,
            "dipidana": 1.4,
            "diancam": 1.3,
        },
        min_score=0.1,         # lowered: don't filter out low-scoring but correct chunks
    ),
    "PERMISSION": SearchStrategy(
        bm25_weight=0.3,
        semantic_weight=0.7,
        boost_pasal=1.2,
        boost_phrases={
            "dilarang": 1.3,
            "tidak boleh": 1.3,
            "melanggar": 1.2,
            "pelanggaran": 1.2,
        },
        min_score=0.15,
    ),
    "PROCEDURE": SearchStrategy(
        bm25_weight=0.5,       # raised: procedural keywords are distinctive
        semantic_weight=0.5,
        boost_penjelasan=0.8,  # procedures live in pasal body, not explanations
        boost_pasal=1.3,
        boost_phrases={
            "prosedur": 1.3,
            "tata cara": 1.4,
            "syarat": 1.2,
            "langkah": 1.2,
            "wajib": 1.2,
            "harus": 1.1,
        },
        min_score=0.1,
    ),
    "EXAMPLE": SearchStrategy(
        bm25_weight=0.2,
        semantic_weight=0.8,
        boost_penjelasan=1.4,
        boost_phrases={
            "contoh": 1.3,
            "misalnya": 1.2,
            "seperti": 1.2,
        },
        min_score=0.15,
    ),
    "GENERAL": SearchStrategy(
        bm25_weight=0.3,
        semantic_weight=0.7,
        min_score=0.1,
    ),
}


# ============================================================================
# Unified Intent Detector
# ============================================================================

class IntentDetector:
    """
    Detects user query intent using a two-tier approach:

    Tier 1 — Structural detection (v3):
      Uses explicit pasal number + sanction keyword signals.
      Returns DUAL_INTENT or SPECIFIC_ARTICLE when applicable.
      These are prioritised because they require specific retrieval routing
      (SEM boost, sanctions_map lookup) that pattern matching alone can't drive.

    Tier 2 — Pattern-based detection:
      Regex patterns for semantic intents: DEFINITION, PENALTY, PERMISSION,
      PROCEDURE, EXAMPLE. Falls through to GENERAL if nothing matches.
    """

    def __init__(self):
        self.patterns = IntentPatterns()

    def detect(self, query: str) -> str:
        """
        Detect intent from user query.

        Priority order:
          DUAL_INTENT > SPECIFIC_ARTICLE > DEFINITION > PENALTY >
          PERMISSION > PROCEDURE > EXAMPLE > GENERAL
        """
        # Tier 1: structural check (v3) 
        structural = detect_query_intent(query)
        if structural in ("DUAL_INTENT", "SPECIFIC_ARTICLE"):
            return structural

        # Tier 2: pattern-based semantic check
        q_lower = query.lower()
        semantic_checks = [
            ("DEFINITION",  self.patterns.DEFINITION),
            ("PROCEDURE",   self.patterns.PROCEDURE),   # before PENALTY/PERMISSION: procedural
            ("PENALTY",     self.patterns.PENALTY),     # signals (pidana, melanggar) are broad
            ("PERMISSION",  self.patterns.PERMISSION),  # and steal procedure queries otherwise
            ("EXAMPLE",     self.patterns.EXAMPLE),
        ]
        for intent_name, patterns in semantic_checks:
            if self._matches_any(q_lower, patterns):
                return intent_name

        return "GENERAL"

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        return any(re.search(p, text) for p in patterns)

    def get_matched_patterns(self, query: str, intent: str) -> List[str]:
        """Return which patterns fired for a given intent (debug helper)."""
        q_lower = query.lower()
        pattern_map = {
            "DEFINITION":  self.patterns.DEFINITION,
            "PENALTY":     self.patterns.PENALTY,
            "PERMISSION":  self.patterns.PERMISSION,
            "PROCEDURE":   self.patterns.PROCEDURE,
            "EXAMPLE":     self.patterns.EXAMPLE,
        }
        matched = []
        for p in pattern_map.get(intent, []):
            if re.search(p, q_lower):
                matched.append(p)
        return matched


# ============================================================================
# Intent-Aware Retriever
# ============================================================================

class IntentAwareRetriever(HybridRetriever):
    """
    Extends HybridRetriever (v3) with intent-aware search capabilities.

    Routing logic inside search_with_intent():

      DUAL_INTENT / SPECIFIC_ARTICLE
        → delegated to HybridRetriever.search() (v3 path)
          Uses SEM boost + sanctions_map lookup.
          These need structural precision, not just semantic tuning.

      DEFINITION / PENALTY / PERMISSION / PROCEDURE / EXAMPLE / GENERAL
        → handled by _search_with_strategy() (original path)
          Uses per-intent weight/boost/filter configs from STRATEGIES dict.
          These benefit from semantic tuning, not structural overrides.
    """

    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.intent_detector = IntentDetector()
        self._current_intent: str = "GENERAL"

    def search_with_intent(
        self,
        query: str,
        top_k_pasal: int = 3,
        top_k_penjelasan_per_pasal: int = 2,
        auto_detect: bool = True,
        force_intent: Optional[str] = None,
        debug: bool = False,
    ) -> Dict:
        """
        Main search entry point — detects intent and routes accordingly.

        Args:
            query:                    User's question
            top_k_pasal:              Number of PASAL results to return
            top_k_penjelasan_per_pasal: PENJELASAN results per PASAL
            auto_detect:              If False, forces GENERAL intent
            force_intent:             Override detected intent (for testing)
            debug:                    If True, include debug_info in output

        Returns:
            {
                "query":              str,
                "detected_intent":    str,   # unified 8-class label
                "strategy_name":      str,   # same as detected_intent (compat)
                "pasal_results":      List[SearchResult],
                "penjelasan_results": List[SearchResult],
                "debug_info":         dict   # only if debug=True
            }
        """
        # --- Intent resolution ---
        if force_intent:
            intent = force_intent
        elif auto_detect:
            intent = self.intent_detector.detect(query)
        else:
            intent = "GENERAL"

        self._current_intent = intent

        # --- Routing ---
        if intent in ("DUAL_INTENT", "SPECIFIC_ARTICLE"):
            # Delegate entirely to v3's routing which handles SEM boost and
            # sanctions_map lookup. We still respect top_k_pasal here.
            raw = super().search(
                query,
                top_k_pasal=top_k_pasal,
                top_k_penjelasan_per_pasal=top_k_penjelasan_per_pasal,
            )
            pasal_results = raw["pasal_results"]
            penjelasan_results = raw["penjelasan_results"]
        else:
            # Use per-intent strategy configs for semantic intents
            strategy = STRATEGIES.get(intent, STRATEGIES["GENERAL"])
            pasal_results = self._search_with_strategy(
                query, strategy, top_k=top_k_pasal
            )
            penjelasan_results = self.get_linked_penjelasan(
                pasal_results,
                max_per_pasal=top_k_penjelasan_per_pasal,
            )

        result = {
            "query":           query,
            "detected_intent": intent,
            "strategy_name":   intent,   # kept for backwards compatibility
            "pasal_results":   pasal_results,
            "penjelasan_results": penjelasan_results,
        }

        if debug:
            matched = self.intent_detector.get_matched_patterns(query, intent)
            q_pasal, q_ayat = extract_pasal_ayat_from_query(query)
            result["debug_info"] = {
                "matched_patterns":    matched,
                "intent_tier":         "structural" if intent in ("DUAL_INTENT", "SPECIFIC_ARTICLE") else "pattern",
                "extracted_pasal":     q_pasal,
                "extracted_ayat":      q_ayat,
                "total_docs_searched": len(self.documents),
                "pasal_count":         sum(1 for d in self.documents if d.doc_type == "PASAL"),
                "penjelasan_count":    sum(1 for d in self.documents if d.doc_type == "PENJELASAN"),
                "preprocessed_query":  self._preprocess_query(query) if intent == "DEFINITION" else query,
            }

        return result

    # ========================================================================
    # Strategy-based search (semantic intents — unchanged logic from original)
    # ========================================================================

    def _search_with_strategy(
        self,
        query: str,
        strategy: SearchStrategy,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Execute hybrid search with per-intent strategy config.
        Used for: DEFINITION, PENALTY, PERMISSION, PROCEDURE, EXAMPLE, GENERAL.
        """
        processed_query = self._preprocess_query(query)

        # Retrieve broadly — PENALTY/PROCEDURE chunks rank low before intent boosts.
        # top_k*5 (=15) was too narrow; floor at 50 ensures the right chunk is seen.
        results = self.search_hybrid(
            processed_query,
            top_k=max(top_k * 20, 50),  # wide net: right chunk may rank low before intent boosts
            bm25_weight=strategy.bm25_weight,
            semantic_weight=strategy.semantic_weight,
            return_pasal_only=True,
        )

        # Apply document-level boosts
        for result in results:
            doc = result.document
            doc_text_lower = doc.text.lower()

            if doc.doc_type == "PENJELASAN":
                result.score *= strategy.boost_penjelasan
            elif doc.doc_type == "PASAL":
                result.score *= strategy.boost_pasal

            for phrase, boost in strategy.boost_phrases.items():
                if phrase in doc_text_lower:
                    result.score *= boost

            # Proximity boost for multi-word queries
            query_words = processed_query.lower().split()
            if len(query_words) >= 2:
                if self._contains_phrase_proximity(doc_text_lower, query_words, max_distance=5):
                    result.score *= 1.2  # was 1.8 — over-boosted common words causing regressions

        # Filter by pasal range (e.g. PENALTY only looks at pasal 45–52)
        if strategy.filter_pasal_range:
            min_p, max_p = strategy.filter_pasal_range
            filtered = []
            for result in results:
                try:
                    n = int(re.search(r'\d+', result.document.pasal).group())
                    if min_p <= n <= max_p:
                        filtered.append(result)
                except (AttributeError, ValueError):
                    filtered.append(result)
            results = filtered

        # Aggressive relevance filter for DEFINITION queries
        if self._current_intent == "DEFINITION":
            results = self._filter_definition_relevance(results, processed_query)

        # Score threshold — never return empty. Filter by min_score only if
        # enough results survive; otherwise fall back to top-k by raw score.
        # This prevents the GOT:[] failure where all candidates are cut.
        above = [r for r in results if r.score >= strategy.min_score]
        results = above if len(above) >= top_k else results

        # Re-sort after boosts
        results.sort(key=lambda r: r.score, reverse=True)

        # Re-assign ranks
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank

        return results[:top_k]

    def _preprocess_query(self, query: str) -> str:
        """Strip question-framing words for DEFINITION queries."""
        if self._current_intent == "DEFINITION":
            question_words = [
                r'\bapa\s+itu\b', r'\bapa\s+definisi\b', r'\bpengertian\b',
                r'\bjelaskan\b', r'\bjelasin\b', r'\bkasih\s+tau\b',
                r'\byang\s+dimaksud\s+dengan\b', r'\barti\s+dari\b',
                r'\bmaksud\s+dari\b',
            ]
            processed = query
            for pattern in question_words:
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
            processed = re.sub(r'\s+', ' ', processed).strip()
            return processed if len(processed) >= 3 else query
        return query

    def _contains_phrase_proximity(
        self, text: str, query_words: List[str], max_distance: int = 5
    ) -> bool:
        """Return True if all query_words appear within max_distance of each other."""
        text_words = text.split()
        positions = {w: [] for w in query_words}
        for i, tw in enumerate(text_words):
            for qw in query_words:
                if qw in tw:
                    positions[qw].append(i)
        if any(len(p) == 0 for p in positions.values()):
            return False
        for combo in itertools.product(*positions.values()):
            if max(combo) - min(combo) <= max_distance:
                return True
        return False

    def _filter_definition_relevance(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """
        For DEFINITION queries, remove results that contain query words but
        not the actual concept being defined.
        """
        q_lower = query.lower()
        q_words = q_lower.split()
        if len(q_words) <= 1:
            return results

        filtered = []
        for result in results:
            doc_lower = result.document.text.lower()
            if q_lower in doc_lower:
                filtered.append(result)
                continue
            if self._contains_phrase_proximity(doc_lower, q_words, max_distance=3):
                filtered.append(result)
                continue
            if sum(1 for w in q_words if w in doc_lower) >= len(q_words):
                result.score *= 0.7
                filtered.append(result)

        return filtered if filtered else results

    # ========================================================================
    # Load
    # ========================================================================

    @classmethod
    def load(cls, index_dir: Path, model_name: Optional[str] = None):
        """
        Load v3 index and return an IntentAwareRetriever.

        Copies all v3 state attributes (including pasal_index and
        sanctions_map which are needed for DUAL_INTENT routing) from the
        base HybridRetriever loader.
        """
        base = HybridRetriever.load(index_dir, model_name)

        retriever = cls(model_name=base.model_name)

        # Core v2-compatible state
        retriever.documents        = base.documents
        retriever.bm25             = base.bm25
        retriever.embeddings       = base.embeddings
        retriever.faiss_index      = base.faiss_index
        retriever.embedder         = base.embedder
        retriever.penjelasan_lookup = base.penjelasan_lookup

        # v3 additions — required for DUAL_INTENT and SEM boost routing
        retriever.pasal_index      = base.pasal_index
        retriever.sanctions_map    = base.sanctions_map

        return retriever


# ============================================================================
# CLI for Testing
# ============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python intent_aware_retriever.py <index_dir>")
        print("\nOr test intent detection:")
        print("  python intent_aware_retriever.py test")
        sys.exit(1)

    if sys.argv[1] == "test":
        detector = IntentDetector()
        test_queries = [
            "Apa itu nama domain?",
            "Jelaskan pengertian pencemaran nama baik",
            "Berapa denda untuk Pasal 27?",
            "Bolehkah saya posting foto orang tanpa izin?",
            "Bagaimana cara mendaftar sertifikat elektronik?",
            "Kasih contoh pelanggaran Pasal 28",
            "Pasal 27 ayat 3 tentang apa?",
            "Kalo posting fitnah bisa kena hukuman berapa tahun?",
            "Apa isi Pasal 27 ayat 3 dan sanksinya?",       # DUAL_INTENT
            "Apa hukuman pelanggaran Pasal 31 ayat 1?",     # DUAL_INTENT
        ]
        print("=" * 70)
        print("INTENT DETECTION TEST (unified v3 + pattern)")
        print("=" * 70)
        for query in test_queries:
            intent = detector.detect(query)
            patterns = detector.get_matched_patterns(query, intent)
            tier = "structural" if intent in ("DUAL_INTENT", "SPECIFIC_ARTICLE") else "pattern"
            print(f"\nQuery  : {query}")
            print(f"Intent : {intent}  [{tier}]")
            if patterns:
                print(f"Matched: {patterns[:2]}")
        sys.exit(0)

    index_dir = Path(sys.argv[1])
    retriever = IntentAwareRetriever.load(index_dir)

    print("\n" + "=" * 70)
    print("Intent-Aware Retrieval System  [v3 backend]")
    print("=" * 70)
    print(f"Loaded {len(retriever.documents)} documents")
    print("\nType your query (or 'quit' to exit)")
    print("Append '--debug' to see intent detection details\n")

    try:
        while True:
            query_input = input("QUERY> ").strip()
            if not query_input:
                continue
            if query_input.lower() in ('quit', 'exit', 'q'):
                break

            debug = '--debug' in query_input
            if debug:
                query_input = query_input.replace('--debug', '').strip()

            results = retriever.search_with_intent(query_input, debug=debug)

            print(f"\n{'=' * 70}")
            print(f"🎯 DETECTED INTENT: {results['detected_intent']}")
            if debug and 'debug_info' in results:
                di = results['debug_info']
                print(f"   Tier    : {di['intent_tier']}")
                print(f"   Pasal   : {di['extracted_pasal'] or '—'}  "
                      f"Ayat: {di['extracted_ayat'] or '—'}")
                if di['matched_patterns']:
                    print(f"   Patterns: {di['matched_patterns'][:2]}")
            print(f"{'=' * 70}")

            print(f"\n📜 PASAL RESULTS")
            print(f"{'-' * 70}")
            if not results['pasal_results']:
                print("No PASAL found.")
            else:
                for res in results['pasal_results']:
                    doc = res.document
                    print(f"\n[{res.rank}] Score: {res.score:.4f}")
                    print(f"    {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    {doc.text[:250]}...")

            print(f"\n💡 PENJELASAN")
            print(f"{'-' * 70}")
            if not results['penjelasan_results']:
                print("No PENJELASAN found.")
            else:
                for res in results['penjelasan_results']:
                    doc = res.document
                    print(f"\n[{res.rank}] {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    {doc.text[:250]}...")
            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
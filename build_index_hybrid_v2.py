#!/usr/bin/env python3
"""
build_index_hybrid_v2.py

Clean hybrid index builder + retriever for Indonesian legal documents.

Key Features:
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
  python build_index_hybrid_v2.py build \
    --jsonl pasal_master.jsonl penjelasan_master.jsonl \
    --outdir ./index_v2

  # Query interactively
  python build_index_hybrid_v2.py query --index-dir ./index

  # Query programmatically
  from build_index_hybrid_v2 import HybridRetriever
  retriever = HybridRetriever.load("./index")
  results = retriever.search("Apa itu pencemaran nama baik?", top_k=3)
"""

import os
import json
import argparse
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_MODEL = "sentence-transformers/LaBSE"
EMBEDDING_BATCH_SIZE = 64
MIN_SCORE_THRESHOLD = 0.01  # Filter out very low scores


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class Document:
    """Represents a single legal document chunk"""
    chunk_id: str
    text: str
    source: str
    pasal: str
    ayat: str
    doc_type: str  # "PASAL" or "PENJELASAN"
    
    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "pasal": self.pasal,
            "ayat": self.ayat,
            "type": self.doc_type
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            chunk_id=d.get("chunk_id", ""),
            text=d.get("text", ""),
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


# ============================================================================
# Utilities
# ============================================================================
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


def extract_pasal_ayat_from_query(query: str) -> Tuple[str, str]:
    """
    Extract pasal and ayat mentions from query.
    
    Examples:
        "Pasal 27 ayat 3" -> ("27", "3")
        "pasal 27A" -> ("27A", "")
        "ayat 2" -> ("", "2")
    """
    pasal_match = re.search(r'pasal\s+([0-9]+[A-Za-z]?)', query, re.IGNORECASE)
    ayat_match = re.search(r'ayat\s+\(?([0-9]+)\)?', query, re.IGNORECASE)
    
    pasal = pasal_match.group(1) if pasal_match else ""
    ayat = ayat_match.group(1) if ayat_match else ""
    
    return pasal, ayat


# ============================================================================
# Main Retriever Class
# ============================================================================
class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 and semantic search.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.documents: List[Document] = []
        
        # Indices
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.embedder: Optional[SentenceTransformer] = None
        
        # Lookup structures for efficient PENJELASAN retrieval
        self.penjelasan_lookup: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        
    
    # ========================================================================
    # Building Index
    # ========================================================================
    
    def add_document(self, doc_dict: dict):
        """Add a document from dictionary"""
        # Validate required fields
        text = normalize_text(doc_dict.get("clean_text", ""))
        if not text:
            return  # Skip empty documents
        
        # Create document object
        doc = Document(
            chunk_id=doc_dict.get("chunk_id", f"doc_{len(self.documents)}"),
            text=text,
            source=safe_str(doc_dict.get("source")),
            pasal=safe_str(doc_dict.get("pasal")),
            ayat=safe_str(doc_dict.get("ayat")),
            doc_type=doc_dict.get("type", "PASAL").upper()
        )
        
        idx = len(self.documents)
        self.documents.append(doc)
        
        # Build PENJELASAN lookup for efficient retrieval
        if doc.doc_type == "PENJELASAN":
            key = (doc.source, doc.pasal)
            self.penjelasan_lookup[key].append(idx)
    
    
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
        """Build BM25 index from documents"""
        print("Building BM25 index...")
        
        # simple tokenization with lowercase + split on whitespace
        tokenized = [doc.text.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"  BM25 index built with {len(self.documents)} documents")
    
    
    def build_semantic_index(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """Build FAISS semantic index"""
        print(f"Building semantic index with model: {self.model_name}")
        
        # Load embedder
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)
        
        # Encode all documents
        texts = [doc.text for doc in self.documents]
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
        
        # Normalize for cosine similarity (using inner product)
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index (IndexFlatIP for exact search with inner product)
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)
        
        print(f"  Semantic index built: {self.embeddings.shape}")
    
    
    def build_all_indices(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """Build both BM25 and semantic indices"""
        if len(self.documents) == 0:
            raise ValueError("No documents loaded. Call load_from_jsonl() first.")
        
        self.build_bm25_index()
        self.build_semantic_index(batch_size=batch_size)
    
    
    # ========================================================================
    # Retrieval
    # ========================================================================
    
    def search_bm25(self, query: str, top_k: int = 50) -> np.ndarray:
        """
        Search using BM25.
        Returns: array of scores for all documents (length = n_docs)
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")
        
        query_tokens = normalize_text(query).lower().split()
        scores = self.bm25.get_scores(query_tokens)
        return np.array(scores, dtype=float)
    
    
    def search_semantic(self, query: str, top_k: int = 50) -> np.ndarray:
        """
        Search using FAISS semantic index.
        Returns: array of scores for all documents (length = n_docs)
        """
        if self.faiss_index is None or self.embedder is None:
            raise RuntimeError("Semantic index not built")
        
        # Encode query
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        query_emb = query_emb.astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search FAISS
        scores = np.zeros(len(self.documents), dtype=float)
        D, I = self.faiss_index.search(query_emb, min(top_k, len(self.documents)))
        
        # Fill scores array
        for idx, sim in zip(I[0], D[0]):
            if idx >= 0 and idx < len(scores):
                scores[idx] = sim
        
        return scores
    
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        pasal_boost: float = 0.0,
        ayat_boost: float = 0.0,
        return_pasal_only: bool = True
    ) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and semantic scores.
        
        Args:
            query: Search query
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
            pasal_boost: Multiplicative boost when pasal matches (e.g., 1.5 = 50% boost)
            ayat_boost: Multiplicative boost when ayat matches
            return_pasal_only: If True, only return PASAL documents (exclude PENJELASAN)
        
        Returns:
            List of SearchResult objects, sorted by score descending
        """
        # Get scores from both methods
        bm25_scores = self.search_bm25(query)
        semantic_scores = self.search_semantic(query)
        
        # Normalize both to 0-1 range
        bm25_norm = normalize_scores(bm25_scores)
        semantic_norm = normalize_scores(semantic_scores)
        
        # Hybrid fusion
        hybrid_scores = (bm25_weight * bm25_norm) + (semantic_weight * semantic_norm)
        
        # Apply query-based boosts (multiplicative, not additive)
        query_pasal, query_ayat = extract_pasal_ayat_from_query(query)
        
        if query_pasal or query_ayat:
            for i, doc in enumerate(self.documents):
                boost_factor = 1.0
                
                # Pasal match boost
                if query_pasal and doc.pasal.lower() == query_pasal.lower():
                    boost_factor *= (1.0 + pasal_boost)
                
                # Ayat match boost (only if pasal also matches)
                if query_ayat and doc.ayat.lower() == query_ayat.lower():
                    if not query_pasal or doc.pasal.lower() == query_pasal.lower():
                        boost_factor *= (1.0 + ayat_boost)
                
                hybrid_scores[i] *= boost_factor
        
        # Filter by type if needed
        valid_indices = []
        for i, doc in enumerate(self.documents):
            if return_pasal_only and doc.doc_type != "PASAL":
                continue
            if hybrid_scores[i] < MIN_SCORE_THRESHOLD:
                continue
            valid_indices.append(i)
        
        # Sort by score descending
        valid_indices.sort(key=lambda i: hybrid_scores[i], reverse=True)
        
        # Take top-k
        top_indices = valid_indices[:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(hybrid_scores[idx]),
                rank=rank
            ))
        
        return results
    
    
    def get_linked_penjelasan(
        self,
        pasal_results: List[SearchResult],
        max_per_pasal: int = 3
    ) -> List[SearchResult]:
        """
        Get PENJELASAN documents linked to retrieved PASAL results.
        
        Args:
            pasal_results: List of PASAL search results
            max_per_pasal: Maximum number of PENJELASAN to return per PASAL
        
        Returns:
            List of PENJELASAN documents linked to the pasal results
        """
        linked = []
        seen = set()
        
        for pasal_result in pasal_results:
            doc = pasal_result.document
            key = (doc.source, doc.pasal)
            
            # Get all PENJELASAN for this (source, pasal)
            penjelasan_indices = self.penjelasan_lookup.get(key, [])
            
            # Take up to max_per_pasal, prioritizing those with matching ayat
            matching_ayat = []
            other = []
            
            for idx in penjelasan_indices:
                penj_doc = self.documents[idx]
                
                # Skip if already seen
                chunk_key = (penj_doc.source, penj_doc.pasal, penj_doc.ayat, penj_doc.chunk_id)
                if chunk_key in seen:
                    continue
                seen.add(chunk_key)
                
                # Prioritize matching ayat
                if penj_doc.ayat == doc.ayat:
                    matching_ayat.append(idx)
                else:
                    other.append(idx)
            
            # Take matching ayat first, then others
            selected_indices = (matching_ayat + other)[:max_per_pasal]
            
            for idx in selected_indices:
                linked.append(SearchResult(
                    document=self.documents[idx],
                    score=pasal_result.score,  # Inherit score from linked pasal
                    rank=len(linked) + 1
                ))
        
        return linked
    
    
    def search(
        self,
        query: str,
        top_k_pasal: int = 3,
        top_k_penjelasan_per_pasal: int = 2,
        **search_kwargs
    ) -> Dict:
        """
        Main search interface - returns both PASAL and linked PENJELASAN.
        
        Returns:
            {
                "query": str,
                "pasal_results": List[SearchResult],
                "penjelasan_results": List[SearchResult]
            }
        """
        # Search for PASAL
        pasal_results = self.search_hybrid(
            query,
            top_k=top_k_pasal,
            return_pasal_only=True,
            **search_kwargs
        )
        
        # Get linked PENJELASAN
        penjelasan_results = self.get_linked_penjelasan(
            pasal_results,
            max_per_pasal=top_k_penjelasan_per_pasal
        )
        
        return {
            "query": query,
            "pasal_results": pasal_results,
            "penjelasan_results": penjelasan_results
        }
    
    
    # ========================================================================
    # Save/Load
    # ========================================================================
    
    def save(self, output_dir: Path):
        """Save index to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        with open(output_dir / "documents.jsonl", 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        
        # Save BM25
        with open(output_dir / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(output_dir / "embeddings.npy", self.embeddings)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(output_dir / "faiss.index"))
        
        # Save PENJELASAN lookup
        with open(output_dir / "penjelasan_lookup.json", 'w', encoding='utf-8') as f:
            # Convert tuple keys to strings for JSON
            lookup_json = {
                f"{src}|{pasal}": indices
                for (src, pasal), indices in self.penjelasan_lookup.items()
            }
            json.dump(lookup_json, f, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
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
        
        # Load metadata
        with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Use saved model name if not specified
        if model_name is None:
            model_name = metadata.get("model_name", DEFAULT_MODEL)
        
        # Create instance
        retriever = cls(model_name=model_name)
        
        # Load documents
        with open(index_dir / "documents.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                doc_dict = json.loads(line)
                retriever.documents.append(Document.from_dict(doc_dict))
        
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
        
        # Load PENJELASAN lookup
        lookup_path = index_dir / "penjelasan_lookup.json"
        if lookup_path.exists():
            with open(lookup_path, 'r', encoding='utf-8') as f:
                lookup_json = json.load(f)
                # Convert string keys back to tuples
                for key_str, indices in lookup_json.items():
                    src, pasal = key_str.split('|', 1)
                    retriever.penjelasan_lookup[(src, pasal)] = indices
        
        # Load embedder for queries
        retriever.embedder = SentenceTransformer(retriever.model_name)
        
        print(f"Index loaded from: {index_dir}")
        print(f"  Documents: {len(retriever.documents)}")
        print(f"  PASAL: {metadata.get('n_pasal', 'unknown')}")
        print(f"  PENJELASAN: {metadata.get('n_penjelasan', 'unknown')}")
        
        return retriever


# ============================================================================
# CLI Interface
# ============================================================================

def build_command(args):
    """Build index from JSONL files"""
    retriever = HybridRetriever(model_name=args.model)
    
    # Load documents
    jsonl_paths = [Path(p) for p in args.jsonl]
    retriever.load_from_jsonl(jsonl_paths)
    
    # Build indices
    retriever.build_all_indices(batch_size=args.batch_size)
    
    # Save
    retriever.save(Path(args.outdir))


def query_command(args):
    """Interactive query mode"""
    # Load index
    retriever = HybridRetriever.load(Path(args.index_dir), model_name=args.model)
    
    print("\n" + "="*70)
    print("Indonesian ITE Law Hybrid Retrieval System")
    print("="*70)
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
            
            # Search
            results = retriever.search(
                query,
                top_k_pasal=args.top_k,
                top_k_penjelasan_per_pasal=args.penj_per_pasal,
                bm25_weight=args.bm25_weight,
                semantic_weight=args.semantic_weight,
                pasal_boost=args.pasal_boost,
                ayat_boost=args.ayat_boost
            )
            
            # Display PASAL results
            print(f"\n{'='*70}")
            print(f"📜 PASAL RESULTS (Top {args.top_k})")
            print(f"{'='*70}")
            
            if not results['pasal_results']:
                print("No PASAL found matching your query.")
            else:
                for res in results['pasal_results']:
                    doc = res.document
                    print(f"\n[{res.rank}] Score: {res.score:.4f}")
                    print(f"    {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    ID: {doc.chunk_id}")
                    print(f"    Text:")
                    print(f"    {doc.text}")
                    print()
            
            # Display PENJELASAN results
            print(f"\n{'='*70}")
            print(f"💡 PENJELASAN (Explanations)")
            print(f"{'='*70}")
            
            if not results['penjelasan_results']:
                print("No PENJELASAN found for the retrieved PASAL.")
            else:
                for res in results['penjelasan_results']:
                    doc = res.document
                    print(f"\n[{res.rank}]")
                    print(f"    {doc.source} | Pasal {doc.pasal} Ayat {doc.ayat}")
                    print(f"    ID: {doc.chunk_id}")
                    print(f"    Text:")
                    print(f"    {doc.text}")
                    print()
            
            print()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval system for Indonesian ITE law"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build index from JSONL files')
    build_parser.add_argument(
        '--jsonl',
        nargs='+',
        required=True,
        help='Input JSONL files (e.g., pasal_master.jsonl penjelasan_master.jsonl)'
    )
    build_parser.add_argument(
        '--outdir',
        required=True,
        help='Output directory for index'
    )
    build_parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'SentenceTransformer model name (default: {DEFAULT_MODEL})'
    )
    build_parser.add_argument(
        '--batch-size',
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help='Batch size for embedding encoding'
    )

    # Query command
    query_parser = subparsers.add_parser('query', help='Query an existing index')
    query_parser.add_argument(
        '--index-dir',
        required=True,
        help='Directory containing the built index'
    )
    query_parser.add_argument(
        '--model',
        default=None,
        help='Override SentenceTransformer model (optional)'
    )
    query_parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of PASAL results to return'
    )
    query_parser.add_argument(
        '--penj-per-pasal',
        type=int,
        default=2,
        help='Number of PENJELASAN per PASAL'
    )
    query_parser.add_argument(
        '--bm25-weight',
        type=float,
        default=0.5,
        help='Weight for BM25 score'
    )
    query_parser.add_argument(
        '--semantic-weight',
        type=float,
        default=0.5,
        help='Weight for semantic score'
    )
    query_parser.add_argument(
        '--pasal-boost',
        type=float,
        default=0.0,
        help='Boost factor for matching pasal'
    )
    query_parser.add_argument(
        '--ayat-boost',
        type=float,
        default=0.0,
        help='Boost factor for matching ayat'
    )

    args = parser.parse_args()

    if args.command == 'build':
        build_command(args)
    elif args.command == 'query':
        query_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


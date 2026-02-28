"""
engine/relationship.py
----------------------
Production Hybrid SKU relationship classifier.
Combines HuggingFace Embeddings (60%) and Sequence Matching (40%).
Includes Category Guard and Complement Keyword overrides.
"""

from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config.settings import get_embedding_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds tuned for SKU naming conventions
VARIANT_THRESHOLD = 0.55
SUBSTITUTE_THRESHOLD = 0.30

# Signal weights: 40% literal overlap, 60% semantic meaning
_WEIGHTS = {"sequence": 0.40, "embedding": 0.60}

# Keywords that explicitly define a Complement relationship (Halo effect)
COMPLEMENT_KEYWORDS = frozenset({
    "mixed", "variety", "bundle", "combo", "sampler", "assorted"
})

SIZE_TOKENS = frozenset({
    "8oz", "12oz", "16oz", "24oz", "32oz", "64oz",
    "8", "12", "16", "24", "32", "64",
    "small", "medium", "large", "xl", "mini",
    "6pack", "12pack", "24pack", "pack", "ct", "count"
})

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic & Utilities
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Initialize the HuggingFace model once and cache in memory."""
    model_name = get_embedding_model()
    logger.info(f"Initializing Embedding Model: {model_name}")
    return SentenceTransformer(model_name)

def _get_embeddings(texts: list[str]) -> np.ndarray:
    """Generate L2-normalized embeddings for cosine similarity."""
    model = _get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2 Normalization ensures dot product equals cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)

def _normalize(text: str) -> str:
    """Standardize text for comparison."""
    return re.sub(r"[^\w\s]", " ", (str(text) or "").lower()).strip()

def _get_size_tokens(text: str) -> set[str]:
    """Identify size-related keywords."""
    tokens = _normalize(text).split()
    return {t for t in tokens if t in SIZE_TOKENS}

def _sequence_sim(a: str, b: str) -> float:
    """Literal character-level similarity using SequenceMatcher."""
    na, nb = _normalize(a), _normalize(b)
    return SequenceMatcher(None, na, nb).ratio() if na and nb else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Classification Logic
# ─────────────────────────────────────────────────────────────────────────────

def _classify_pair(name_a, name_b, cat_a, cat_b, emb_score):
    # 1. Category Guard: Force string conversion and strip whitespace
    if str(cat_a).strip().lower() != str(cat_b).strip().lower():
        return "unrelated"

    # 2. Complement Check: Priority keywords
    name_b_norm = _normalize(name_b)
    if any(kw in name_b_norm for kw in COMPLEMENT_KEYWORDS):
        return "complement"

    # 3. Blended Score Calculation
    seq_s = _sequence_sim(name_a, name_b)
    score = (_WEIGHTS["sequence"] * seq_s) + (_WEIGHTS["embedding"] * emb_score)
    
    # 4. Root Name Extraction (The key to your variant vs substitute fix)
    size_a = _get_size_tokens(name_a)
    size_b = _get_size_tokens(name_b)
    
    def get_root(name, sizes):
        tokens = _normalize(name).split()
        return " ".join([t for t in tokens if t not in sizes])

    root_a = get_root(name_a, size_a)
    root_b = get_root(name_b, size_b)

    # 5. Final Classification Logic
    if score >= SUBSTITUTE_THRESHOLD:
        # Same Root (e.g., 'salted pistachios') but different size = Variant
        if root_a == root_b and size_a != size_b:
            return "variant"
        
        # Different Root but same Category = Substitute
        return "substitute"
    
    return "unrelated"
# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_relationships(catalog: list[dict], promoted_id: str) -> list[str]:
    """
    Classify every SKU in catalog relative to the promoted SKU.
    Used for real-time dashboard updates.
    """
    if not catalog:
        return []

    names = [str(s.get("name", s.get("id", ""))) for s in catalog]
    cats  = [str(s.get("category", "Unknown")) for s in catalog]
    ids   = [str(s.get("id", s.get("sku", ""))) for s in catalog]
    
    try:
        p_idx = ids.index(str(promoted_id))
    except ValueError:
        logger.error(f"Promoted ID {promoted_id} not found in catalog.")
        return ["unrelated"] * len(catalog)

    # Vectorized similarity calculation
    embeddings = _get_embeddings(names)
    promoted_emb = embeddings[p_idx]
    emb_scores = np.dot(embeddings, promoted_emb)

    p_name, p_cat = names[p_idx], cats[p_idx]
    results = []
    
    for i in range(len(catalog)):
        if i == p_idx:
            results.append("promoted")
        else:
            rel = _classify_pair(p_name, names[i], p_cat, cats[i], float(emb_scores[i]))
            results.append(rel)

    return results

def relationship_score_matrix(catalog: list[dict]) -> pd.DataFrame:
    """
    N×N blended similarity score matrix with category-aware logic.
    Used during initial data ingestion.
    """
    names = [str(s.get("name", s.get("id", ""))) for s in catalog]
    cats  = [str(s.get("category", "Unknown")) for s in catalog]
    ids   = [str(s.get("id", s.get("sku", i))) for i, s in enumerate(catalog)]
    n = len(catalog)

    embeddings = _get_embeddings(names)
    emb_matrix = np.dot(embeddings, embeddings.T)

    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0 
        for j in range(i + 1, n):
            # Guard ensures cross-category scores stay at zero
            if cats[i].lower() == cats[j].lower():
                seq_s = _sequence_sim(names[i], names[j])
                emb_s = float(emb_matrix[i, j])
                matrix[i, j] = matrix[j, i] = (_WEIGHTS["sequence"] * seq_s) + (_WEIGHTS["embedding"] * emb_s)
            else:
                matrix[i, j] = matrix[j, i] = 0.0

    df = pd.DataFrame(matrix, index=ids, columns=ids)
    df.attrs["embedding_backend"] = "huggingface"
    df.attrs["embedding_model"] = get_embedding_model()
    return df

def get_embedding_backend() -> str:
    return "huggingface"